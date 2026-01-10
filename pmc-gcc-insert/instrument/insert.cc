// instrument_callsites_plugin.cc
//
// GCC plugin that wraps callsites to "target" functions with:
//
//   pmc_multi_handle_t *handle = pmc_measure_begin_csv(label, csv_path);
//   callee(...);
//   pmc_measure_end(handle, report);
//
// BUT it does NOT wrap calls made from inside a target function.
// This matches: measure b() when called from main/non-target,
// but do NOT measure b() when called from inside target a().
//
// Build (example):
//   g++ -shared -fPIC -O2 -std=c++17 \
//     -I$(gcc -print-file-name=plugin)/include \
//     instrument_callsites_plugin.cc -o instrument_callsites_plugin.so
//
// Use (example):
//   gcc -O2 -fplugin=./instrument_callsites_plugin.so \
//     -fplugin-arg-instrument_callsites_plugin-include-function-list=b,c \
//     -fplugin-arg-instrument_callsites_plugin-include-file-list=aes.c,sha256.c \
//     -fplugin-arg-instrument_callsites_plugin-csv-path=pmc_events.csv \
//     test.c -o test -L/path/to/libpmc -lpmc
//
// Optional file filtering:
//   - If include-file-list is provided, only those source files are instrumented
//   - Supports suffix matching (e.g., "aes.c" matches "src/crypto/aes.c")
//   - If omitted, all files are instrumented
//
// Results are always exported to pmc_results.json (or $PMC_OUTPUT_FILE)
//
// Notes:
// - Works on direct calls (gimple_call_fndecl != NULL). Function-pointer/virtual calls won't match.
// - If a target call can throw (C++ exceptions), the inserted "end" after the call won't run on the
//   exceptional edge. Handling EH correctly is more involved.
// - Integrates with libpmc: pmc_measure_begin_csv() / pmc_measure_end() API
//

#include "gcc-plugin.h"
#include "plugin-version.h"

#include "context.h"
#include "tree.h"
#include "tree-pass.h"
#include "function.h"
#include "stringpool.h"
#include "attribs.h"

#include "gimple.h"
#include "gimple-iterator.h"
#include "basic-block.h"
#include "cgraph.h"
#include "ssa.h"

#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <set>
#include <map>

int plugin_is_GPL_compatible = 1;

// ---- plugin args ----
static const char* ARG_DEBUG = "debug";
static const char* ARG_INCLUDE_FUNCTION_LIST = "include-function-list";
static const char* ARG_INCLUDE_FILE_LIST = "include-file-list";
static const char* ARG_CSV_PATH = "csv-path";

// default pmc function names (matching libpmc API)
static const char* PMC_BEGIN_CSV_NAME = "pmc_measure_begin_csv";
static const char* PMC_END_NAME = "pmc_measure_end";

// CSV path for pmc_measure_begin_csv (NULL means use default "pmc_events.csv")
static std::string g_csv_path = "";
// Always enable reporting (export to JSON)
static const int g_report = 1;

static bool g_debug = false;

static std::vector<std::string> g_targets;
static std::vector<std::string> g_file_whitelist;  // Optional file filter

// Set of function names that are transitively reachable from target functions.
// We don't instrument calls FROM functions in this set (to avoid nested measurements).
static std::set<std::string> g_reachable_from_targets;

// Call graph built from GIMPLE analysis (caller -> set of callees)
static std::map<std::string, std::set<std::string>> g_call_graph;

// Cached external function declarations (built once, reused)
static tree g_pmc_begin_csv_decl = NULL_TREE;
static tree g_pmc_end_decl = NULL_TREE;

// Track if we've computed the reachable set (done at FINISH_UNIT)
static bool g_reachable_set_computed = false;

// ---- helpers ----
static const char* decl_name(tree fndecl) {
  if (!fndecl) return nullptr;
  tree n = DECL_NAME(fndecl);
  return n ? IDENTIFIER_POINTER(n) : nullptr;
}

static const char* decl_asm_name(tree fndecl) {
  if (!fndecl) return nullptr;
  tree n = DECL_ASSEMBLER_NAME(fndecl);
  return n ? IDENTIFIER_POINTER(n) : nullptr;
}

static bool str_eq(const char* a, const char* b) {
  return a && b && std::strcmp(a, b) == 0;
}

// Check if a file path matches the whitelist
// Supports both exact matches and suffix matches (e.g., "aes.c" matches "src/crypto/aes.c")
static bool file_matches_whitelist(const char* filepath) {
  if (!filepath) return false;
  if (g_file_whitelist.empty()) return true;  // No filter = allow all
  
  for (const auto& pattern : g_file_whitelist) {
    // Check if filepath ends with pattern (allows matching subdirectories)
    size_t filepath_len = std::strlen(filepath);
    size_t pattern_len = pattern.length();
    
    if (pattern_len <= filepath_len) {
      const char* suffix = filepath + (filepath_len - pattern_len);
      if (std::strcmp(suffix, pattern.c_str()) == 0) {
        return true;
      }
      
      // Also check if there's a path separator before the match
      if (pattern_len < filepath_len && 
          (filepath[filepath_len - pattern_len - 1] == '/' || 
           filepath[filepath_len - pattern_len - 1] == '\\')) {
        if (std::strcmp(suffix, pattern.c_str()) == 0) {
          return true;
        }
      }
    }
  }
  
  return false;
}

static bool is_target_function(tree fndecl) {
  if (!fndecl || TREE_CODE(fndecl) != FUNCTION_DECL) return false;

  // Never treat pmc wrapper functions as targets.
  // (Prevents accidental self-wrapping if user adds them.)
  const char* n  = decl_name(fndecl);
  const char* an = decl_asm_name(fndecl);
  if (str_eq(n, PMC_BEGIN_CSV_NAME) || str_eq(an, PMC_BEGIN_CSV_NAME) ||
      str_eq(n, PMC_END_NAME) || str_eq(an, PMC_END_NAME)) {
    return false;
  }

  const char* name = decl_name(fndecl);
  const char* asmname = decl_asm_name(fndecl);

  for (const auto& t : g_targets) {
    if ((name && t == name) || (asmname && t == asmname)) {
      return true;
    }
  }
  return false;
}

// Check if a function is noreturn (exit, abort, etc.)
// We skip instrumenting calls to noreturn functions because the "end" call would be unreachable
static bool is_noreturn_function(tree fndecl) {
  if (!fndecl) return false;
  
  // Check TREE_THIS_VOLATILE flag (set for noreturn)
  if (TREE_THIS_VOLATILE(fndecl)) {
    return true;
  }
  
  // Check for explicit noreturn attribute
  if (lookup_attribute("noreturn", DECL_ATTRIBUTES(fndecl))) {
    return true;
  }
  
  return false;
}

// Scan a function's GIMPLE and record all direct calls it makes
static void record_function_calls(function* fun) {
  if (!fun || !fun->decl) return;
  
  const char* caller_name = decl_name(fun->decl);
  if (!caller_name) return;
  
  std::set<std::string>& callees = g_call_graph[caller_name];
  
  basic_block bb;
  FOR_EACH_BB_FN(bb, fun) {
    for (gimple_stmt_iterator gsi = gsi_start_bb(bb); !gsi_end_p(gsi); gsi_next(&gsi)) {
      gimple* stmt = gsi_stmt(gsi);
      if (is_gimple_call(stmt)) {
        tree callee_decl = gimple_call_fndecl(stmt);
        if (callee_decl) {
          const char* callee_name = decl_name(callee_decl);
          if (callee_name) {
            callees.insert(callee_name);
          }
        }
      }
    }
  }
}

// Recursively collect all functions reachable from a given function using our call graph
static void collect_reachable_from_call_graph(const std::string& fn, std::set<std::string>& visited) {
  // Already visited?
  if (visited.count(fn)) return;
  visited.insert(fn);
  
  // Find callees in our call graph and recurse
  auto it = g_call_graph.find(fn);
  if (it == g_call_graph.end()) return;
  
  for (const auto& callee : it->second) {
    collect_reachable_from_call_graph(callee, visited);
  }
}

// Compute reachable set from our own call graph
static void compute_reachable_set() {
  g_reachable_from_targets.clear();
  
  if (g_targets.empty()) return;
  
  if (g_debug) {
    fprintf(stderr, "\n╔════════════════════════════════════════════════════════════════\n");
    fprintf(stderr, "║ Call Graph Analysis (Direct Calls Only)\n");
    fprintf(stderr, "╚════════════════════════════════════════════════════════════════\n");
    
    // Show direct calls for each target
    for (const auto& target : g_targets) {
      fprintf(stderr, "  %s:\n", target.c_str());
      auto it = g_call_graph.find(target);
      if (it != g_call_graph.end() && !it->second.empty()) {
        for (const auto& callee : it->second) {
          fprintf(stderr, "    └─> %s\n", callee.c_str());
        }
      } else {
        fprintf(stderr, "    (no calls recorded)\n");
      }
    }
  }
  
  // Compute transitive closure: collect all functions reachable from targets
  for (const auto& target : g_targets) {
    // Check if this target is in our call graph
    if (g_call_graph.find(target) != g_call_graph.end()) {
      collect_reachable_from_call_graph(target, g_reachable_from_targets);
    } else {
      // Target exists but we haven't seen its body yet, just add it
      g_reachable_from_targets.insert(target);
    }
  }
}


static void split_csv_into(const char* csv, std::vector<std::string>& out) {
  if (!csv) return;
  const char* p = csv;
  while (*p) {
    while (*p == ' ' || *p == '\t' || *p == ',') p++;
    if (!*p) break;

    const char* start = p;
    while (*p && *p != ',') p++;
    const char* end = p;

    while (end > start && (end[-1] == ' ' || end[-1] == '\t')) end--;
    if (end > start) out.emplace_back(start, static_cast<size_t>(end - start));

    if (*p == ',') p++;
  }
}

static void parse_plugin_args(plugin_name_args* plugin_info) {
  static bool config_printed = false;  // Only print config once
  
  for (int i = 0; i < plugin_info->argc; ++i) {
    const plugin_argument& a = plugin_info->argv[i];
    if (!a.key) continue;

    if (std::strcmp(a.key, ARG_DEBUG) == 0) {
      g_debug = true;
    } else if (std::strcmp(a.key, ARG_INCLUDE_FUNCTION_LIST) == 0 && a.value) {
      split_csv_into(a.value, g_targets);
    } else if (std::strcmp(a.key, ARG_INCLUDE_FILE_LIST) == 0 && a.value) {
      split_csv_into(a.value, g_file_whitelist);
    } else if (std::strcmp(a.key, ARG_CSV_PATH) == 0 && a.value) {
      g_csv_path = a.value;
    }
  }

  if (g_debug && !config_printed) {
    config_printed = true;
    fprintf(stderr, "\n╔════════════════════════════════════════════════════════════════\n");
    fprintf(stderr, "║ PMC Instrumentation Plugin Configuration\n");
    fprintf(stderr, "╠════════════════════════════════════════════════════════════════\n");
    fprintf(stderr, "║ API: %s() / %s()\n", PMC_BEGIN_CSV_NAME, PMC_END_NAME);
    fprintf(stderr, "║ CSV: %s\n", g_csv_path.empty() ? "pmc_events.csv (default)" : g_csv_path.c_str());
    fprintf(stderr, "║ Output: pmc_results.json (always enabled)\n");
    fprintf(stderr, "║ File Filter: %s\n", g_file_whitelist.empty() ? "none (all files)" : "enabled");
    if (!g_file_whitelist.empty()) {
      fprintf(stderr, "║   Files (%zu):\n", g_file_whitelist.size());
      for (auto& f : g_file_whitelist) {
        fprintf(stderr, "║     • %s\n", f.c_str());
      }
    }
    fprintf(stderr, "║ Targets (%zu):\n", g_targets.size());
    for (auto& t : g_targets) {
      fprintf(stderr, "║   • %s\n", t.c_str());
    }
    fprintf(stderr, "╚════════════════════════════════════════════════════════════════\n");
  }
}

// Build external decl for pmc_measure_begin_csv:
//   void* pmc_measure_begin_csv(const char* label, const char* csv_path);
static tree build_pmc_begin_csv_decl() {
  // const char* is pointer to const char
  tree const_char_ptr = build_pointer_type(build_qualified_type(char_type_node, TYPE_QUAL_CONST));
  // Return type is void* (pmc_multi_handle_t*)
  tree fntype = build_function_type_list(ptr_type_node, const_char_ptr, const_char_ptr, NULL_TREE);
  tree fndecl = build_fn_decl(PMC_BEGIN_CSV_NAME, fntype);
  DECL_EXTERNAL(fndecl) = 1;
  TREE_PUBLIC(fndecl) = 1;
  return fndecl;
}

// Build external decl for pmc_measure_end:
//   void pmc_measure_end(void* handle, int report);
static tree build_pmc_end_decl() {
  tree fntype = build_function_type_list(void_type_node, ptr_type_node, integer_type_node, NULL_TREE);
  tree fndecl = build_fn_decl(PMC_END_NAME, fntype);
  DECL_EXTERNAL(fndecl) = 1;
  TREE_PUBLIC(fndecl) = 1;
  return fndecl;
}

// Get cached pmc_measure_begin_csv decl (build once, reuse)
static tree get_pmc_begin_csv_decl() {
  if (!g_pmc_begin_csv_decl) {
    g_pmc_begin_csv_decl = build_pmc_begin_csv_decl();
  }
  return g_pmc_begin_csv_decl;
}

// Get cached pmc_measure_end decl (build once, reuse)
static tree get_pmc_end_decl() {
  if (!g_pmc_end_decl) {
    g_pmc_end_decl = build_pmc_end_decl();
  }
  return g_pmc_end_decl;
}

// Build a string literal that can be used in GIMPLE
// Uses GCC's fold_build_string_literal approach for correctness
static tree build_string_literal_ptr(const char* str) {
  size_t len = std::strlen(str) + 1;
  
  // Create the string constant node
  tree str_cst = build_string(len, str);
  
  // Create the proper array type for the string
  tree elem_type = char_type_node;
  tree index_type = build_index_type(size_int(len - 1));
  tree array_type = build_array_type(elem_type, index_type);
  
  TREE_TYPE(str_cst) = array_type;
  TREE_CONSTANT(str_cst) = 1;
  TREE_READONLY(str_cst) = 1;
  TREE_STATIC(str_cst) = 1;
  
  // Build pointer type for const char*
  tree const_char_ptr = build_pointer_type(
    build_qualified_type(char_type_node, TYPE_QUAL_CONST));
  
  // Create ADDR_EXPR to get pointer to the string
  tree addr = build1(ADDR_EXPR, const_char_ptr, str_cst);
  TREE_CONSTANT(addr) = 1;
  TREE_STATIC(addr) = 1;
  
  return addr;
}

// Insert: handle = pmc_measure_begin_csv(label, csv_path);
// Returns SSA name or VAR_DECL for the handle (SSA-safe)
static tree insert_pmc_begin_before(gimple_stmt_iterator* gsi, gimple* anchor_stmt, 
                                     tree pmc_begin_decl, tree callee_decl) {
  const char* callee_name = decl_name(callee_decl);
  if (!callee_name) callee_name = "<unknown>";
  
  // Build string arguments
  tree label_arg = build_string_literal_ptr(callee_name);
  tree csv_path_arg;
  if (g_csv_path.empty()) {
    csv_path_arg = null_pointer_node;
  } else {
    csv_path_arg = build_string_literal_ptr(g_csv_path.c_str());
  }
  
  // Create temporary for handle
  // Note: We use create_tmp_var which is safe for pre-SSA or will be converted to SSA
  // by returning TODO_update_ssa from the pass
  tree handle_lhs = create_tmp_var(ptr_type_node, "pmc_handle");
  
  // Build call: handle = pmc_measure_begin_csv(label, csv_path)
  gcall* call = gimple_build_call(pmc_begin_decl, 2, label_arg, csv_path_arg);
  gimple_call_set_lhs(call, handle_lhs);
  gimple_set_location(call, gimple_location(anchor_stmt));
  gsi_insert_before(gsi, call, GSI_SAME_STMT);
  
  return handle_lhs;
}

// Insert: pmc_measure_end(handle, report); after the current statement
static void insert_pmc_end_after(gimple_stmt_iterator gsi, gimple* anchor_stmt, 
                                  tree pmc_end_decl, tree handle_tmp) {
  tree report_arg = build_int_cst(integer_type_node, g_report);
  
  gcall* call = gimple_build_call(pmc_end_decl, 2, handle_tmp, report_arg);
  gimple_set_location(call, gimple_location(anchor_stmt));
  // Use GSI_SAME_STMT to not modify the passed-by-value iterator
  gsi_insert_after(&gsi, call, GSI_SAME_STMT);
}

// Core instrumentation logic: instrument target function calls in a given function
// Returns true if any instrumentation was done
static bool instrument_function_calls(function* fun) {
  if (!fun || !fun->decl) return false;
  
  const char* fun_name = decl_name(fun->decl);
  if (!fun_name) return false;
  
  // Check if this function is in the exclusion zone
  if (g_reachable_from_targets.count(fun_name)) {
    if (g_debug) {
      bool is_direct_target = is_target_function(fun->decl);
      if (is_direct_target) {
        fprintf(stderr, "⊗ SKIP [TARGET]: %s\n", fun_name);
      } else {
        fprintf(stderr, "⊗ SKIP [TRANSITIVE]: %s (called by target)\n", fun_name);
      }
      
      // Report any target calls inside that we're skipping
      basic_block bb;
      FOR_EACH_BB_FN(bb, fun) {
        for (gimple_stmt_iterator gsi = gsi_start_bb(bb); !gsi_end_p(gsi); gsi_next(&gsi)) {
          gimple* stmt = gsi_stmt(gsi);
          if (is_gimple_call(stmt)) {
            tree callee = gimple_call_fndecl(stmt);
            if (callee && is_target_function(callee)) {
              const char* callee_n = decl_name(callee);
              location_t loc = gimple_location(stmt);
              expanded_location xloc = expand_location(loc);
              fprintf(stderr, "    └─> %s() calls %s() [nested - skipped]\n",
                      fun_name,
                      callee_n ? callee_n : "<unnamed>");
              fprintf(stderr, "        at %s:%d\n",
                      xloc.file ? xloc.file : "<unknown>",
                      xloc.line);
            }
          }
        }
      }
    }
    return false;
  }

  tree pmc_begin_decl = get_pmc_begin_csv_decl();
  tree pmc_end_decl = get_pmc_end_decl();
  bool instrumented = false;

  basic_block bb;
  FOR_EACH_BB_FN(bb, fun) {
    for (gimple_stmt_iterator gsi = gsi_start_bb(bb); !gsi_end_p(gsi); ) {
      gimple* stmt = gsi_stmt(gsi);

      if (is_gimple_call(stmt)) {
        tree callee = gimple_call_fndecl(stmt);

        // Direct call to a known decl?
        if (callee && is_target_function(callee)) {
          // Skip noreturn functions (end call would be unreachable)
          if (is_noreturn_function(callee)) {
            if (g_debug) {
              const char* callee_n = decl_name(callee);
              fprintf(stderr, "⊗ SKIP [NORETURN]: %s() → %s()\n",
                      fun_name, callee_n ? callee_n : "<unnamed>");
            }
            gsi_next(&gsi);
            continue;
          }
          
          if (g_debug) {
            const char* callee_n = decl_name(callee);
            location_t loc = gimple_location(stmt);
            expanded_location xloc = expand_location(loc);
            fprintf(stderr, "✓ INSTRUMENT: %s() → %s()\n",
                    fun_name,
                    callee_n ? callee_n : "<unnamed>");
            fprintf(stderr, "    └─ Location: %s:%d\n", 
                    xloc.file ? xloc.file : "<unknown>",
                    xloc.line);
          }

          // Insert: handle = pmc_measure_begin_csv(label, csv_path);
          tree handle_tmp = insert_pmc_begin_before(&gsi, stmt, pmc_begin_decl, callee);

          // Insert: pmc_measure_end(handle, report); after the target call
          insert_pmc_end_after(gsi, stmt, pmc_end_decl, handle_tmp);

          instrumented = true;
          gsi_next(&gsi); // past original call
          gsi_next(&gsi); // past inserted end call
          continue;
        }
      }

      gsi_next(&gsi);
    }
  }

  return instrumented;
}

// ---- Pass 1: Build call graph ----
static const pass_data pass_data_build_call_graph = {
  GIMPLE_PASS,
  "build_pmc_call_graph",
  OPTGROUP_NONE,
  TV_NONE,
  PROP_cfg,
  0, 0, 0, 0
};

class pass_build_call_graph final : public gimple_opt_pass {
public:
  pass_build_call_graph(gcc::context* ctxt)
    : gimple_opt_pass(pass_data_build_call_graph, ctxt) {}

  unsigned int execute(function* fun) final override {
    if (!fun || !fun->decl) return 0;
    if (g_targets.empty()) return 0;

    // Check file filter
    const char* source_file = DECL_SOURCE_FILE(fun->decl);
    if (!file_matches_whitelist(source_file)) {
      if (g_debug && source_file) {
        const char* fun_name = decl_name(fun->decl);
        fprintf(stderr, "⊘ SKIP FILE: %s (function: %s)\n", 
                source_file, fun_name ? fun_name : "<unnamed>");
      }
      return 0;
    }

    // Record all calls this function makes
    record_function_calls(fun);

    return 0;
  }
};

// ---- Instrumentation pass: Compute reachability once, then instrument each function ----
// This runs late in the pipeline after call graph is complete
static const pass_data pass_data_instrument_calls = {
  GIMPLE_PASS,
  "pmc_instrument",
  OPTGROUP_NONE,
  TV_NONE,
  PROP_cfg | PROP_ssa,
  0, 0, 0, 0
};

class pass_instrument_calls final : public gimple_opt_pass {
public:
  pass_instrument_calls(gcc::context* ctxt)
    : gimple_opt_pass(pass_data_instrument_calls, ctxt) {}

  unsigned int execute(function* fun) final override {
    if (!fun || !fun->decl) return 0;
    if (g_targets.empty()) return 0;
    
    // Check file filter first
    const char* source_file = DECL_SOURCE_FILE(fun->decl);
    if (!file_matches_whitelist(source_file)) {
      return 0;
    }
    
    // Step 1: Compute reachable set ONCE (first time any function reaches here)
    if (!g_reachable_set_computed) {
      g_reachable_set_computed = true;
      compute_reachable_set();
    
      if (g_debug) {
        // Separate targets from non-targets in the reachable set
        std::set<std::string> targets_set(g_targets.begin(), g_targets.end());
        std::set<std::string> transitive_nontargets;
        
        for (const auto& fn : g_reachable_from_targets) {
          if (targets_set.count(fn) == 0) {
            transitive_nontargets.insert(fn);
          }
        }
        
        fprintf(stderr, "\n╔════════════════════════════════════════════════════════════════\n");
        fprintf(stderr, "║ Instrumentation Strategy (Call Graph Complete)\n");
        fprintf(stderr, "╠════════════════════════════════════════════════════════════════\n");
        fprintf(stderr, "║ EXCLUSION ZONE (calls FROM these functions won't be instrumented):\n");
        fprintf(stderr, "║\n");
        fprintf(stderr, "║   TARGETS (skip their internal calls): %zu\n", targets_set.size());
        for (const auto& fn : targets_set) {
          fprintf(stderr, "║     • %s\n", fn.c_str());
        }
        
        if (!transitive_nontargets.empty()) {
          fprintf(stderr, "║\n");
          fprintf(stderr, "║   NON-TARGETS (transitively reachable from targets): %zu\n", transitive_nontargets.size());
          for (const auto& fn : transitive_nontargets) {
            fprintf(stderr, "║     • %s\n", fn.c_str());
          }
        }
        
        fprintf(stderr, "║\n");
        fprintf(stderr, "║ RESULT:\n");
        fprintf(stderr, "║   • Targets WILL be measured when called from outside exclusion zone\n");
        fprintf(stderr, "║   • Calls FROM exclusion zone are NOT instrumented (avoids nesting)\n");
        fprintf(stderr, "╚════════════════════════════════════════════════════════════════\n\n");
      }
    }
    
    // Step 2: Instrument this function
    bool did_instrument = instrument_function_calls(fun);
    
    // Return TODO_update_ssa if we instrumented (for SSA correctness)
    return did_instrument ? TODO_update_ssa : 0;
  }
};

// ---- plugin init ----
extern "C" int plugin_init(plugin_name_args* plugin_info, plugin_gcc_version* version) {
  if (!plugin_default_version_check(version, &gcc_version)) {
    fprintf(stderr, "[instrument_callsites] GCC version mismatch\n");
    return 1;
  }

  parse_plugin_args(plugin_info);

  // Pass 1: Build call graph (runs early, after cfg)
  // This pass scans each function and records direct calls
  auto* pass1 = new pass_build_call_graph(g);
  register_pass_info pass1_info;
  pass1_info.pass = pass1;
  pass1_info.reference_pass_name = "cfg";
  pass1_info.ref_pass_instance_number = 1;
  pass1_info.pos_op = PASS_POS_INSERT_AFTER;
  register_callback(plugin_info->base_name, PLUGIN_PASS_MANAGER_SETUP, nullptr, &pass1_info);

  // Pass 2: Late instrumentation pass
  // Runs after Pass 1 has built call graph for all functions
  // Uses "optimized" as reference point - late enough that all functions have been seen
  auto* pass2 = new pass_instrument_calls(g);
  register_pass_info pass2_info;
  pass2_info.pass = pass2;
  pass2_info.reference_pass_name = "optimized";
  pass2_info.ref_pass_instance_number = 1;
  pass2_info.pos_op = PASS_POS_INSERT_BEFORE;
  register_callback(plugin_info->base_name, PLUGIN_PASS_MANAGER_SETUP, nullptr, &pass2_info);

  return 0;
}
