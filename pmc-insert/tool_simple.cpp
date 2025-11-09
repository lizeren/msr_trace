#include "clang/AST/ASTContext.h"
#include "clang/AST/Expr.h"
#include "clang/AST/Stmt.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Lex/Lexer.h"
#include "clang/Rewrite/Core/Rewriter.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/Support/CommandLine.h"
#include <unordered_set>
#include <string>
#include <memory>
#include <fstream>
#include <sstream>
#include <vector>

using namespace clang;
using namespace clang::tooling;
using namespace clang::ast_matchers;

//  OptionCategory lets us hide all the default clang options displayed in the help, we don't want those for our tool.
static llvm::cl::OptionCategory Cat("pmc-insert options");

static llvm::cl::opt<std::string> ApiListCSV(
  "api-list", llvm::cl::desc("CSV file containing function names to wrap (one per line)"),
  llvm::cl::Required, llvm::cl::cat(Cat));

static llvm::cl::opt<std::string> IncludeHeader(
  "include-header", llvm::cl::desc("Header to insert once per file (optional)"),
  llvm::cl::init(""), llvm::cl::cat(Cat));

// Function to read target function names from CSV file
static std::unordered_set<std::string> readApiListFromCSV(const std::string& filepath) {
  std::unordered_set<std::string> targets;
  std::ifstream file(filepath);
  
  if (!file.is_open()) {
    llvm::errs() << "Error: Cannot open CSV file: " << filepath << "\n";
    return targets;
  }
  
  std::string line;
  while (std::getline(file, line)) {
    // Trim whitespace and commas
    size_t start = line.find_first_not_of(" \t\r\n");
    if (start == std::string::npos) continue; // Skip empty lines
    
    size_t end = line.find_first_of(",\r\n", start);
    if (end == std::string::npos) end = line.length();
    
    std::string funcName = line.substr(start, end - start);
    
    // Trim trailing whitespace
    size_t lastNonSpace = funcName.find_last_not_of(" \t\r\n");
    if (lastNonSpace != std::string::npos) {
      funcName = funcName.substr(0, lastNonSpace + 1);
    }
    
    if (!funcName.empty()) {
      targets.insert(funcName);
    }
  }
  
  file.close();
  return targets;
}

namespace {

struct State {
// Tracks all source code modifications (insertions, replacements, deletions)
// Accumulates changes in memory without modifying the actual file yet
// At the end, R.overwriteChangedFiles() writes everything to disk
  Rewriter R;
  std::unordered_set<std::string> Targets;
  std::unordered_set<const FileEntry*> IncludedFiles; // Tracks which files have already had the header included
  unsigned TempCounter = 0;
  std::string genTemp(const char* base = "__pmc_tmp") {
    return (std::string(base) + "_" + std::to_string(++TempCounter));
  }
};

static const Expr* peel(const Expr* E) {
  for(;;) {
    if (auto P = dyn_cast_or_null<ParenExpr>(E)) { E = P->getSubExpr(); continue; }
    if (auto C = dyn_cast_or_null<ImplicitCastExpr>(E)) { E = C->getSubExpr(); continue; }
    if (auto C = dyn_cast_or_null<ExprWithCleanups>(E)) { E = C->getSubExpr(); continue; }
    break;
  }
  return E;
}

static std::string getText(const SourceManager& SM, const LangOptions& LO, SourceRange SR) {
  return Lexer::getSourceText(CharSourceRange::getTokenRange(SR), SM, LO).str();
}

static SourceRange rangeWithTrailingSemi(const Stmt* S, const ASTContext& Ctx) {
  const SourceManager& SM = Ctx.getSourceManager();
  SourceLocation begin = S->getBeginLoc();
  SourceLocation afterSemi = Lexer::findLocationAfterToken(
      S->getEndLoc(), tok::semi, SM, Ctx.getLangOpts(), true);
  if (afterSemi.isInvalid()) return SourceRange();
  return SourceRange(begin, afterSemi);
}


// For each CallExpr found, Callback::run() is invoked
class Callback : public MatchFinder::MatchCallback {
public:
  explicit Callback(State& S) : S(S) {} // Constructor

  void run(const MatchFinder::MatchResult& Res) override {
    const auto* CE = Res.Nodes.getNodeAs<CallExpr>("call"); // Get the CallExpr node
    if (!CE) return;

    // CE->dump();

    const SourceManager& SM = *Res.SourceManager; // this contains the source code of the file
    
    const FunctionDecl* FD = CE->getDirectCallee();
    if (!FD) return; // ignore indirects
    
    std::string fname = FD->getName().str();
    
    // Skip PMC library functions to avoid re-instrumenting instrumentation
    if (fname == "pmc_measure_begin_csv" || fname == "pmc_measure_begin" ||
        fname == "pmc_measure_end" || fname == "pmc_report_all" ||
        fname == "pmc_get_count" || fname == "pmc_get_samples") {
      return;
    }
    
    if (!S.Targets.count(fname)) return;

    auto& Ctx = *Res.Context; // ASTContext object for accessing the AST
    auto& R   = S.R; // Rewriter object for modifying source code
    LangOptions LO = Ctx.getLangOpts(); // Language options for accessing the language options

    // Insert header once
    // If the user doesn't provide a header, skip the block below
    if (!IncludeHeader.empty()) {
      FileID FID = SM.getMainFileID(); // FID is a unique identifier for the current file being processed
      const FileEntry* FE = SM.getFileEntryForID(FID); // get FileEntry: information about one file (name, size, etc.)
      if (FE && !S.IncludedFiles.count(FE)) {
        R.InsertText(SM.getLocForStartOfFile(FID),
                     "#include \"" + IncludeHeader + "\"\n"); // Insert the #include at the start of the file
        S.IncludedFiles.insert(FE); // Mark this file as processed
      }
    }

    // Helpers - use unique variable names to avoid redefinition
    auto callText = getText(SM, LO, CE->getSourceRange());
    std::string hvar = S.genTemp(("__pmc_h_" + fname).c_str());
    unsigned counterNum = S.TempCounter; // Get the current counter for this call
    std::string callLabel = fname + "_" + std::to_string(counterNum);
    std::string begin = "pmc_multi_handle_t* " + hvar + " = pmc_measure_begin_csv(\"" + callLabel + "\", NULL); ";
    std::string end   = "pmc_measure_end(" + hvar + ", 1); ";

    // Try different patterns based on parent context
    
    // Walk up parent chain to find context
    // Start from the CallExpr and walk up the AST
    const BinaryOperator* FoundAssign = nullptr;
    const VarDecl* FoundVarDecl = nullptr;
    const ReturnStmt* FoundReturn = nullptr;
    
    // Walk up through parents to find BinaryOperator, VarDecl, or ReturnStmt
    auto Parents = Ctx.getParents(*CE);
    for (int depth = 0; depth < 10 && !Parents.empty(); depth++) {
      
      // We start checking the most complicated case first

      // Check for variable declaration
      if (const VarDecl* VD = Parents[0].get<VarDecl>()) {
        FoundVarDecl = VD;
        break;
      }


      // Check for assignment  
      if (const BinaryOperator* BO = Parents[0].get<BinaryOperator>()) {
        if (BO->getOpcode() == BO_Assign) {
          FoundAssign = BO;
          break;
        }
      }

      // Check for return statement
      if (const ReturnStmt* RS = Parents[0].get<ReturnStmt>()) {
        FoundReturn = RS;
        break;
      }
      
      
      // Move up to next parent
      Parents = Ctx.getParents(Parents[0]);
    }
    
    // 2) Handle assignment statement: `x = foo();`
    // Strategy: Insert "begin" before assignment, insert "end" after semicolon
    if (FoundAssign) {
      // Use the assignment's own location (don't walk up to avoid finding CompoundStmt)

      /*
      x = foo()
      ^       ^
      |       |
      |       assignEnd (points to START of last token ')')
      assignStart (points to 'x')
      */
      SourceLocation assignStart = FoundAssign->getBeginLoc(); 
      SourceLocation assignEnd = FoundAssign->getEndLoc(); // points to START of last token ')' of the expression
      
      // Insert "begin" at start of assignment
      R.InsertTextBefore(assignStart, begin);
      
      // we cannot directly insert assignafter becuase that would place the code inbetween foo() and ;
      // Find semicolon after the assignment and insert "end" after it
      SourceLocation afterToken = Lexer::getLocForEndOfToken(assignEnd, 0, SM, LO); // points after )
      
      // Scan forward for semicolon (usually very close, within 10 chars)
      for (unsigned offset = 0; offset < 20; offset++) {
        SourceLocation testLoc = afterToken.getLocWithOffset(offset);
        if (testLoc.isInvalid()) break;
        
        const char* charData = SM.getCharacterData(testLoc);
        if (*charData == ';') {
          R.InsertTextAfter(testLoc.getLocWithOffset(1), " " + end);
          return;  // Success!
        }
        // Skip whitespace
        if (*charData != ' ' && *charData != '\t' && *charData != '\n' && *charData != '\r') {
          // Hit non-whitespace, non-semicolon - give up
          break;
        }
      }
      
      // If we reach here, couldn't find semicolon - return anyway to prevent fallthrough
      return;
    }
    
    // 3) Handle declaration with initializer: `T x = foo();`
    if (FoundVarDecl) {
      // the parent of the VarDecl is the DeclStmt
      auto ParentsVD = Ctx.getParents(*FoundVarDecl);
      // usually there is only one parent, but a node might have multiple "logical" parents. so...
      for (auto& P : ParentsVD) {
        if (const DeclStmt* DS = P.get<DeclStmt>()) {
          // Find semicolon after the declaration statement
          SourceLocation DSStart = DS->getBeginLoc();
          SourceLocation DSEnd = DS->getEndLoc();
          
          SourceLocation afterSemi;
          
          // Scan forward from DSEnd itself (might already be at/near the semicolon)
          for (int offset = -5; offset < 30; offset++) {
            SourceLocation testLoc = DSEnd.getLocWithOffset(offset);
            if (testLoc.isInvalid()) continue;
            
            const char* charData = SM.getCharacterData(testLoc);
            if (*charData == ';') {
              afterSemi = testLoc.getLocWithOffset(1);
              break;
            }
          }
          if (afterSemi.isValid() && DSStart.isValid()) {
            SourceRange RR(DSStart, afterSemi);
            std::string T = FoundVarDecl->getType().getAsString(Ctx.getPrintingPolicy()); // variable type
            std::string name = FoundVarDecl->getName().str(); // variable name
            std::string repl = begin + T + " " + name + " = (" + callText + "); " + end ;
            R.ReplaceText(RR, repl);
            return;
          }
          // Don't fall through to case 1 if we're in a VarDecl
          return;
        }
      }
      // Found VarDecl but no DeclStmt - unusual but prevent fallthrough
      return;
    }

    // 4) Handle return statement: `return foo();`
    if (FoundReturn) {
      // Find semicolon after the return statement
      SourceLocation RSStart = FoundReturn->getBeginLoc();
      SourceLocation RSEnd = FoundReturn->getEndLoc();
      
      SourceLocation afterSemi;
      
      // Scan forward from RSEnd to find the semicolon
      for (int offset = -5; offset < 30; offset++) {
        SourceLocation testLoc = RSEnd.getLocWithOffset(offset);
        if (testLoc.isInvalid()) continue;
        
        const char* charData = SM.getCharacterData(testLoc);
        if (*charData == ';') {
          afterSemi = testLoc.getLocWithOffset(1);
          break;
        }
      }
      
      if (afterSemi.isValid() && RSStart.isValid()) {
        SourceRange RR(RSStart, afterSemi);
        
        // Get the return type from the function we're in
        // We need to create a temporary to store the result
        std::string tmp = S.genTemp();
        
        // For now, use 'auto' for the return type (C++11)
        // In C, we'd need to infer the type from the CallExpr
        QualType RetType = CE->getType();
        std::string T = RetType.getAsString(Ctx.getPrintingPolicy());
        
        std::string repl =  begin + T + " " + tmp + " = (" + callText + "); " 
                           + end + "return " + tmp + ";";
        R.ReplaceText(RR, repl);
        return;
      }
      
      // Don't fall through if we're in a return statement
      return;
    }

    // 1) Expression statement: `foo();` (fallback for simple calls)
    // Find the semicolon after the call
    SourceLocation afterSemi = Lexer::findLocationAfterToken(
        CE->getEndLoc(), tok::semi, SM, LO, false);
    
    if (afterSemi.isValid()) {
      SourceRange range(CE->getBeginLoc(), afterSemi);
      std::string repl = begin + callText + "; " + end;
      R.ReplaceText(range, repl);
      return;
    }
  }

private:
  State& S;
};

class Action : public ASTFrontendAction {
public:
  Action(std::unordered_set<std::string> Targets) : Targets(std::move(Targets)) {}

  void EndSourceFileAction() override {
    S.R.overwriteChangedFiles(); // Write all modifications to disk
  } 

  std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance& CI, llvm::StringRef) override {
    S.R.setSourceMgr(CI.getSourceManager(), CI.getLangOpts()); // Initialize rewriter with source manager and language options
    S.Targets = std::unordered_set<std::string>(Targets.begin(), Targets.end());
    Finder.addMatcher(callExpr().bind("call"), &CB); // Register the matcher with the name "call", which will find all CallExpr nodes
    return Finder.newASTConsumer(); // Return consumer that will traverse AST
  }
private:
  State S; // Create State object for tracking all modifications
  std::unordered_set<std::string> Targets; // Set of target function names we want to wrap
  MatchFinder Finder; // MatchFinder object for finding matches in the AST
  Callback CB{S}; // Callback is initialized with reference to S
};

class ActionFactory : public FrontendActionFactory {
public:
  ActionFactory(std::unordered_set<std::string> Targets) : Targets(std::move(Targets)) {}
  
  std::unique_ptr<clang::FrontendAction> create() override {
    return std::make_unique<Action>(Targets);
  }
  
private:
  std::unordered_set<std::string> Targets;
};

} // namespace

int main(int argc, const char** argv) {
  CommonOptionsParser OP(argc, argv, Cat);

  // Read target function names from CSV file
  std::unordered_set<std::string> targets = readApiListFromCSV(ApiListCSV);
  
  if (targets.empty()) {
    llvm::errs() << "Error: No target functions found in CSV file: " << ApiListCSV << "\n";
    return 1;
  }
  
  llvm::outs() << "Loaded " << targets.size() << " target functions from " << ApiListCSV << ":\n";
  for (const auto& func : targets) {
    llvm::outs() << "  - " << func << "\n";
  }

  // OP.getCompilations() - Returns a CompilationDatabase object that contains compilation information 
  // (compiler flags, include paths, defines, etc.) for each source file
  // OP.getSourcePathList() - Returns the list of source files to process
  ClangTool Tool(OP.getCompilations(), OP.getSourcePathList());

  // Tool.run(...) - Executes the tool on all source files from OP.getSourcePathList()
  // Per source file, ClangTool does:
  // ActionFactory::create() returns new Action(Targets)
  // -> Action::CreateASTConsumer() 
  // -> Callback::run()
  auto factory = std::make_unique<ActionFactory>(std::move(targets));
  return Tool.run(factory.get());
}

