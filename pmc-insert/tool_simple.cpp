// Simplified version for LLVM 6.0 - wraps only simple expression statements
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

using namespace clang;
using namespace clang::tooling;
using namespace clang::ast_matchers;

static llvm::cl::OptionCategory Cat("pmc-insert options");

static llvm::cl::list<std::string> WrapNames(
  "wrap", llvm::cl::desc("Function name to wrap (repeatable): --wrap=foo"),
  llvm::cl::OneOrMore, llvm::cl::cat(Cat));

static llvm::cl::opt<std::string> IncludeHeader(
  "include-header", llvm::cl::desc("Header to insert once per file (optional)"),
  llvm::cl::init(""), llvm::cl::cat(Cat));

namespace {

struct State {
  Rewriter R;
  std::unordered_set<std::string> Targets;
  std::unordered_set<const FileEntry*> IncludedFiles;
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
  SourceLocation begin = S->getLocStart();
  SourceLocation afterSemi = Lexer::findLocationAfterToken(
      S->getLocEnd(), tok::semi, SM, Ctx.getLangOpts(), true);
  if (afterSemi.isInvalid()) return SourceRange();
  return SourceRange(begin, afterSemi);
}

class Callback : public MatchFinder::MatchCallback {
public:
  explicit Callback(State& S) : S(S) {}

  void run(const MatchFinder::MatchResult& Res) override {
    const auto* CE = Res.Nodes.getNodeAs<CallExpr>("call");
    if (!CE) return;

    const SourceManager& SM = *Res.SourceManager;
    
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

    auto& Ctx = *Res.Context;
    auto& R   = S.R;
    LangOptions LO = Ctx.getLangOpts();

    // Insert header once
    if (!IncludeHeader.empty()) {
      FileID FID = SM.getMainFileID();
      const FileEntry* FE = SM.getFileEntryForID(FID);
      if (FE && !S.IncludedFiles.count(FE)) {
        R.InsertText(SM.getLocForStartOfFile(FID),
                     "#include \"" + IncludeHeader + "\"\n");
        S.IncludedFiles.insert(FE);
      }
    }

    // Helpers - use unique variable names to avoid redefinition
    auto callText = getText(SM, LO, CE->getSourceRange());
    std::string hvar = S.genTemp("__pmc_h");
    std::string begin = "void* " + hvar + " = pmc_measure_begin_csv(__func__, NULL); ";
    std::string end   = "pmc_measure_end(" + hvar + ", 1); ";

    // Try different patterns based on parent context
    
    // Walk up parent chain to find context
    const Expr* Current = CE;
    const BinaryOperator* FoundAssign = nullptr;
    const VarDecl* FoundVarDecl = nullptr;
    
    // Walk up through implicit casts and find BinaryOperator or VarDecl
    for (int depth = 0; depth < 10; depth++) {  // Limit depth to avoid infinite loops
      auto Parents = Ctx.getParents(*Current);
      if (Parents.empty()) break;
      
      // Check for assignment
      if (const BinaryOperator* BO = Parents[0].get<BinaryOperator>()) {
        if (BO->getOpcode() == BO_Assign) {
          FoundAssign = BO;
          break;
        }
      }
      
      // Check for variable declaration
      if (const VarDecl* VD = Parents[0].get<VarDecl>()) {
        if (VD->hasInit()) {
          FoundVarDecl = VD;
          break;
        }
      }
      
      // Move up to next parent (through implicit casts, etc.)
      if (const Expr* E = Parents[0].get<Expr>()) {
        Current = E;
      } else {
        break;
      }
    }
    
    // 2) Handle assignment statement: `x = foo();`
    // Strategy: Insert "begin" before assignment, insert "end" after semicolon
    if (FoundAssign) {
      // Use the assignment's own location (don't walk up to avoid finding CompoundStmt)
      SourceLocation assignStart = FoundAssign->getLocStart();
      SourceLocation assignEnd = FoundAssign->getLocEnd();
      
      // Insert "begin" at start of assignment
      R.InsertTextBefore(assignStart, begin);
      
      // Find semicolon after the assignment and insert "end" after it
      SourceLocation afterToken = Lexer::getLocForEndOfToken(assignEnd, 0, SM, LO);
      
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
      auto ParentsVD = Ctx.getParents(*FoundVarDecl);
      for (auto& P : ParentsVD) {
        if (const DeclStmt* DS = P.get<DeclStmt>()) {
          if (DS->isSingleDecl()) {
            // Find semicolon after the declaration statement
            SourceLocation DSEnd = DS->getLocEnd();
            SourceLocation afterSemi = Lexer::findLocationAfterToken(
                DSEnd, tok::semi, SM, LO, false);
            
            if (afterSemi.isValid()) {
              SourceRange RR(DS->getLocStart(), afterSemi);
              std::string T = FoundVarDecl->getType().getAsString(Ctx.getPrintingPolicy());
              std::string name = FoundVarDecl->getName().str();
              std::string tmp = S.genTemp();
              std::string repl = "{ " + begin + T + " " + tmp + " = (" + callText + "); "
                                 + end + T + " " + name + " = " + tmp + "; }";
              R.ReplaceText(RR, repl);
              return;
            }
          }
        }
      }
    }

    // 1) Expression statement: `foo();` (fallback for simple calls)
    // Find the semicolon after the call
    SourceLocation afterSemi = Lexer::findLocationAfterToken(
        CE->getLocEnd(), tok::semi, SM, LO, false);
    
    if (afterSemi.isValid()) {
      SourceRange range(CE->getLocStart(), afterSemi);
      std::string repl = "{ " + begin + callText + "; " + end + "}";
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
    S.R.overwriteChangedFiles();
  }

  std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance& CI, llvm::StringRef) override {
    S.R.setSourceMgr(CI.getSourceManager(), CI.getLangOpts());
    S.Targets = std::unordered_set<std::string>(Targets.begin(), Targets.end());
    Finder.addMatcher(callExpr().bind("call"), &CB);
    return Finder.newASTConsumer();
  }
private:
  State S;
  std::unordered_set<std::string> Targets;
  MatchFinder Finder;
  Callback CB{S};
};

class ActionFactory : public FrontendActionFactory {
public:
  ActionFactory(std::unordered_set<std::string> Targets) : Targets(std::move(Targets)) {}
  
  clang::FrontendAction* create() override {
    return new Action(Targets);
  }
  
private:
  std::unordered_set<std::string> Targets;
};

} // namespace

int main(int argc, const char** argv) {
  CommonOptionsParser OP(argc, argv, Cat);

  std::unordered_set<std::string> targets;
  for (auto& w : WrapNames) targets.insert(w);

  ClangTool Tool(OP.getCompilations(), OP.getSourcePathList());
  return Tool.run(new ActionFactory(std::move(targets)));
}

