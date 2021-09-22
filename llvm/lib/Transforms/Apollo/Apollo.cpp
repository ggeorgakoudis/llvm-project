//===- Apollo.cpp - Apollo instrumentation pass ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements Apollo instrumentation and tuning for OpenMP parallel
// regions
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/Frontend/OpenMP/OMPConstants.h"
#include "llvm/Frontend/OpenMP/OMPIRBuilder.h"
#include "llvm/IR/AbstractCallSite.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/ValueMapper.h"

using namespace llvm;
using namespace omp;

#define DEBUG_TYPE "apollo"

STATISTIC(ApolloOpenMPRegionsInstrumented,
          "Number of OpenMP parallel regions instrumented for Apollo");

static cl::list<unsigned> NumThreadsList("apollo-omp-numthreads", cl::CommaSeparated,
                             cl::desc("Num threads for OpenMP tuning"),
                             cl::value_desc("1, 2, 3, ..."), cl::OneOrMore);
static cl::opt<bool> ApolloEnableOMPClause("apollo-enable-omp-clause",cl::init(false),
                                    cl::Hidden);
static cl::opt<bool> ApolloEnableThreadInstrumentation("apollo-enable-thread-instrumentation",cl::init(false),
                                    cl::Hidden);

#define EnumAttr(Kind) Attribute::get(Ctx, Attribute::AttrKind::Kind)
#define AttributeSet(...)                                                      \
  AttributeSet::get(Ctx, ArrayRef<Attribute>({__VA_ARGS__}))

namespace {
struct Apollo : public ModulePass {
  static char ID;
  bool Changed = false;
  static bool HasRun;
  static unsigned RegionCount;
  Apollo() : ModulePass(ID) {}

  bool runOnModule(Module &M) override {
    if (HasRun)
      return false;

    HasRun = true;

    LLVMContext &Ctx = M.getContext();
    IRBuilder<> IRB(Ctx);
    OpenMPIRBuilder OMPIRB(M);
    OMPIRB.initialize();

    
    FunctionCallee ApolloRegionCreate = M.getOrInsertFunction(
        "__apollo_region_create",
        AttributeList::get(Ctx, AttributeSet(EnumAttr(ArgMemOnly)),
                           AttributeSet(), {}),
        IRB.getInt8PtrTy(), IRB.getInt32Ty(), IRB.getInt8PtrTy(),
        IRB.getInt32Ty());
    FunctionCallee ApolloRegionBegin = M.getOrInsertFunction(
        "__apollo_region_begin",
        AttributeList::get(Ctx, AttributeSet(EnumAttr(ArgMemOnly)),
                           AttributeSet(), {}),
        IRB.getVoidTy(), IRB.getInt8PtrTy());
    FunctionCallee ApolloRegionSetFeature =
        M.getOrInsertFunction("__apollo_region_set_feature",
        AttributeList::get(Ctx, AttributeSet(EnumAttr(ArgMemOnly)),
                           AttributeSet(), {}),
        IRB.getVoidTy(),
                              IRB.getInt8PtrTy(), IRB.getFloatTy());
    FunctionCallee ApolloGetPolicy = M.getOrInsertFunction(
        "__apollo_region_get_policy",
        AttributeList::get(Ctx, AttributeSet(EnumAttr(ArgMemOnly)),
                           AttributeSet(), {}),
        IRB.getInt32Ty(), IRB.getInt8PtrTy());
    FunctionCallee ApolloRegionEnd = M.getOrInsertFunction(
        "__apollo_region_end",
        AttributeList::get(Ctx, AttributeSet(EnumAttr(ArgMemOnly)),
                           AttributeSet(), {}),
        IRB.getVoidTy(), IRB.getInt8PtrTy());
    

   /*
    FunctionCallee ApolloRegionCreate = M.getOrInsertFunction(
        "__apollo_region_create",
        AttributeList::get(Ctx, AttributeSet(),
                           AttributeSet(), {}),
        IRB.getInt8PtrTy(), IRB.getInt32Ty(), IRB.getInt8PtrTy(),
        IRB.getInt32Ty());
    FunctionCallee ApolloRegionBegin = M.getOrInsertFunction(
        "__apollo_region_begin",
        AttributeList::get(Ctx, AttributeSet(),
                           AttributeSet(), {}),
        IRB.getVoidTy(), IRB.getInt8PtrTy());
    FunctionCallee ApolloRegionSetFeature =
        M.getOrInsertFunction("__apollo_region_set_feature",
        AttributeList::get(Ctx, AttributeSet(),
                           AttributeSet(), {}),
        IRB.getVoidTy(),
                              IRB.getInt8PtrTy(), IRB.getFloatTy());
    FunctionCallee ApolloGetPolicy = M.getOrInsertFunction(
        "__apollo_region_get_policy",
        AttributeList::get(Ctx, AttributeSet(),
                           AttributeSet(), {}),
        IRB.getInt32Ty(), IRB.getInt8PtrTy());
    FunctionCallee ApolloRegionEnd = M.getOrInsertFunction(
        "__apollo_region_end",
        AttributeList::get(Ctx, AttributeSet(),
                           AttributeSet(), {}),
        IRB.getVoidTy(), IRB.getInt8PtrTy());
    */
    DenseMap<CallBase *, SmallVector<SmallVector<CallBase *, 4>, 4>>
        ForkCItoLoopInitCI;

    auto OMPLoopIDs = {
        OMPRTL___kmpc_for_static_init_4,
        OMPRTL___kmpc_for_static_init_4u,
        OMPRTL___kmpc_for_static_init_8,
        OMPRTL___kmpc_for_static_init_8u,
        OMPRTL___kmpc_dispatch_init_4,
        OMPRTL___kmpc_dispatch_init_4u,
        OMPRTL___kmpc_dispatch_init_8,
        OMPRTL___kmpc_dispatch_init_8u,
    };

    // Find all calls parallel for init populate a map from
    // the fork call to the callpath of the parallel for init
    for (auto &LID : OMPLoopIDs) {
      Function *F = OMPIRB.getOrCreateRuntimeFunctionPtr(LID);
      if (!F)
        continue;

      errs() << "Found FUNCTION " << F->getName() << "\n";
      for (User *U : F->users()) {
        CallBase *LoopInitCI = dyn_cast<CallBase>(U);
        if (!LoopInitCI)
          continue;

        Function *Caller = LoopInitCI->getFunction();
        errs() << "Caller " << Caller->getName() << "\n";

        // Find the AbstractCallSite fork call user and the callpath leading
        // to the parallel for init call
        CallBase *ForkCI = nullptr;
        SmallVector<CallBase *, 4> Callpath;
        Callpath.clear();
        errs() << "Init push " << *LoopInitCI << "\n";
        Callpath.push_back(LoopInitCI);

        auto FindFork = [&](Function *F, auto &self) -> void {
          for (Use &U : F->uses()) {
            errs() << "User U " << *U.getUser() << "\n";
            CallBase *CI = dyn_cast<CallBase>(U.getUser());
            // If found a direct call, recurse to find the fork callback.
            if (CI) {
              errs() << "Push " << *CI << "\n";
              Callpath.push_back(CI);
              self(CI->getFunction(), self);
              // Pop up the call to continue on a different callpath.
              Callpath.pop_back();
            }
            AbstractCallSite ACS(&U);
            if (!ACS || !ACS.isCallbackCall())
              continue;

            ForkCI = dyn_cast<CallBase>(ACS.getInstruction());
            errs() << "ForkCI " << *ForkCI << "\n";
            if (ApolloEnableOMPClause) {
              // Check if Apollo is enabled for the parallel region.
              MDNode *MD = ForkCI->getMetadata("metadata.apollo");
              if (!MD)
                return;
              errs() << "=== MD\n";
              if (MD)
                MD->dump();
              errs() << "=== End of MD\n";
            }

            assert(ForkCI->getCalledFunction() ==
                       OMPIRB.getOrCreateRuntimeFunctionPtr(
                           OMPRTL___kmpc_fork_call) &&
                   "Must call OpenMP fork call");
            // Add the callpath to the ForkCI callsite.
            ForkCItoLoopInitCI[ForkCI].push_back(Callpath);
          }
        };

        errs() << "FindFork caller " << Caller->getName() << " <- LoopInit " << *LoopInitCI << "\n";
        FindFork(Caller, FindFork);
      }
    }

    for (auto &Iter : ForkCItoLoopInitCI) {
      CallBase *ForkCI = Iter.first;
      errs() << "===\n";
      errs() << "ForkCI " << *ForkCI << "\n";
      std::string nest = "";
      for (SmallVector<CallBase *, 4> Callpath : Iter.second) {
        errs() << "Callpath!\n";
        for (CallBase *Call : reverse(Callpath)) {
          nest += "\t";
          errs() << nest << " Call " << *Call << "\n";
        }
      }
      errs() << "~~~\n";
    }

    // Instrument fork call and set features using the iteration count of each
    // parallel for loop in the callpath
    for (auto &Iter : ForkCItoLoopInitCI) {
      const unsigned CallbackCalleeOperand = 2;
      const unsigned CallbackFirstArgOperand = 3;
      const unsigned CalleeFirstArgNo = 2;

      Changed = true;

      CallBase *ForkCI = Iter.first;
      SmallVector<SmallVector<CallBase *, 4>, 4> &LoopInitCIVector =
          Iter.second;

      // Inject __apollo_region_create using a GV to store the handle if
      // it does not exist
      IRB.SetInsertPoint(ForkCI);
      Constant *SrcLocStr = nullptr;

      std::string ModuleSubstr = M.getName().str();
      ModuleSubstr = ModuleSubstr.substr(ModuleSubstr.find_last_of("/") + 1);
          //(M.getName().size() > 32) ? M.getName().size() - 32 : 0);
     std::string RegionID;
     // Use the line number for the ID if it is available.
     if (ForkCI->getDebugLoc())
      RegionID = "l" + std::to_string(ForkCI->getDebugLoc().getLine());
     else
      RegionID = "n" + std::to_string(RegionCount);
      //RegionID = Twine("l").concat(Twine(ForkCI->getDebugLoc().getLine()));
     // Get last 32 chars of Module name
      SrcLocStr = IRB.CreateGlobalStringPtr(
          ModuleSubstr + ".apollo.region." + RegionID);

      assert(SrcLocStr != nullptr);
      errs() << "SrcLocStr " << ModuleSubstr << "\n";
      GlobalVariable *ApolloRegionHandleGV =
          new GlobalVariable(M, IRB.getInt8PtrTy(), /*isConstant=*/false,
                             GlobalValue::PrivateLinkage,
                             ConstantPointerNull::get(IRB.getInt8PtrTy()),
                             ".apollo.region.handle." + Twine(RegionCount));
      ApolloRegionHandleGV->setUnnamedAddr(GlobalValue::UnnamedAddr::Global);
      ApolloRegionHandleGV->setAlignment(Align(8));

      Value *Cond = IRB.CreateICmpEQ(
          IRB.CreateLoad(ApolloRegionHandleGV), ConstantPointerNull::get(IRB.getInt8PtrTy()));
      Instruction *ThenTI =
          SplitBlockAndInsertIfThen(Cond, ForkCI, /*Unreachable=*/false);
      IRB.SetInsertPoint(ThenTI);
      CallBase *ApolloRegionCreateCI = IRB.CreateCall(
          ApolloRegionCreate, {IRB.getInt32(LoopInitCIVector.size()), SrcLocStr,
                               IRB.getInt32(NumThreadsList.size())});
      IRB.CreateStore(ApolloRegionCreateCI, ApolloRegionHandleGV);

      // Instrument with __apollo_region_begin.
      IRB.SetInsertPoint(ForkCI);
      IRB.CreateCall(ApolloRegionBegin, {IRB.CreateLoad(ApolloRegionHandleGV)});
      errs() << "ForkCI " << *ForkCI << "\n";

      // Move instruction slices before fork call to calculate number of
      // iterations of each loop as features.
      for (SmallVector<CallBase *, 4> Callpath : LoopInitCIVector) {
        // Set of visited instructions for fast lookup.
        SmallPtrSet<Value *, 8> InstructionSet;
        // The complete instruction backtrace.
        SmallVector<Instruction *, 8> InstructionBacktrace;
        Value *UB = nullptr;

        // Unpeel the callpath and slice instructions.
        for (CallBase *CI : Callpath) {
          errs() << "Callpath CI " << *CI << "\n";

          // Slice of instructions.
          SmallVector<Instruction *, 8> InstructionSlice;

          DominatorTree DT(*CI->getFunction());
          int count = 1;

          // Slice instructions based on Value V
          auto Slice = [&](Value *V, auto &self) -> void {
            Instruction *I = dyn_cast<Instruction>(V);
            if (!I) {
              errs() << "Not an Instruction " << *V << " return\n";
              //for(User *U : V->users())
              //  errs() << "User of V " << *V << " <- " << *U << "\n";
              return;
            }

            if (InstructionSet.count(I))
              return;

            errs() << "Follow I " << *I << "\n";
            InstructionSet.insert(I);
            InstructionSlice.push_back(I);
            errs() << "Added Value I " << *I << "\n";

            for (User *U : V->users()) {
              Instruction *II = dyn_cast<Instruction>(U);
              if (!II)
                continue;
              dbgs() << "=== Examining II " << *II << "\n";
              if (InstructionSet.count(II)) {
                dbgs() << "In InstructionSet continue\n";
                continue;
              }
              if (!isa<StoreInst>(II) && !II->mayWriteToMemory()) {
                dbgs() << "Not store, not write to mem, continue\n";
                continue;
              }
              if (!DT.dominates(II, CI)) {
                dbgs() << "Does not dom CI, continue\n";
                continue;
              }
              if (II == I) {
                dbgs() << "Identical, continue\n";
                continue;
              }
              dbgs() << "=== End of Examining\n";

              errs() << count
                     << " >>> Users User of I " << *I
                     << " -> " << *II << "\n";
              count++;
              self(II, self);
              count--;
              errs() << count << " <<<\n";
            }

            errs() << count << "<<<\n";

            for (Value *V : I->operand_values()) {
              errs() << "I " << *I << " Found OpI " << *V << "\n";
              // Do not follow self operand.
              if(V == I)
                continue;
              self(V, self);
            }
          };

          // LoopInitCI is by construction at the front of the Callpath, so get
          // the upper bound number of iterations to use as feature and slice
          // based to that.
          if (CI == Callpath.front()) {
            Function *Callee = CI->getCalledFunction();
            unsigned UpperBoundArgNo;
            // Set the UB argument number for static scheduling.
            if (Callee == OMPIRB.getOrCreateRuntimeFunctionPtr(
                              OMPRTL___kmpc_for_static_init_4) ||
                Callee == OMPIRB.getOrCreateRuntimeFunctionPtr(
                              OMPRTL___kmpc_for_static_init_4u) ||
                Callee == OMPIRB.getOrCreateRuntimeFunctionPtr(
                              OMPRTL___kmpc_for_static_init_8) ||
                Callee == OMPIRB.getOrCreateRuntimeFunctionPtr(
                              OMPRTL___kmpc_for_static_init_8u))
              UpperBoundArgNo = 5;
            // Set the UB argument number for dynamic, runtime scheduling.
            else
              UpperBoundArgNo = 4;

            UB = CI->getArgOperand(UpperBoundArgNo);
            errs() << "UB " << *UB << "\n";
            Slice(UB, Slice);
          } else {
            errs() << "=== CI Func\n";
            CI->getParent()->getParent()->dump();
            errs() << "=== End of CI Func\n";
            Function *Callee = CI->getCalledFunction();
            SmallPtrSet<Value *, 8> CallpathSet(Callpath.begin(),
                                                      Callpath.end());

            for (unsigned ArgNo = 0, NumArgs = Callee->arg_size();
                 ArgNo < NumArgs; ++ArgNo) {
              auto *A = Callee->getArg(ArgNo);
              errs() << "Arg " << *A << "\n";
              for (User *U : A->users()) {
                errs() << "User of A " << *A << " <- " << *U << "\n";

                Instruction *I = dyn_cast<Instruction>(U);

                if (!I)
                  continue;

                if (!InstructionSet.count(I->stripPointerCasts()) &&
                    !CallpathSet.count(I->stripPointerCasts())) {
                  errs() << "DID NOT find " << *U
                         << " in InstructionSet or CallpathSet\n";
                  continue;
                }

                errs() << "FOUND USER in InstructionSet or CallpathSet!\n";
                Value *V = CI->getArgOperand(ArgNo);
                errs() << "Arg Value " << *V << "\n";
                Slice(V, Slice);
              }
            }
          }

          // Sort slice.
          llvm::sort(InstructionSlice.begin(),
                     InstructionSlice.end(),
                     [&DT](Instruction *A, Instruction *B) {
                       return DT.dominates(A, B);
                     });

          errs() << "=== CURRENT SLICE\n";
          for (auto It = InstructionSlice.begin();
               It < InstructionSlice.end(); ++It) {
            Instruction *I = *It;
            errs() << "I " << *I << "\n";
          }
          errs() << "=== END OF CURRENT SLICE\n";

          InstructionBacktrace.insert(InstructionBacktrace.begin(),
                                      InstructionSlice.begin(),
                                      InstructionSlice.end());

          errs() << "=== CURRENT BACKTRACE\n";
          for (auto It = InstructionBacktrace.begin();
               It < InstructionBacktrace.end(); ++It) {
            Instruction *I = *It;
            errs() << "I " << *I << "\n";
          }
          errs() << "=== END OF CURRENT BACKTRACE\n";
        }

        assert(UB != nullptr);
        errs() << "====================== RESULT ==========================\n";

        llvm::ValueToValueMapTy VMap;

        // Map arguments between the fork call callback and the outlined called
        // to be used by sliced instruction cloning.
        Function *Callee = dyn_cast<Function>(
            ForkCI->getArgOperand(CallbackCalleeOperand)->stripPointerCasts());
        errs() << "ForkCI Forwarding " << *ForkCI << " -> " << Callee->getName()
               << "\n";
        // TODO: generalize for callbacks using metadata for argument list
        // Set the mappings between arguments in the outlined to the callback.
        for (unsigned u = CallbackFirstArgOperand, uu = CalleeFirstArgNo,
                      e = ForkCI->getNumArgOperands();
             u < e; ++u, ++uu) {
          errs() << "Mapping arg " << *Callee->getArg(uu) << " -> "
                 << *ForkCI->getArgOperand(u) << "\n";
          VMap[Callee->getArg(uu)] = ForkCI->getArgOperand(u);
        }

        // Map arguments in the Callpath used by sliced instruction cloning.
        for (CallBase *CI : reverse(Callpath)) {
          Function *Callee = CI->getCalledFunction();
          errs() << "Forwarding " << *CI << " -> " << Callee->getName() << "\n";
          for (unsigned i = 0; i < Callee->arg_size(); i++) {
            // Forward VMap if found due to nested Callpath.
            if (VMap.count(CI->getArgOperand(i)))
              VMap[Callee->getArg(i)] = VMap[CI->getArgOperand(i)];
            else
              VMap[Callee->getArg(i)] = CI->getArgOperand(i);
            errs() << "Mapping arg " << *Callee->getArg(i) << " -> "
                   << *VMap[Callee->getArg(i)] << "\n";
          }
        }

        errs() << "=== INSTRUCTIONS\n";
        for (auto It = InstructionBacktrace.begin();
             It < InstructionBacktrace.end(); ++It) {
          Instruction *I = *It;
          errs() << "I " << *I << "\n";
        }
        errs() << "=== END OF INSTRUCTIONS\n";
        // Clone instructions and remap them. Note inserting clones and
        // remapping them is correct, because instructions insert in program
        // order
        Value *UBClone = nullptr;
        for (auto It = InstructionBacktrace.begin();
             It < InstructionBacktrace.end(); ++It) {
          Instruction *I = *It;
          errs() << "Slice I " << *I << "\n";

          if (PHINode *P = dyn_cast<PHINode>(I)) {
            assert(false && "Cannot analyze PHINode");
            errs() << "Found PHI!\n";
            Value *V = P->getIncomingValue(0);
            assert(VMap[V] && "Expected value to be mapped");
            VMap[I] = VMap[V];
            continue;
          }

          Instruction *CloneI = I->clone();

          Instruction *UBI = dyn_cast<Instruction>(UB);

          // Store the upper bound clone to create the feature
          if (UBI == I)
            UBClone = CloneI;

          //for(auto It : VMap) {
          //  const Value *Key = It.first;
          //  Value *Val = It.second;
          //  errs() << "VMAP[ " << *Key << " ] -> " << *Val << "\n";
          //}
          // TODO: Ignore missing locals or not?
          // TODO: No module level changes?
          RemapInstruction(
              CloneI, VMap
              /*, RF_NoModuleLevelChanges | RF_IgnoreMissingLocals*/);

          //auto *F = IRB.GetInsertBlock()->getParent();
          //errs() << "=== Function\n";
          //F->dump();
          //errs() << "=== End of Function\n";

          if (I->hasName())
            IRB.Insert(CloneI, I->getName() + ".apollo.slice");
          else if (!I->getType()->isVoidTy())
            IRB.Insert(CloneI, "tmp.apollo.slice");
          else
            IRB.Insert(CloneI);

          VMap[I] = CloneI;
          errs() << "Self remap " << *I << " -> " << *CloneI << "\n";

          // Update value map for nested instructions clones in the Callpath.
          for (auto It : VMap) {
            const Value *Key = It.first;
            Value *Val = It.second;
            if (Val == I) {
              errs() << "Remap " << *Key << " -> " << *CloneI << "\n";
              VMap[Key] = CloneI;
            }
          }

          errs() << "CloneI " << *CloneI << "\n\n";
        }

        assert(UBClone != nullptr);

        errs() << "UBClone I " << *UBClone << "\n";

        Value *IntNumIters = nullptr;
        Type *UBType = UBClone->getType();
        errs() << "UBClone Type " << *UBType << "\n";
        // UB clone is pointer in static scheduling.
        if (UBType->isPointerTy()) {
          //UBType = dyn_cast<PointerType>(UBType)->getElementType();
          errs() << "Pointer UBClone Element Type " << *UBType->getPointerElementType() << "\n";
          Value *UBValue = IRB.CreateLoad(UBType->getPointerElementType(), UBClone);
          IntNumIters = IRB.CreateAdd(UBValue, ConstantInt::get(UBType->getPointerElementType(), 1));
        }
        // UB clone is a scalar in dynamic scheduling.
        else
          IntNumIters = IRB.CreateAdd(UBClone, ConstantInt::get(UBType, 1));

        assert(IntNumIters && "Expected non-null IntNumIters");
        Value *FloatNumIters = IRB.CreateUIToFP(IntNumIters, IRB.getFloatTy());
        IRB.CreateCall(ApolloRegionSetFeature,
                       {IRB.CreateLoad(ApolloRegionHandleGV), FloatNumIters});
      }
      errs() << "ForkCI " << *ForkCI << "\n";

      errs() << "====================== END OF RESULT ==========================\n";

      CallBase *ApolloGetPolicyCI =
          IRB.CreateCall(ApolloGetPolicy, {IRB.CreateLoad(ApolloRegionHandleGV)});

      BasicBlock *ForkBB = SplitBlock(ForkCI->getParent(), ForkCI);
      IRB.SetInsertPoint(ApolloGetPolicyCI->getParent()->getTerminator());
      OMPIRB.updateToLocation(IRB);
      SrcLocStr = OMPIRB.getOrCreateSrcLocStr(IRB);
      Value *Ident = OMPIRB.getOrCreateIdent(SrcLocStr);
      Value *ThreadID = OMPIRB.getOrCreateThreadID(Ident);
      SwitchInst *SwitchI = IRB.CreateSwitch(ApolloGetPolicyCI, ForkBB);
      ApolloGetPolicyCI->getParent()->getTerminator()->eraseFromParent();

      auto CreateCase = [&](int CaseNo, int NumThreads) {
        BasicBlock *Case =
            BasicBlock::Create(Ctx, ".apollo.case." + Twine(CaseNo),
                               ForkBB->getParent(), ForkBB);
        IRB.SetInsertPoint(Case);
        OMPIRB.updateToLocation(IRB);
        // Build call __kmpc_push_num_threads(&Ident, global_tid, num_threads).
        /*
        Value *Args[] = {
            Ident,
            ThreadID,
            IRB.getInt32(NumThreads),
        };
        IRB.CreateCall(OMPIRB.getOrCreateRuntimeFunctionPtr(
                           OMPRTL___kmpc_push_num_threads),
                       Args);
        */
        Value *Args[] = {
            IRB.getInt32(NumThreads),
        };
        IRB.CreateCall(OMPIRB.getOrCreateRuntimeFunctionPtr(
                           OMPRTL_omp_set_num_threads),
                       Args);
        IRB.CreateBr(ForkBB);
        SwitchI->addCase(IRB.getInt32(CaseNo), Case);
      };

      // Create different number of threads cases.
      unsigned int CaseNo = 0;
      for (unsigned NumThreads : NumThreadsList) {
        CreateCase(CaseNo, NumThreads);
        CaseNo++;
      }
      assert(CaseNo == NumThreadsList.size() &&
             "Expected the some number of cases and number of threads in the "
             "list");

      OMPIRB.finalize();

      IRB.SetInsertPoint(ForkCI->getNextNode());
      IRB.CreateCall(ApolloRegionEnd, {IRB.CreateLoad(ApolloRegionHandleGV)});

      if (ApolloEnableThreadInstrumentation) {
        FunctionCallee ApolloRegionThreadBegin = M.getOrInsertFunction(
            "__apollo_region_thread_begin",
            AttributeList::get(Ctx, AttributeSet(EnumAttr(ArgMemOnly)),
                               AttributeSet(), {}),
            IRB.getVoidTy(), IRB.getInt8PtrTy());
        FunctionCallee ApolloRegionThreadEnd = M.getOrInsertFunction(
            "__apollo_region_thread_end",
            AttributeList::get(Ctx, AttributeSet(EnumAttr(ArgMemOnly)),
                               AttributeSet(), {}),
            IRB.getVoidTy(), IRB.getInt8PtrTy());

        Function *OutlinedFn =
            dyn_cast<Function>(ForkCI->getArgOperand(2)->stripPointerCasts());
        assert(OutlinedFn && "Expected OutlinedFn in ForkCI operand");
        IRB.SetInsertPoint(&OutlinedFn->getEntryBlock(),
                           OutlinedFn->getEntryBlock().getFirstInsertionPt());
        IRB.CreateCall(ApolloRegionThreadBegin,
                       {IRB.CreateLoad(ApolloRegionHandleGV)});

        // Find all return instructions in the function and instrument with
        // __apollo_region_thread_end.
        for (BasicBlock &BB : *OutlinedFn)
          for (Instruction &I : BB)
            if (dyn_cast<ReturnInst>(&I)) {
              IRB.SetInsertPoint(&I);
              IRB.CreateCall(ApolloRegionThreadEnd,
                             {IRB.CreateLoad(ApolloRegionHandleGV)});
            }
      }

      RegionCount++;
      ApolloOpenMPRegionsInstrumented++;
    }

    return Changed;
  }

  // We don't modify the program, so we preserve all analyses.
  void getAnalysisUsage(AnalysisUsage &AU) const override {
    if (!Changed)
      AU.setPreservesAll();
  }
};
} // namespace

bool Apollo::HasRun = false;
unsigned Apollo::RegionCount = 1;
char Apollo::ID = 0;
static RegisterPass<Apollo> X("apollo", "Apollo instrumentation module pass");

static RegisterStandardPasses Y(PassManagerBuilder::EP_ModuleOptimizerEarly,
                                [](const PassManagerBuilder &Builder,
                                   legacy::PassManagerBase &PM) {
                                  PM.add(new Apollo());
                                });

static RegisterStandardPasses Z(PassManagerBuilder::EP_EnabledOnOptLevel0,
                                [](const PassManagerBuilder &Builder,
                                   legacy::PassManagerBase &PM) {
                                  PM.add(new Apollo());
                                });
