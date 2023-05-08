#include "CGIntrinsicsOpenMP.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/Debug.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/CodeExtractor.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"

#define DEBUG_TYPE "intrinsics-openmp"

using namespace llvm;
using namespace omp;
using namespace iomp;

Function *CGIntrinsicsOpenMP::createOutlinedFunction(
    MapVector<Value *, DSAType> &DSAValueMap, ValueToValueMapTy *VMap,
    Function *OuterFn, BasicBlock *BBEntry, BasicBlock *StartBB,
    BasicBlock *EndBB, BasicBlock *AfterBB,
    SmallVectorImpl<Value *> &CapturedVars, StringRef Suffix) {
  SmallVector<Value *, 16> Privates;
  SmallVector<Value *, 16> CapturedShared;
  SmallVector<Value *, 16> CapturedFirstprivate;

  OpenMPIRBuilder::OutlineInfo OI;
  OI.EntryBB = StartBB;
  OI.ExitBB = EndBB;
  SmallPtrSet<BasicBlock *, 8> BlockSet;
  SmallVector<BasicBlock *, 8> BlockVector;
  OI.collectBlocks(BlockSet, BlockVector);

  CodeExtractorAnalysisCache CEAC(*OuterFn);
    CodeExtractor Extractor(BlockVector, /* DominatorTree */ nullptr,
                          /* AggregateArgs */ false,
                          /* BlockFrequencyInfo */ nullptr,
                          /* BranchProbabilityInfo */ nullptr,
                          /* AssumptionCache */ nullptr,
                          /* AllowVarArgs */ true,
                          /* AllowAlloca */ true,
                          /* Suffix */ ".");

  // Find inputs to, outputs from the code region.
  BasicBlock *CommonExit = nullptr;
  SetVector<Value *> Inputs, Outputs, SinkingCands, HoistingCands;
  Extractor.findAllocas(CEAC, SinkingCands, HoistingCands, CommonExit);
  Extractor.findInputsOutputs(Inputs, Outputs, SinkingCands);

  assert(Outputs.empty() && "Expected empty outputs from outlined region");
  assert(SinkingCands.empty() && "Expected empty alloca sinking candidates");

  // Scan Inputs and define any missing values as Privates. Those values must
  // correspond to Numba-generated temporaries that should be privatized.
  for (auto *V : Inputs) {
    if (!DSAValueMap.count(V)) {
      LLVM_DEBUG(dbgs() << "Missing V " << *V << " from DSAValueMap, will privatize\n");
      assert(V->getName().startswith(".") &&
             "Expected Numba temporary value, named starting with .");
      Privates.push_back(V);
      continue;
    }

    auto DSA = DSAValueMap[V];

    if (DSA_PRIVATE == DSA)
      Privates.push_back(V);
    else if (DSA_FIRSTPRIVATE == DSA)
      CapturedFirstprivate.push_back(V);
    else if (DSA_SHARED == DSA)
      CapturedShared.push_back(V);
    else
      assert(false && "Unsupported DSA type");
  }

  SmallVector<Type *, 16> Params;
  // tid
  Params.push_back(OMPBuilder.Int32Ptr);
  // bound_tid
  Params.push_back(OMPBuilder.Int32Ptr);
  for (auto *V : CapturedShared)
    Params.push_back(V->getType());
  for (auto *V : CapturedFirstprivate) {
    Type *VPtrElemTy = V->getType()->getPointerElementType();
    if (VPtrElemTy->isSingleValueType())
      Params.push_back(VPtrElemTy);
    else
      Params.push_back(V->getType());
  }

  FunctionType *OutlinedFnTy =
      FunctionType::get(OMPBuilder.Void, Params, /* isVarArgs */ false);
  Function *OutlinedFn =
      Function::Create(OutlinedFnTy, GlobalValue::InternalLinkage,
                       OuterFn->getName() + Suffix, M);

  // Name the parameters and add attributes. Shared are ordered before
  // firstprivate in the parameter list.
  OutlinedFn->arg_begin()->setName("global_tid");
  std::next(OutlinedFn->arg_begin())->setName("bound_tid");
  Function::arg_iterator AI = std::next(OutlinedFn->arg_begin(), 2);
  int arg_no = 2;
  for (auto *V : CapturedShared) {
    AI->setName(V->getName() + ".shared");
    OutlinedFn->addParamAttr(arg_no, Attribute::NonNull);
    OutlinedFn->addParamAttr(
        arg_no, Attribute::get(M.getContext(), Attribute::Dereferenceable, 8));
    ++AI;
    ++arg_no;
  }
  for (auto *V : CapturedFirstprivate) {
    Type *VPtrElemTy = V->getType()->getPointerElementType();
    if (VPtrElemTy->isSingleValueType()) {
      AI->setName(V->getName() + ".firstprivate.byval");
    } else {
      AI->setName(V->getName() + ".firstprivate");
      OutlinedFn->addParamAttr(arg_no, Attribute::NonNull);
      OutlinedFn->addParamAttr(
          arg_no,
          Attribute::get(M.getContext(), Attribute::Dereferenceable, 8));
    }
    ++AI;
    ++arg_no;
  }

  BasicBlock *OutlinedEntryBB =
      BasicBlock::Create(M.getContext(), ".outlined.entry", OutlinedFn);
  BasicBlock *OutlinedExitBB =
      BasicBlock::Create(M.getContext(), ".outlined.exit", OutlinedFn);
  OMPBuilder.Builder.SetInsertPoint(OutlinedEntryBB);

  OutlinedFn->addParamAttr(0, Attribute::NoAlias);
  OutlinedFn->addParamAttr(1, Attribute::NoAlias);
  OutlinedFn->addFnAttr(Attribute::NoUnwind);
  OutlinedFn->addFnAttr(Attribute::NoRecurse);

  auto CollectUses = [&BlockSet](Value *V, SetVector<Use *> &Uses) {
    for (Use &U : V->uses())
      if (auto *UserI = dyn_cast<Instruction>(U.getUser()))
        if (BlockSet.count(UserI->getParent()))
          Uses.insert(&U);
  };

  auto ReplaceUses = [](SetVector<Use *> &Uses, Value *ReplacementValue) {
    for (Use *UPtr : Uses)
      UPtr->set(ReplacementValue);
  };

  for (auto *V : Privates) {
    SetVector<Use *> Uses;
    CollectUses(V, Uses);

    Type *VTy = V->getType()->getPointerElementType();
    Value *ReplacementValue = OMPBuilder.Builder.CreateAlloca(
        VTy, nullptr, V->getName() + ".private");

    if (VMap)
      (*VMap)[V] = ReplacementValue;

    ReplaceUses(Uses, ReplacementValue);
  }

  AI = std::next(OutlinedFn->arg_begin(), 2);
  for (auto *V : CapturedShared) {
    SetVector<Use *> Uses;
    CollectUses(V, Uses);

    Value *ReplacementValue = AI;

    if (VMap)
      (*VMap)[V] = ReplacementValue;

    ReplaceUses(Uses, ReplacementValue);
    ++AI;
  }

  for (auto *V : CapturedFirstprivate) {
    SetVector<Use *> Uses;
    CollectUses(V, Uses);

    Type *VPtrElemTy = V->getType()->getPointerElementType();
    Value *ReplacementValue =
        OMPBuilder.Builder.CreateAlloca(VPtrElemTy, nullptr, V->getName() + ".copy");
    if (VPtrElemTy->isSingleValueType())
      OMPBuilder.Builder.CreateStore(AI, ReplacementValue);
    else {
      Value *Load =
          OMPBuilder.Builder.CreateLoad(VPtrElemTy, AI, V->getName() + ".reload");
      OMPBuilder.Builder.CreateStore(Load, ReplacementValue);
    }

    if (VMap)
      (*VMap)[V] = ReplacementValue;

    ReplaceUses(Uses, ReplacementValue);

    ++AI;
  }

  OMPBuilder.Builder.CreateBr(StartBB);

  EndBB->getTerminator()->setSuccessor(0, OutlinedExitBB);
  OMPBuilder.Builder.SetInsertPoint(OutlinedExitBB);
  OMPBuilder.Builder.CreateRetVoid();

  for (auto *BB : BlockSet)
    BB->moveAfter(&OutlinedFn->getEntryBlock());

  LLVM_DEBUG(dbgs() << "=== Dump OutlinedFn\n"
                    << *OutlinedFn << "=== End of Dump OutlinedFn\n");

  /*
  const DebugLoc DL = BBEntry->getTerminator()->getDebugLoc();
  BBEntry->getTerminator()->eraseFromParent();
  OpenMPIRBuilder::LocationDescription Loc(
      InsertPointTy(BBEntry, BBEntry->end()), DL);


  dbgs() << "=== BEFORE Dump OuterFn\n" << *OuterFn << "=== End of Dump
  OuterFn\n";

  Constant *SrcLocStr = OMPBuilder.getOrCreateSrcLocStr(Loc);
  OMPBuilder.Builder.restoreIP(Loc.IP);
  OMPBuilder.Builder.SetCurrentDebugLocation(Loc.DL);

  Value *Ident = OMPBuilder.getOrCreateIdent(SrcLocStr);
  //Value *ThreadID = OMPBuilder.getOrCreateThreadID(Ident);

  auto *OutlinedFnCast = OMPBuilder.Builder.CreateBitCast(
      OutlinedFn, OMPBuilder.ParallelTaskPtr);
  FunctionCallee ForkCall = OMPBuilder.getOrCreateRuntimeFunction(M,
  OMPRTL___kmpc_fork_call); SmallVector<Value *, 16> ForkArgs;
  ForkArgs.append(
      {Ident,
       OMPBuilder.Builder.getInt32(CapturedShared.size() +
                                   CapturedFirstprivate.size()),
       OutlinedFnCast});
  ForkArgs.append(CapturedShared);
  ForkArgs.append(CapturedFirstprivate);

  OMPBuilder.Builder.CreateCall(ForkCall, ForkArgs);
  OMPBuilder.Builder.CreateBr(AfterBB);

  dbgs() << "=== Dump OuterFn\n" << *OuterFn << "=== End of Dump OuterFn\n";
  */

  if (verifyFunction(*OutlinedFn, &errs()))
    report_fatal_error("Verification of OutlinedFn failed!");

  CapturedVars.append(CapturedShared);
  CapturedVars.append(CapturedFirstprivate);
  return OutlinedFn;
}

CGIntrinsicsOpenMP::CGIntrinsicsOpenMP(Module &M) : OMPBuilder(M), M(M) {
  OMPBuilder.initialize();

  TgtOffloadEntryTy = StructType::create({OMPBuilder.Int8Ptr,
                                          OMPBuilder.Int8Ptr, OMPBuilder.SizeTy,
                                          OMPBuilder.Int32, OMPBuilder.Int32},
                                         "struct.__tgt_offload_entry");
}

void CGIntrinsicsOpenMP::emitOMPParallel(
    MapVector<Value *, DSAType> &DSAValueMap, ValueToValueMapTy *VMap,
    const DebugLoc &DL, Function *Fn, BasicBlock *BBEntry, BasicBlock *StartBB,
    BasicBlock *EndBB, BasicBlock *AfterBB, FinalizeCallbackTy FiniCB,
    ParRegionInfoStruct &ParRegionInfo) {
  if (isOpenMPDeviceRuntime())
    emitOMPParallelDeviceRuntime(DSAValueMap, VMap, DL, Fn, BBEntry, StartBB,
                                 EndBB, AfterBB, FiniCB, ParRegionInfo);
  else
    emitOMPParallelHostRuntime(DSAValueMap, VMap, DL, Fn, BBEntry, StartBB,
                               EndBB, AfterBB, FiniCB, ParRegionInfo);
}

void CGIntrinsicsOpenMP::emitOMPParallelHostRuntime(
    MapVector<Value *, DSAType> &DSAValueMap, ValueToValueMapTy *VMap,
    const DebugLoc &DL, Function *Fn, BasicBlock *BBEntry, BasicBlock *StartBB,
    BasicBlock *EndBB, BasicBlock *AfterBB, FinalizeCallbackTy FiniCB,
    ParRegionInfoStruct &ParRegionInfo) {
  InsertPointTy BodyIP, BodyAllocaIP;
  SmallVector<OpenMPIRBuilder::ReductionInfo> ReductionInfos;

  auto PrivCB = [&](InsertPointTy AllocaIP, InsertPointTy CodeGenIP,
                    Value &Orig, Value &Inner,
                    Value *&ReplacementValue) -> InsertPointTy {
    auto It = DSAValueMap.find(&Orig);
    LLVM_DEBUG(dbgs() << "DSAValueMap for Orig " << Orig << " Inner " << Inner);
    if (It != DSAValueMap.end())
      LLVM_DEBUG(dbgs() << It->second);
    else
      LLVM_DEBUG(dbgs() << " (null)!");
    LLVM_DEBUG(dbgs() << "\n ");

    if (It == DSAValueMap.end()) {
      DSAValueMap[&Orig] = DSA_PRIVATE;
      LLVM_DEBUG(dbgs() << "Missing V " << Orig << " from DSAValueMap, will privatize\n");
      assert(Orig.getName().startswith(".") &&
             "Expected Numba temporary value, named starting with .");
    }
    assert(It != DSAValueMap.end() && "Expected Value in DSAValueMap");

    DSAType DSA = It->second;

    if (DSA == DSA_PRIVATE) {
      OMPBuilder.Builder.restoreIP(AllocaIP);
      Type *VTy = Inner.getType()->getPointerElementType();
      ReplacementValue = OMPBuilder.Builder.CreateAlloca(
          VTy, /*ArraySize */ nullptr, Inner.getName());
      OMPBuilder.Builder.CreateStore(Constant::getNullValue(VTy),
                                     ReplacementValue);
      LLVM_DEBUG(dbgs() << "Privatizing Inner " << Inner << " -> to -> "
                        << *ReplacementValue << "\n");
      if (VMap)
        (*VMap)[&Orig] = ReplacementValue;
    } else if (DSA == DSA_FIRSTPRIVATE) {
      OMPBuilder.Builder.restoreIP(AllocaIP);
      Type *VTy = Inner.getType()->getPointerElementType();
      Value *V = OMPBuilder.Builder.CreateLoad(VTy, &Inner,
                                               Orig.getName() + ".reload");
      ReplacementValue = OMPBuilder.Builder.CreateAlloca(
          VTy, /*ArraySize */ nullptr, Orig.getName() + ".copy");
      OMPBuilder.Builder.restoreIP(CodeGenIP);
      OMPBuilder.Builder.CreateStore(V, ReplacementValue);
      LLVM_DEBUG(dbgs() << "Firstprivatizing Inner " << Inner << " -> to -> "
                        << *ReplacementValue << "\n");
      if (VMap)
        (*VMap)[&Orig] = ReplacementValue;
    } else if (DSA == DSA_REDUCTION_ADD) {
      OMPBuilder.Builder.restoreIP(AllocaIP);
      Type *VTy = Inner.getType()->getPointerElementType();
      Value *V = OMPBuilder.Builder.CreateAlloca(VTy, /* ArraySize */ nullptr,
                                                 Orig.getName() + ".red.priv");
      ReplacementValue = V;
      if (VMap)
        (*VMap)[&Orig] = ReplacementValue;

      OMPBuilder.Builder.restoreIP(CodeGenIP);
      // Store idempotent value based on operation and type.
      // TODO: create templated emitInitAndAppendInfo in CGReduction
      if (VTy->isIntegerTy())
        OMPBuilder.Builder.CreateStore(ConstantInt::get(VTy, 0), V);
      else if (VTy->isFloatTy() || VTy->isDoubleTy())
        OMPBuilder.Builder.CreateStore(ConstantFP::get(VTy, 0.0), V);
      else
        assert(false &&
               "Unsupported type to init with idempotent reduction value");

      //ReductionInfos.push_back({&Orig, V, CGReduction::sumReduction,
      ReductionInfos.push_back(OpenMPIRBuilder::ReductionInfo(VTy, &Orig, V, CGReduction::sumReduction,
                                CGReduction::sumAtomicReduction));

      return OMPBuilder.Builder.saveIP();
    } else {
      ReplacementValue = &Inner;
      LLVM_DEBUG(dbgs() << "Shared Inner " << Inner << " -> to -> "
                        << *ReplacementValue << "\n");
    }

    return CodeGenIP;
  };

  auto BodyGenCB = [&](InsertPointTy AllocaIP, InsertPointTy CodeGenIP,
                       BasicBlock &ContinuationIP) {
    BasicBlock *CGStartBB = CodeGenIP.getBlock();
    BasicBlock *CGEndBB = SplitBlock(CGStartBB, &*CodeGenIP.getPoint());
    assert(StartBB != nullptr && "StartBB should not be null");
    CGStartBB->getTerminator()->setSuccessor(0, StartBB);
    assert(EndBB != nullptr && "EndBB should not be null");
    EndBB->getTerminator()->setSuccessor(0, CGEndBB);

    BodyIP = InsertPointTy(CGEndBB, CGEndBB->getFirstInsertionPt());
    BodyAllocaIP = AllocaIP;
  };

  IRBuilder<>::InsertPoint AllocaIP(&Fn->getEntryBlock(),
                                    Fn->getEntryBlock().getFirstInsertionPt());

  // Set the insertion location at the end of the BBEntry.
  BBEntry->getTerminator()->eraseFromParent();

  Value *IfConditionEval = nullptr;
  if (ParRegionInfo.IfCondition) {
    OMPBuilder.Builder.SetInsertPoint(BBEntry);
    if (ParRegionInfo.IfCondition->getType()->isFloatingPointTy())
      IfConditionEval = OMPBuilder.Builder.CreateFCmpUNE(
          ParRegionInfo.IfCondition,
          ConstantFP::get(ParRegionInfo.IfCondition->getType(), 0));
    else
      IfConditionEval = OMPBuilder.Builder.CreateICmpNE(
          ParRegionInfo.IfCondition,
          ConstantInt::get(ParRegionInfo.IfCondition->getType(), 0));
  }

  OpenMPIRBuilder::LocationDescription Loc(
      InsertPointTy(BBEntry, BBEntry->end()), DL);

  // TODO: support cancellable, binding.
  InsertPointTy AfterIP = OMPBuilder.createParallel(
      Loc, AllocaIP, BodyGenCB, PrivCB, FiniCB,
      /* IfCondition */ IfConditionEval,
      /* NumThreads */ ParRegionInfo.NumThreads, OMP_PROC_BIND_default,
      /* IsCancellable */ false);

  if (!ReductionInfos.empty())
    OMPBuilder.createReductions(BodyIP, BodyAllocaIP, ReductionInfos);

  BranchInst::Create(AfterBB, AfterIP.getBlock());

  LLVM_DEBUG(dbgs() << "=== Before Fn\n" << *Fn << "=== End of Before Fn\n");
  OMPBuilder.finalize(Fn);
  //OMPBuilder.finalize(Fn, /* AllowExtractorSinking */ true);
  LLVM_DEBUG(dbgs() << "=== Finalize Fn\n"
                    << *Fn << "=== End of Finalize Fn\n");
}

void CGIntrinsicsOpenMP::emitOMPParallelDeviceRuntime(
    MapVector<Value *, DSAType> &DSAValueMap, ValueToValueMapTy *VMap,
    const DebugLoc &DL, Function *Fn, BasicBlock *BBEntry, BasicBlock *StartBB,
    BasicBlock *EndBB, BasicBlock *AfterBB, FinalizeCallbackTy FiniCB,
    ParRegionInfoStruct &ParRegionInfo) {
  // Extract parallel region
  SmallVector<Value *, 16> CapturedVars;
  Function *OutlinedFn =
      createOutlinedFunction(DSAValueMap, VMap, Fn, BBEntry, StartBB, EndBB, AfterBB,
                             CapturedVars, ".omp_outlined_parallel");
  // TODO: remove
  //OutlinedFn->addFnAttr(Attribute::NoInline);

  // Create wrapper for worker threads
  SmallVector<Type *, 2> Params;
  // parallelism level, unused?
  Params.push_back(OMPBuilder.Int16);
  // tid
  Params.push_back(OMPBuilder.Int32);

  FunctionType *OutlinedWrapperFnTy =
      FunctionType::get(OMPBuilder.Void, Params, /* isVarArgs */ false);
  Function *OutlinedWrapperFn =
      Function::Create(OutlinedWrapperFnTy, GlobalValue::InternalLinkage,
                       OutlinedFn->getName() + ".wrapper", M);
  BasicBlock *OutlinedWrapperEntryBB =
      BasicBlock::Create(M.getContext(), "entry", OutlinedWrapperFn);

  // Code generation for the outlined wrapper function.
  OMPBuilder.Builder.SetInsertPoint(OutlinedWrapperEntryBB);

  constexpr const int TIDArgNo = 1;
  AllocaInst *TIDAddr =
      OMPBuilder.Builder.CreateAlloca(OMPBuilder.Int32, nullptr, ".tid.addr");
  AllocaInst *ZeroAddr =
      OMPBuilder.Builder.CreateAlloca(OMPBuilder.Int32, nullptr, ".zero.addr");
  AllocaInst *GlobalArgs = OMPBuilder.Builder.CreateAlloca(
      OMPBuilder.Int8PtrPtr, nullptr, "global_args");

  OMPBuilder.Builder.CreateStore(OutlinedWrapperFn->getArg(TIDArgNo), TIDAddr);
  OMPBuilder.Builder.CreateStore(Constant::getNullValue(OMPBuilder.Int32),
                                 ZeroAddr);
  FunctionCallee KmpcGetSharedVariables = OMPBuilder.getOrCreateRuntimeFunction(
      M, OMPRTL___kmpc_get_shared_variables);
  OMPBuilder.Builder.CreateCall(KmpcGetSharedVariables, {GlobalArgs});

  SmallVector<Value *, 16> OutlinedFnArgs;
  OutlinedFnArgs.push_back(TIDAddr);
  OutlinedFnArgs.push_back(ZeroAddr);

  for (size_t Idx = 0; Idx < CapturedVars.size(); ++Idx) {
    Value *LoadGlobalArgs =
        OMPBuilder.Builder.CreateLoad(OMPBuilder.Int8PtrPtr, GlobalArgs);
    Value *GEP = OMPBuilder.Builder.CreateConstInBoundsGEP1_64(
        OMPBuilder.Int8Ptr, LoadGlobalArgs, Idx);

    // Pass firstprivate scalar by value.
    if (DSAValueMap[CapturedVars[Idx]] == DSA_FIRSTPRIVATE &&
        CapturedVars[Idx]
            ->getType()
            ->getPointerElementType()
            ->isSingleValueType()) {
      Type *VPtrElemTy = CapturedVars[Idx]->getType()->getPointerElementType();
      Value *Bitcast =
          OMPBuilder.Builder.CreateBitCast(GEP, CapturedVars[Idx]->getType());
      Value *Load = OMPBuilder.Builder.CreateLoad(VPtrElemTy, Bitcast);
      OutlinedFnArgs.push_back(Load);

      continue;
    }

    Value *Bitcast = OMPBuilder.Builder.CreateBitCast(
        GEP, CapturedVars[Idx]->getType()->getPointerTo());
    Value *Load =
        OMPBuilder.Builder.CreateLoad(CapturedVars[Idx]->getType(), Bitcast);
    OutlinedFnArgs.push_back(Load);
  }

  OMPBuilder.Builder.CreateCall(OutlinedFn->getFunctionType(), OutlinedFn,
                                OutlinedFnArgs);
  OMPBuilder.Builder.CreateRetVoid();

  if (verifyFunction(*OutlinedWrapperFn, &errs()))
    report_fatal_error("Verification of OutlinedWrapperFn failed!");

  LLVM_DEBUG(dbgs() << "=== Dump OutlinedWrapper\n"
                    << *OutlinedWrapperFn
                    << "=== End of Dump OutlinedWrapper\n");

  // Setup the call to kmpc_parallel_51
  BBEntry->getTerminator()->eraseFromParent();
  OpenMPIRBuilder::LocationDescription Loc(
      InsertPointTy(BBEntry, BBEntry->end()), DL);

  uint32_t SrcLocStrSize;
  Constant *SrcLocStr = OMPBuilder.getOrCreateSrcLocStr(Loc, SrcLocStrSize);
  OMPBuilder.Builder.restoreIP(Loc.IP);
  OMPBuilder.Builder.SetCurrentDebugLocation(Loc.DL);

  // Create the address table of the global data.
  // The number of outlined arguments without global_tid, bound_tid.
  Value *NumCapturedArgs =
      ConstantInt::get(OMPBuilder.SizeTy, CapturedVars.size());
  Type *CapturedVarsAddrsTy =
      ArrayType::get(OMPBuilder.Int8Ptr, CapturedVars.size());

  // TODO: Re-think allocas, move to start of outlined function entry for
  // optimization.
  Value *CapturedVarsAddrs = OMPBuilder.Builder.CreateAlloca(
      CapturedVarsAddrsTy, nullptr, "captured_var_addrs");

  for (size_t Idx = 0; Idx < CapturedVars.size(); ++Idx) {
    LLVM_DEBUG(dbgs() << "CapturedVar " << Idx << " " << *CapturedVars[Idx]
                      << "\n");
    Value *GEP = OMPBuilder.Builder.CreateConstInBoundsGEP2_64(
        CapturedVarsAddrsTy, CapturedVarsAddrs, 0, Idx);

    // Pass firstprivate scalar by value.
    if (DSAValueMap[CapturedVars[Idx]] == DSA_FIRSTPRIVATE &&
        CapturedVars[Idx]
            ->getType()
            ->getPointerElementType()
            ->isSingleValueType()) {
      // TODO: check type conversions.
      Value *BitCast = OMPBuilder.Builder.CreateBitCast(CapturedVars[Idx],
                                                        OMPBuilder.Int64Ptr);
      Value *Load = OMPBuilder.Builder.CreateLoad(OMPBuilder.Int64, BitCast);
      Value *IntToPtr =
          OMPBuilder.Builder.CreateIntToPtr(Load, OMPBuilder.Int8Ptr);
      OMPBuilder.Builder.CreateStore(IntToPtr, GEP);

      continue;
    }

    Value *Bitcast =
        OMPBuilder.Builder.CreateBitCast(CapturedVars[Idx], OMPBuilder.Int8Ptr);
    OMPBuilder.Builder.CreateStore(Bitcast, GEP);
  }

  Value *Ident = OMPBuilder.getOrCreateIdent(SrcLocStr, SrcLocStrSize);
  Value *ThreadID = OMPBuilder.getOrCreateThreadID(Ident);

  Value *IfCondition = ParRegionInfo.IfCondition;
  Value *NumThreads = ParRegionInfo.NumThreads;
  if (!IfCondition)
    // Set condition to 1 (execute in parallel) if not set.
    IfCondition = ConstantInt::get(OMPBuilder.Int32, 1);

  if (!NumThreads)
    NumThreads = ConstantInt::get(OMPBuilder.Int32, -1);

  FunctionCallee KmpcParallel51 =
      OMPBuilder.getOrCreateRuntimeFunction(M, OMPRTL___kmpc_parallel_51);

  // Set proc_bind to -1 by default as it is unused.
  assert(Ident && "Expected non-null Ident");
  assert(ThreadID && "Expected non-null ThreadID");
  assert(IfCondition && "Expected non-null IfCondition");
  assert(NumThreads && "Expected non-null NumThreads");
  assert(OutlinedWrapperFn && "Expected non-null OutlinedWrapperFn");
  assert(CapturedVarsAddrs && "Expected non-null CapturedVarsAddrs");
  assert(NumCapturedArgs && "Expected non-null NumCapturedArgs");

  Value *ProcBind = OMPBuilder.Builder.getInt32(-1);
  Value *OutlinedFnBitcast =
      OMPBuilder.Builder.CreateBitCast(OutlinedFn, OMPBuilder.VoidPtr);
  Value *OutlinedWrapperFnBitcast =
      OMPBuilder.Builder.CreateBitCast(OutlinedWrapperFn, OMPBuilder.VoidPtr);
  Value *CapturedVarAddrsBitcast = OMPBuilder.Builder.CreateBitCast(
      CapturedVarsAddrs, OMPBuilder.VoidPtrPtr);
  OMPBuilder.Builder.CreateCall(
      KmpcParallel51,
      {Ident, ThreadID, IfCondition, NumThreads, ProcBind, OutlinedFnBitcast,
       OutlinedWrapperFnBitcast, CapturedVarAddrsBitcast, NumCapturedArgs});
  OMPBuilder.Builder.CreateBr(AfterBB);

  LLVM_DEBUG(dbgs() << "=== Dump OuterFn\n"
                    << *Fn << "=== End of Dump OuterFn\n");

  if (verifyFunction(*Fn, &errs()))
    report_fatal_error("Verification of OuterFn failed!");
}

FunctionCallee CGIntrinsicsOpenMP::getKmpcForStaticInit(Type *Ty) {
  LLVM_DEBUG(dbgs() << "Type " << *Ty << "\n");
  unsigned Bitwidth = Ty->getIntegerBitWidth();
  LLVM_DEBUG(dbgs() << "Bitwidth " << Bitwidth << "\n");
  if (Bitwidth == 32)
    return OMPBuilder.getOrCreateRuntimeFunction(
        M, OMPRTL___kmpc_for_static_init_4u);
  if (Bitwidth == 64)
    return OMPBuilder.getOrCreateRuntimeFunction(
        M, OMPRTL___kmpc_for_static_init_8u);
  llvm_unreachable("unknown OpenMP loop iterator bitwidth");
}

void CGIntrinsicsOpenMP::emitOMPFor(MapVector<Value *, DSAType> &DSAValueMap,
                                    OMPLoopInfoStruct &OMPLoopInfo,
                                    BasicBlock *StartBB, BasicBlock *ExitBB,
                                    bool IsStandalone) {
  LLVM_DEBUG(dbgs() << "OMPLoopInfo.IV " << *OMPLoopInfo.IV << "\n");
  LLVM_DEBUG(dbgs() << "OMPLoopInfo.UB " << *OMPLoopInfo.UB << "\n");
  assert(OMPLoopInfo.IV && "Expected non-null IV");
  assert(OMPLoopInfo.UB && "Expected non-null UB");

  BasicBlock *PreHeader = StartBB;
  BasicBlock *Header = PreHeader->getUniqueSuccessor();
  BasicBlock *Exit = ExitBB;
  assert(Header && "Expected unique successor from PreHeader to Header");
  LLVM_DEBUG(dbgs() << "=== PreHeader\n"
                    << *PreHeader << "=== End of PreHeader\n");
  LLVM_DEBUG(dbgs() << "=== Header\n" << *Header << "=== End of Header\n");
  LLVM_DEBUG(dbgs() << "=== Exit \n" << *Exit << "=== End of Exit\n");

  Type *IVTy = OMPLoopInfo.IV->getType()->getPointerElementType();
  SmallVector<OpenMPIRBuilder::ReductionInfo> ReductionInfos;

  FunctionCallee KmpcForStaticInit = getKmpcForStaticInit(IVTy);
  FunctionCallee KmpcForStaticFini =
      OMPBuilder.getOrCreateRuntimeFunction(M, OMPRTL___kmpc_for_static_fini);

  const DebugLoc DL = PreHeader->getTerminator()->getDebugLoc();
  OpenMPIRBuilder::LocationDescription Loc(
      InsertPointTy(PreHeader, PreHeader->getTerminator()->getIterator()), DL);
  uint32_t SrcLocStrSize;
  Constant *SrcLocStr = OMPBuilder.getOrCreateSrcLocStr(Loc, SrcLocStrSize);
  Value *SrcLoc = OMPBuilder.getOrCreateIdent(SrcLocStr, SrcLocStrSize);

  // Create allocas for static init values.
  // TODO: Move the AllocaIP to the start of the containing function.
  InsertPointTy AllocaIP(PreHeader, PreHeader->getFirstInsertionPt());
  Type *I32Type = Type::getInt32Ty(M.getContext());
  OMPBuilder.Builder.restoreIP(AllocaIP);
  Value *PLastIter =
      OMPBuilder.Builder.CreateAlloca(I32Type, nullptr, "omp.for.is_last");
  //Value *PStart = OMPBuilder.Builder.CreateAlloca(IVTy, nullptr, "omp.for.start");
  Value *PLowerBound = OMPBuilder.Builder.CreateAlloca(IVTy, nullptr, "omp.for.lb");
  Value *PStride = OMPBuilder.Builder.CreateAlloca(IVTy, nullptr, "omp.for.stride");
  Value *PUpperBound = OMPBuilder.Builder.CreateAlloca(IVTy, nullptr, "omp.for.ub");

  OpenMPIRBuilder::OutlineInfo OI;
  OI.EntryBB = PreHeader;
  OI.ExitBB = Exit;
  SmallPtrSet<BasicBlock *, 8> BlockSet;
  SmallVector<BasicBlock *, 8> BlockVector;
  OI.collectBlocks(BlockSet, BlockVector);

  // TODO: De-duplicate privatization code.
  auto PrivatizeWithReductions = [&]() {
    auto CurrentIP = OMPBuilder.Builder.saveIP();
    for (auto &It : DSAValueMap) {
      Value *Orig = It.first;
      DSAType DSA = It.second;
      Value *ReplacementValue = nullptr;
      Type *VTy = Orig->getType()->getPointerElementType();

      if (DSA == DSA_SHARED)
        continue;

      // Store previous uses to set them to the ReplacementValue after
      // privatization codegen.
      SetVector<Use *> Uses;
      for (Use &U : Orig->uses())
        if (auto *UserI = dyn_cast<Instruction>(U.getUser()))
          if (BlockSet.count(UserI->getParent()))
            Uses.insert(&U);

      OMPBuilder.Builder.restoreIP(AllocaIP);
      if (DSA == DSA_PRIVATE) {
        ReplacementValue = OMPBuilder.Builder.CreateAlloca(
            VTy, /*ArraySize */ nullptr, Orig->getName() + ".for.priv");
        OMPBuilder.Builder.CreateStore(Constant::getNullValue(VTy),
                                       ReplacementValue);
      } else if (DSA == DSA_FIRSTPRIVATE) {
        Value *V = OMPBuilder.Builder.CreateLoad(
            VTy, Orig, Orig->getName() + ".for.firstpriv.reload");
        ReplacementValue = OMPBuilder.Builder.CreateAlloca(
            VTy, /*ArraySize */ nullptr,
            Orig->getName() + ".for.firstpriv.copy");
        OMPBuilder.Builder.CreateStore(V, ReplacementValue);
        // ReplacementValue = Orig;
      } else if (DSA == DSA_REDUCTION_ADD) {
        ReplacementValue = OMPBuilder.Builder.CreateAlloca(
            VTy, /* ArraySize */ nullptr, Orig->getName() + ".red.priv");

        // Store idempotent value based on operation and type.
        // TODO: create templated emitInitAndAppendInfo in CGReduction
        if (VTy->isIntegerTy())
          OMPBuilder.Builder.CreateStore(ConstantInt::get(VTy, 0),
                                         ReplacementValue);
        else if (VTy->isFloatTy() || VTy->isDoubleTy())
          OMPBuilder.Builder.CreateStore(ConstantFP::get(VTy, 0.0),
                                         ReplacementValue);
        else
          report_fatal_error(
              "Unsupported type to init with idempotent reduction value");

        ReductionInfos.push_back(OpenMPIRBuilder::ReductionInfo(VTy, Orig, ReplacementValue,
                                  CGReduction::sumReduction,
                                  CGReduction::sumAtomicReduction));
      } else
        assert(false && "Unsupported privatization");

      assert(ReplacementValue && "Expected non-null ReplacementValue");

      for (Use *UPtr : Uses)
        UPtr->set(ReplacementValue);
    }

    OMPBuilder.Builder.restoreIP(CurrentIP);
  };

  OMPBuilder.Builder.SetInsertPoint(PreHeader->getTerminator());

  // Store the initial normalized upper bound to PUpperBound.
  Value *LoadUB = OMPBuilder.Builder.CreateLoad(IVTy, OMPLoopInfo.UB);
  OMPBuilder.Builder.CreateStore(LoadUB, PUpperBound);

  Constant *One = ConstantInt::get(IVTy, 1);
  // Value *LoadStart = OMPBuilder.Builder.CreateLoad(
  //     IVTy, OMPLoopInfo.Start);
  // OMPBuilder.Builder.CreateStore(LoadStart, PStart);
  Value *LoadLB = OMPBuilder.Builder.CreateLoad(IVTy, OMPLoopInfo.LB);
  OMPBuilder.Builder.CreateStore(LoadLB, PLowerBound);
  OMPBuilder.Builder.CreateStore(One, PStride);

  // If Chunk is not specified (nullptr), default to one, complying with the
  // OpenMP specification.
  if (!OMPLoopInfo.Chunk)
    OMPLoopInfo.Chunk = One;
  Value *ChunkCast = OMPBuilder.Builder.CreateIntCast(OMPLoopInfo.Chunk, IVTy,
                                                      /*isSigned*/ false);

  Value *ThreadNum = OMPBuilder.getOrCreateThreadID(SrcLoc);

  // TODO: add more scheduling types.
  Constant *SchedulingType =
      ConstantInt::get(I32Type, static_cast<int>(OMPLoopInfo.Sched));

  LLVM_DEBUG(dbgs() << "=== SchedulingType " << *SchedulingType << "\n");
  LLVM_DEBUG(dbgs() << "=== PLowerBound " << *PLowerBound << "\n");
  LLVM_DEBUG(dbgs() << "=== PUpperBound " << *PUpperBound << "\n");
  LLVM_DEBUG(dbgs() << "=== PStride " << *PStride << "\n");
  LLVM_DEBUG(dbgs() << "=== Incr " << *One << "\n");
  LLVM_DEBUG(dbgs() << "=== ChunkCast " << *ChunkCast << "\n");
  OMPBuilder.Builder.CreateCall(
      KmpcForStaticInit, {SrcLoc, ThreadNum, SchedulingType, PLastIter,
                          PLowerBound, PUpperBound, PStride, One, ChunkCast});
  // Load returned upper bound to UB.
  Value *LoadPUpperBound = OMPBuilder.Builder.CreateLoad(IVTy, PUpperBound);
  OMPBuilder.Builder.CreateStore(LoadPUpperBound, OMPLoopInfo.UB);
  // Add lower bound to IV.
  Value *LowerBound = OMPBuilder.Builder.CreateLoad(IVTy, PLowerBound);
  OMPBuilder.Builder.CreateStore(LowerBound, OMPLoopInfo.IV);

  // Add fini call, reductions, and barrier after the loop exit block.
  BasicBlock *FiniBB = SplitBlock(Exit, &*Exit->getFirstInsertionPt());
  FiniBB->setName("omp.for.exit");
  BasicBlock *NextFiniBB = SplitBlock(FiniBB, &*FiniBB->getFirstInsertionPt());
  NextFiniBB->setName("omp.for.exit.next");
  OMPBuilder.Builder.SetInsertPoint(FiniBB, FiniBB->getFirstInsertionPt());
  OMPBuilder.Builder.CreateCall(KmpcForStaticFini, {SrcLoc, ThreadNum});

  // Emit reductions, barrier, privatize if standalone.
  if (IsStandalone) {
    PrivatizeWithReductions();
    if (!ReductionInfos.empty())
      OMPBuilder.createReductions(OMPBuilder.Builder.saveIP(), AllocaIP,
                                  ReductionInfos);

    OMPBuilder.Builder.SetInsertPoint(NextFiniBB->getTerminator());
    OMPBuilder.createBarrier(OpenMPIRBuilder::LocationDescription(
                                 OMPBuilder.Builder.saveIP(), Loc.DL),
                             omp::Directive::OMPD_for,
                             /* ForceSimpleCall */ false,
                             /* CheckCancelFlag */ false);
  }

  if (verifyFunction(*PreHeader->getParent(), &errs()))
    report_fatal_error("Verification of omp for lowering failed!");
}

void CGIntrinsicsOpenMP::emitOMPTask(MapVector<Value *, DSAType> &DSAValueMap,
                                     Function *Fn, BasicBlock *BBEntry,
                                     BasicBlock *StartBB, BasicBlock *EndBB,
                                     BasicBlock *AfterBB) {
  // Define types.
  // ************** START TYPE DEFINITION ************** //
  enum {
    TiedFlag = 0x1,
    FinalFlag = 0x2,
    DestructorsFlag = 0x8,
    PriorityFlag = 0x20,
    DetachableFlag = 0x40,
  };

  // This is a union for priority/firstprivate destructors, use the
  // routine entry pointer to allocate space since it is larger than
  // Int32Ty for priority, see kmp.h. Unused for now.
  StructType *KmpCmplrdataTy =
      StructType::create({OMPBuilder.TaskRoutineEntryPtr});
  StructType *KmpTaskTTy =
      StructType::create({OMPBuilder.VoidPtr, OMPBuilder.TaskRoutineEntryPtr,
                          OMPBuilder.Int32, KmpCmplrdataTy, KmpCmplrdataTy},
                         "struct.kmp_task_t");
  Type *KmpTaskTPtrTy = KmpTaskTTy->getPointerTo();

  FunctionCallee KmpcOmpTaskAlloc =
      OMPBuilder.getOrCreateRuntimeFunction(M, OMPRTL___kmpc_omp_task_alloc);
  SmallVector<Type *, 8> SharedsTy;
  SmallVector<Type *, 8> PrivatesTy;
  for (auto &It : DSAValueMap) {
    Value *OriginalValue = It.first;
    if (It.second == DSA_SHARED)
      SharedsTy.push_back(OriginalValue->getType());
    else if (It.second == DSA_PRIVATE || It.second == DSA_FIRSTPRIVATE) {
      assert(isa<PointerType>(OriginalValue->getType()) &&
             "Expected private, firstprivate value with pointer type");
      // Store a copy of the value, thus get the pointer element type.
      PrivatesTy.push_back(OriginalValue->getType()->getPointerElementType());
    } else
      assert(false && "Unknown DSA type");
  }

  StructType *KmpSharedsTTy = nullptr;
  if (SharedsTy.empty())
    KmpSharedsTTy = StructType::create(M.getContext(), "struct.kmp_shareds");
  else
    KmpSharedsTTy = StructType::create(SharedsTy, "struct.kmp_shareds");
  assert(KmpSharedsTTy && "Expected non-null KmpSharedsTTy");
  Type *KmpSharedsTPtrTy = KmpSharedsTTy->getPointerTo();
  StructType *KmpPrivatesTTy =
      StructType::create(PrivatesTy, "struct.kmp_privates");
  Type *KmpPrivatesTPtrTy = KmpPrivatesTTy->getPointerTo();
  StructType *KmpTaskTWithPrivatesTy = StructType::create(
      {KmpTaskTTy, KmpPrivatesTTy}, "struct.kmp_task_t_with_privates");
  Type *KmpTaskTWithPrivatesPtrTy = KmpTaskTWithPrivatesTy->getPointerTo();

  // Declare the task entry function.
  Function *TaskEntryFn = Function::Create(
      OMPBuilder.TaskRoutineEntry, GlobalValue::InternalLinkage,
      Fn->getAddressSpace(), Fn->getName() + ".omp_task_entry", &M);
  // Name arguments.
  TaskEntryFn->getArg(0)->setName(".global_tid");
  TaskEntryFn->getArg(1)->setName(".task_t_with_privates");

  // Declare the task outlined function.
  FunctionType *TaskOutlinedFnTy =
      FunctionType::get(OMPBuilder.Void,
                        {OMPBuilder.Int32, OMPBuilder.Int32Ptr,
                         OMPBuilder.VoidPtr, KmpTaskTPtrTy, KmpSharedsTPtrTy},
                        /*isVarArg=*/false);
  Function *TaskOutlinedFn = Function::Create(
      TaskOutlinedFnTy, GlobalValue::InternalLinkage, Fn->getAddressSpace(),
      Fn->getName() + ".omp_task_outlined", &M);
  TaskOutlinedFn->getArg(0)->setName(".global_tid");
  TaskOutlinedFn->getArg(1)->setName(".part_id");
  TaskOutlinedFn->getArg(2)->setName(".privates");
  TaskOutlinedFn->getArg(3)->setName(".task.data");
  TaskOutlinedFn->getArg(4)->setName(".shareds");

  // ************** END TYPE DEFINITION ************** //

  // Emit kmpc_omp_task_alloc, kmpc_omp_task
  {
    const DebugLoc DL = BBEntry->getTerminator()->getDebugLoc();
    OpenMPIRBuilder::LocationDescription Loc(
        InsertPointTy(BBEntry, BBEntry->getTerminator()->getIterator()), DL);
    uint32_t SrcLocStrSize;
    Constant *SrcLocStr = OMPBuilder.getOrCreateSrcLocStr(Loc, SrcLocStrSize);
    Value *SrcLoc = OMPBuilder.getOrCreateIdent(SrcLocStr, SrcLocStrSize);
    // TODO: parse clauses, for now fix flags to tied
    unsigned TaskFlags = TiedFlag;
    Value *SizeofShareds = nullptr;
    if (KmpSharedsTTy->isEmptyTy())
      SizeofShareds = OMPBuilder.Builder.getInt64(0);
    else
      SizeofShareds = OMPBuilder.Builder.getInt64(
          M.getDataLayout().getTypeAllocSize(KmpSharedsTTy));
    Value *SizeofKmpTaskTWithPrivates = OMPBuilder.Builder.getInt64(
        M.getDataLayout().getTypeAllocSize(KmpTaskTWithPrivatesTy));
    OMPBuilder.Builder.SetInsertPoint(BBEntry, BBEntry->getFirstInsertionPt());
    Value *ThreadNum = OMPBuilder.getOrCreateThreadID(SrcLoc);
    Value *KmpTaskTWithPrivatesVoidPtr = OMPBuilder.Builder.CreateCall(
        KmpcOmpTaskAlloc,
        {SrcLoc, ThreadNum, OMPBuilder.Builder.getInt32(TaskFlags),
         SizeofKmpTaskTWithPrivates, SizeofShareds, TaskEntryFn},
        ".task.data");
    Value *KmpTaskTWithPrivates = OMPBuilder.Builder.CreateBitCast(
        KmpTaskTWithPrivatesVoidPtr, KmpTaskTWithPrivatesPtrTy);

    const unsigned KmpTaskTIdx = 0;
    const unsigned KmpSharedsIdx = 0;
    Value *KmpTaskT = OMPBuilder.Builder.CreateStructGEP(
        KmpTaskTWithPrivatesTy, KmpTaskTWithPrivates, KmpTaskTIdx);
    Value *KmpSharedsGEP =
        OMPBuilder.Builder.CreateStructGEP(KmpTaskTTy, KmpTaskT, KmpSharedsIdx);
    Value *KmpSharedsVoidPtr =
        OMPBuilder.Builder.CreateLoad(OMPBuilder.VoidPtr, KmpSharedsGEP);
    Value *KmpShareds =
        OMPBuilder.Builder.CreateBitCast(KmpSharedsVoidPtr, KmpSharedsTPtrTy);
    const unsigned KmpPrivatesIdx = 1;
    Value *KmpPrivates = OMPBuilder.Builder.CreateStructGEP(
        KmpTaskTWithPrivatesTy, KmpTaskTWithPrivates, KmpPrivatesIdx);

    // Store shareds by reference, firstprivates by value, in task data
    // storage.
    unsigned SharedsGEPIdx = 0;
    unsigned PrivatesGEPIdx = 0;
    for (auto &It : DSAValueMap) {
      Value *OriginalValue = It.first;
      if (It.second == DSA_SHARED) {
        Value *SharedGEP = OMPBuilder.Builder.CreateStructGEP(
            KmpSharedsTTy, KmpShareds, SharedsGEPIdx,
            OriginalValue->getName() + ".task.shared");
        OMPBuilder.Builder.CreateStore(OriginalValue, SharedGEP);
        ++SharedsGEPIdx;
      } else if (It.second == DSA_FIRSTPRIVATE) {
        Value *FirstprivateGEP = OMPBuilder.Builder.CreateStructGEP(
            KmpPrivatesTTy, KmpPrivates, PrivatesGEPIdx,
            OriginalValue->getName() + ".task.firstprivate");
        Value *Load = OMPBuilder.Builder.CreateLoad(
            OriginalValue->getType()->getPointerElementType(), OriginalValue);
        OMPBuilder.Builder.CreateStore(Load, FirstprivateGEP);
        ++PrivatesGEPIdx;
      } else if (It.second == DSA_PRIVATE)
        ++PrivatesGEPIdx;
    }

    FunctionCallee KmpcOmpTask =
        OMPBuilder.getOrCreateRuntimeFunction(M, OMPRTL___kmpc_omp_task);
    OMPBuilder.Builder.CreateCall(
        KmpcOmpTask, {SrcLoc, ThreadNum, KmpTaskTWithPrivatesVoidPtr});
  }

  // Emit task entry function.
  {
    BasicBlock *TaskEntryBB =
        BasicBlock::Create(M.getContext(), "entry", TaskEntryFn);
    OMPBuilder.Builder.SetInsertPoint(TaskEntryBB);
    const unsigned TaskTIdx = 0;
    const unsigned PrivatesIdx = 1;
    const unsigned SharedsIdx = 0;
    Value *GTId = TaskEntryFn->getArg(0);
    Value *KmpTaskTWithPrivates = OMPBuilder.Builder.CreateBitCast(
        TaskEntryFn->getArg(1), KmpTaskTWithPrivatesPtrTy);
    Value *KmpTaskT = OMPBuilder.Builder.CreateStructGEP(
        KmpTaskTWithPrivatesTy, KmpTaskTWithPrivates, TaskTIdx, ".task.data");
    Value *SharedsGEP = OMPBuilder.Builder.CreateStructGEP(
        KmpTaskTTy, KmpTaskT, SharedsIdx, ".shareds.gep");
    Value *SharedsVoidPtr = OMPBuilder.Builder.CreateLoad(
        OMPBuilder.VoidPtr, SharedsGEP, ".shareds.void.ptr");
    Value *Shareds = OMPBuilder.Builder.CreateBitCast(
        SharedsVoidPtr, KmpSharedsTPtrTy, ".shareds");

    Value *Privates = nullptr;
    if (PrivatesTy.empty()) {
      Privates = Constant::getNullValue(OMPBuilder.VoidPtr);
    } else {
      Value *PrivatesTyped = OMPBuilder.Builder.CreateStructGEP(
          KmpTaskTWithPrivatesTy, KmpTaskTWithPrivates, PrivatesIdx,
          ".privates");
      Privates = OMPBuilder.Builder.CreateBitCast(
          PrivatesTyped, OMPBuilder.VoidPtr, ".privates.void.ptr");
    }
    assert(Privates && "Expected non-null privates");

    const unsigned PartIdIdx = 2;
    Value *PartId = OMPBuilder.Builder.CreateStructGEP(KmpTaskTTy, KmpTaskT,
                                                       PartIdIdx, ".part_id");
    OMPBuilder.Builder.CreateCall(TaskOutlinedFnTy, TaskOutlinedFn,
                                  {GTId, PartId, Privates, KmpTaskT, Shareds});
    OMPBuilder.Builder.CreateRet(OMPBuilder.Builder.getInt32(0));
  }

  // Emit TaskOutlinedFn code.
  {
    OpenMPIRBuilder::OutlineInfo OI;
    OI.EntryBB = StartBB;
    OI.ExitBB = EndBB;
    SmallPtrSet<BasicBlock *, 8> OutlinedBlockSet;
    SmallVector<BasicBlock *, 8> OutlinedBlockVector;
    OI.collectBlocks(OutlinedBlockSet, OutlinedBlockVector);
    BasicBlock *TaskOutlinedEntryBB =
        BasicBlock::Create(M.getContext(), "entry", TaskOutlinedFn);
    BasicBlock *TaskOutlinedExitBB =
        BasicBlock::Create(M.getContext(), "exit", TaskOutlinedFn);
    for (BasicBlock *BB : OutlinedBlockVector)
      BB->moveBefore(TaskOutlinedExitBB);
    // Explicitly move EndBB to the outlined functions, since OutlineInfo
    // does not contain it in the OutlinedBlockVector.
    EndBB->moveBefore(TaskOutlinedExitBB);
    EndBB->getTerminator()->setSuccessor(0, TaskOutlinedExitBB);

    OMPBuilder.Builder.SetInsertPoint(TaskOutlinedEntryBB);
    const unsigned KmpPrivatesArgNo = 2;
    const unsigned KmpSharedsArgNo = 4;
    Value *KmpPrivatesArgVoidPtr = TaskOutlinedFn->getArg(KmpPrivatesArgNo);
    Value *KmpPrivatesArg = OMPBuilder.Builder.CreateBitCast(
        KmpPrivatesArgVoidPtr, KmpPrivatesTPtrTy);
    Value *KmpSharedsArg = TaskOutlinedFn->getArg(KmpSharedsArgNo);

    // Replace shareds, privates, firstprivates to refer to task data
    // storage.
    unsigned SharedsGEPIdx = 0;
    unsigned PrivatesGEPIdx = 0;
    for (auto &It : DSAValueMap) {
      Value *OriginalValue = It.first;
      Value *ReplacementValue = nullptr;
      if (It.second == DSA_SHARED) {
        Value *SharedGEP = OMPBuilder.Builder.CreateStructGEP(
            KmpSharedsTTy, KmpSharedsArg, SharedsGEPIdx,
            OriginalValue->getName() + ".task.shared.gep");
        ReplacementValue = OMPBuilder.Builder.CreateLoad(
            OriginalValue->getType(), SharedGEP,
            OriginalValue->getName() + ".task.shared");
        ++SharedsGEPIdx;
      } else if (It.second == DSA_PRIVATE) {
        Value *PrivateGEP = OMPBuilder.Builder.CreateStructGEP(
            KmpPrivatesTTy, KmpPrivatesArg, PrivatesGEPIdx,
            OriginalValue->getName() + ".task.private.gep");
        ReplacementValue = PrivateGEP;
        ++PrivatesGEPIdx;
      } else if (It.second == DSA_FIRSTPRIVATE) {
        Value *FirstprivateGEP = OMPBuilder.Builder.CreateStructGEP(
            KmpPrivatesTTy, KmpPrivatesArg, PrivatesGEPIdx,
            OriginalValue->getName() + ".task.firstprivate.gep");
        ReplacementValue = FirstprivateGEP;
        ++PrivatesGEPIdx;
      } else
        assert(false && "Unknown DSA type");

      assert(ReplacementValue && "Expected non-null ReplacementValue");
      SmallVector<User *, 8> Users(OriginalValue->users());
      for (User *U : Users)
        if (Instruction *I = dyn_cast<Instruction>(U))
          if (OutlinedBlockSet.contains(I->getParent()))
            I->replaceUsesOfWith(OriginalValue, ReplacementValue);
    }

    OMPBuilder.Builder.CreateBr(StartBB);
    OMPBuilder.Builder.SetInsertPoint(TaskOutlinedExitBB);
    OMPBuilder.Builder.CreateRetVoid();
    BBEntry->getTerminator()->setSuccessor(0, AfterBB);
  }
}

void CGIntrinsicsOpenMP::emitOMPOffloadingEntry(const Twine &DevFuncName,
                                                Value *EntryPtr,
                                                Constant *&OMPOffloadEntry) {

  Constant *DevFuncNameConstant =
      ConstantDataArray::getString(M.getContext(), DevFuncName.str());
  auto *GV = new GlobalVariable(
      M, DevFuncNameConstant->getType(),
      /* isConstant */ true, GlobalValue::InternalLinkage, DevFuncNameConstant,
      ".omp_offloading.entry_name", nullptr, GlobalVariable::NotThreadLocal,
      /* AddressSpace */ 0);
  GV->setUnnamedAddr(GlobalValue::UnnamedAddr::Global);

  Constant *EntryConst = dyn_cast<Constant>(EntryPtr);
  assert(EntryConst && "Expected constant entry pointer");
  OMPOffloadEntry = ConstantStruct::get(
      TgtOffloadEntryTy,
      ConstantExpr::getPointerBitCastOrAddrSpaceCast(EntryConst,
                                                     OMPBuilder.VoidPtr),
      ConstantExpr::getPointerBitCastOrAddrSpaceCast(GV, OMPBuilder.Int8Ptr),
      ConstantInt::get(OMPBuilder.SizeTy, 0),
      ConstantInt::get(OMPBuilder.Int32, 0),
      ConstantInt::get(OMPBuilder.Int32, 0));
  auto *OMPOffloadEntryGV = new GlobalVariable(
      M, TgtOffloadEntryTy,
      /* isConstant */ true, GlobalValue::WeakAnyLinkage, OMPOffloadEntry,
      ".omp_offloading.entry." + DevFuncName);
  OMPOffloadEntryGV->setSection("omp_offloading_entries");
  OMPOffloadEntryGV->setAlignment(Align(1));
}

void CGIntrinsicsOpenMP::emitOMPOffloadingMappings(
    InsertPointTy AllocaIP, MapVector<Value *, DSAType> &DSAValueMap,
    MapVector<Value *, SmallVector<FieldMappingInfo, 4>> &StructMappingInfoMap,
    OffloadingMappingArgsTy &OffloadingMappingArgs, bool IsTargetRegion) {

  struct MapperInfo {
    Value *BasePtr;
    Value *Ptr;
    Value *Size;
  };

  SmallVector<MapperInfo, 8> MapperInfos;
  // SmallVector<Constant *, 8> OffloadSizes;
  SmallVector<Constant *, 8> OffloadMapTypes;
  SmallVector<Constant *, 8> OffloadMapNames;

  if (DSAValueMap.empty()) {
    OffloadingMappingArgs.BasePtrs =
        Constant::getNullValue(OMPBuilder.VoidPtrPtr);
    OffloadingMappingArgs.Ptrs = Constant::getNullValue(OMPBuilder.VoidPtrPtr);
    OffloadingMappingArgs.Sizes = Constant::getNullValue(OMPBuilder.Int64Ptr);
    OffloadingMappingArgs.MapTypes =
        Constant::getNullValue(OMPBuilder.Int64Ptr);
    OffloadingMappingArgs.MapNames =
        Constant::getNullValue(OMPBuilder.VoidPtrPtr);

    return;
  }

  auto EmitMappingEntry = [&](Value *Size, uint64_t MapType, Value *BasePtr,
                              Value *Ptr) {
    OffloadMapTypes.push_back(ConstantInt::get(OMPBuilder.SizeTy, MapType));
    // TODO: maybe add debug info.
    uint32_t SrcLocStrSize;
    OffloadMapNames.push_back(
        OMPBuilder.getOrCreateSrcLocStr(BasePtr->getName(), "", 0, 0, SrcLocStrSize));
    LLVM_DEBUG(dbgs() << "Emit mapping entry BasePtr " << *BasePtr << " Ptr "
                      << *Ptr << " Size " << *Size << " MapType " << MapType
                      << "\n");
    MapperInfos.push_back({BasePtr, Ptr, Size});
  };

  auto GetMapType = [IsTargetRegion](DSAType DSA) {
    uint64_t MapType;
    // Determine the map type, completely or partly (structs).
    switch (DSA) {
    case DSA_FIRSTPRIVATE:
      MapType = OMP_TGT_MAPTYPE_LITERAL;
      if (IsTargetRegion)
        MapType |= OMP_TGT_MAPTYPE_TARGET_PARAM;
      break;
    case DSA_MAP_TO:
      MapType = OMP_TGT_MAPTYPE_TO;
      if (IsTargetRegion)
        MapType |= OMP_TGT_MAPTYPE_TARGET_PARAM;
      break;
    case DSA_MAP_FROM:
      MapType = OMP_TGT_MAPTYPE_FROM;
      if (IsTargetRegion)
        MapType |= OMP_TGT_MAPTYPE_TARGET_PARAM;
      break;
    case DSA_MAP_TOFROM:
      MapType = OMP_TGT_MAPTYPE_TO | OMP_TGT_MAPTYPE_FROM;
      if (IsTargetRegion)
        MapType |= OMP_TGT_MAPTYPE_TARGET_PARAM;
      break;
    case DSA_MAP_STRUCT:
      MapType = OMP_TGT_MAPTYPE_NONE;
      if (IsTargetRegion)
        MapType |= OMP_TGT_MAPTYPE_TARGET_PARAM;
      break;
    case DSA_MAP_TO_STRUCT:
      MapType = OMP_TGT_MAPTYPE_TO;
      break;
    case DSA_MAP_FROM_STRUCT:
      MapType = OMP_TGT_MAPTYPE_FROM;
      break;
    case DSA_MAP_TOFROM_STRUCT:
      MapType = OMP_TGT_MAPTYPE_TO | OMP_TGT_MAPTYPE_FROM;
      break;
    case DSA_PRIVATE:
      // do nothing
      break;
    default:
      assert(false && "Unknown mapping type");
      report_fatal_error("Unknown mapping type");
    }

    return MapType;
  };

  // Keep track of argument position, needed for struct mappings.
  for (auto &It : DSAValueMap) {
    Value *V = It.first;
    DSAType DSA = It.second;

    // Emit the mapping entry.
    Value *Size;
    switch (DSA) {
    case DSA_MAP_TO:
    case DSA_MAP_FROM:
    case DSA_MAP_TOFROM:
      Size = ConstantInt::get(OMPBuilder.SizeTy,
                              M.getDataLayout().getTypeAllocSize(V->getType()));
      EmitMappingEntry(Size, GetMapType(DSA), V, V);
      break;
    case DSA_FIRSTPRIVATE: {
      auto *ScalarV = OMPBuilder.Builder.CreateLoad(
          V->getType()->getPointerElementType(), V);
      Size = ConstantInt::get(OMPBuilder.SizeTy,
                              M.getDataLayout().getTypeAllocSize(
                                  V->getType()->getPointerElementType()));
      EmitMappingEntry(Size, GetMapType(DSA), ScalarV, ScalarV);
      break;
    }
    case DSA_MAP_STRUCT: {
      Size = ConstantInt::get(OMPBuilder.SizeTy,
                              M.getDataLayout().getTypeAllocSize(
                                  V->getType()->getPointerElementType()));
      EmitMappingEntry(Size, GetMapType(DSA), V, V);
      // Stores the argument position (starting from 1) of the parent
      // struct, to be used to set MEMBER_OF in the map type.
      size_t ArgPos = MapperInfos.size();

      for (auto &FieldInfo : StructMappingInfoMap[V]) {
        // MEMBER_OF(Argument Position)
        const size_t MemberOfOffset = 48;
        uint64_t MemberOfBits = ArgPos << MemberOfOffset;
        uint64_t FieldMapType = GetMapType(FieldInfo.MapType) | MemberOfBits;
        auto *FieldGEP = OMPBuilder.Builder.CreateInBoundsGEP(
            V->getType()->getPointerElementType(), V,
            {OMPBuilder.Builder.getInt32(0), FieldInfo.Index});

        Value *BasePtr = nullptr;
        Value *Ptr = nullptr;

        if (FieldGEP->getType()->getPointerElementType()->isPointerTy()) {
          FieldMapType |= OMP_TGT_MAPTYPE_PTR_AND_OBJ;
          BasePtr = FieldGEP;
          auto *Load = OMPBuilder.Builder.CreateLoad(
              BasePtr->getType()->getPointerElementType(), BasePtr);
          Ptr = OMPBuilder.Builder.CreateInBoundsGEP(
              Load->getType()->getPointerElementType(), Load, FieldInfo.Offset);
        } else {
          BasePtr = V;
          Ptr = OMPBuilder.Builder.CreateInBoundsGEP(
              FieldGEP->getType()->getPointerElementType(), FieldGEP,
              FieldInfo.Offset);
        }

        assert(BasePtr && "Expected non-null base pointer");
        assert(Ptr && "Expected non-null pointer");

        auto ElementSize = ConstantInt::get(
            OMPBuilder.SizeTy, M.getDataLayout().getTypeAllocSize(
                                   Ptr->getType()->getPointerElementType()));
        Value *NumElements = nullptr;

        // Load the value of NumElements if it is a pointer.
        if (FieldInfo.NumElements->getType()->isPointerTy())
          NumElements = OMPBuilder.Builder.CreateLoad(OMPBuilder.SizeTy,
                                                      FieldInfo.NumElements);
        else
          NumElements = FieldInfo.NumElements;

        auto *Size = OMPBuilder.Builder.CreateMul(ElementSize, NumElements);
        EmitMappingEntry(Size, FieldMapType, BasePtr, Ptr);
      }
      break;
    }
    case DSA_PRIVATE: {
      // do nothing
      break;
    }
    default:
      assert(false && "Unknown mapping type");
      report_fatal_error("Unknown mapping type");
    }
  }

  auto EmitConstantArrayGlobalBitCast = [&](SmallVectorImpl<Constant *> &Vector,
                                            Type *Ty, Type *DestTy,
                                            StringRef Name) {
    auto *Init = ConstantArray::get(ArrayType::get(Ty, Vector.size()), Vector);
    auto *GV = new GlobalVariable(M, ArrayType::get(Ty, Vector.size()),
                                  /* isConstant */ true,
                                  GlobalVariable::PrivateLinkage, Init, Name);
    GV->setUnnamedAddr(GlobalValue::UnnamedAddr::Global);

    return OMPBuilder.Builder.CreateBitCast(GV, DestTy);
  };

  // TODO: offload_sizes can be a global of constants for optimization if all
  // sizes are constants.
  // OffloadingMappingArgs.Sizes =
  //    EmitConstantArrayGlobalBitCast(OffloadSizes, OMPBuilder.SizeTy,
  //                            OMPBuilder.Int64Ptr, ".offload_sizes");
  OffloadingMappingArgs.MapTypes =
      EmitConstantArrayGlobalBitCast(OffloadMapTypes, OMPBuilder.SizeTy,
                                     OMPBuilder.Int64Ptr, ".offload_maptypes");
  OffloadingMappingArgs.MapNames = EmitConstantArrayGlobalBitCast(
      OffloadMapNames, OMPBuilder.Int8Ptr, OMPBuilder.VoidPtrPtr,
      ".offload_mapnames");

  auto EmitArrayAlloca = [&](size_t Size, Type *Ty,
                                                      StringRef Name) {
    InsertPointTy CodeGenIP = OMPBuilder.Builder.saveIP();

    OMPBuilder.Builder.restoreIP(AllocaIP);
    auto *Alloca = OMPBuilder.Builder.CreateAlloca(ArrayType::get(Ty, Size),
                                                   nullptr, Name);

    OMPBuilder.Builder.restoreIP(CodeGenIP);

    return Alloca;
  };

  auto *BasePtrsAlloca = EmitArrayAlloca(MapperInfos.size(), OMPBuilder.VoidPtr,
                                         ".offload_baseptrs");
  auto *PtrsAlloca =
      EmitArrayAlloca(MapperInfos.size(), OMPBuilder.VoidPtr, ".offload_ptrs");
  auto *SizesAlloca =
      EmitArrayAlloca(MapperInfos.size(), OMPBuilder.SizeTy, ".offload_sizes");

  size_t Idx = 0;
  for (auto &MI : MapperInfos) {
    // Store in the base pointers alloca.
    auto *GEP = OMPBuilder.Builder.CreateInBoundsGEP(
        BasePtrsAlloca->getType()->getPointerElementType(), BasePtrsAlloca,
        {OMPBuilder.Builder.getInt32(0), OMPBuilder.Builder.getInt32(Idx)});
    auto *Bitcast = OMPBuilder.Builder.CreateBitCast(
        GEP, MI.BasePtr->getType()->getPointerTo());
    OMPBuilder.Builder.CreateStore(MI.BasePtr, Bitcast);

    // Store in the pointers alloca.
    GEP = OMPBuilder.Builder.CreateInBoundsGEP(
        PtrsAlloca->getType()->getPointerElementType(), PtrsAlloca,
        {OMPBuilder.Builder.getInt32(0), OMPBuilder.Builder.getInt32(Idx)});
    Bitcast = OMPBuilder.Builder.CreateBitCast(
        GEP, MI.Ptr->getType()->getPointerTo());
    OMPBuilder.Builder.CreateStore(MI.Ptr, Bitcast);

    // Store in the sizes alloca.
    GEP = OMPBuilder.Builder.CreateInBoundsGEP(
        SizesAlloca->getType()->getPointerElementType(), SizesAlloca,
        {OMPBuilder.Builder.getInt32(0), OMPBuilder.Builder.getInt32(Idx)});
    Bitcast = OMPBuilder.Builder.CreateBitCast(
        GEP, MI.Size->getType()->getPointerTo());
    OMPBuilder.Builder.CreateStore(MI.Size, Bitcast);

    Idx++;
  }

  OffloadingMappingArgs.Size = MapperInfos.size();
  OffloadingMappingArgs.BasePtrs =
      OMPBuilder.Builder.CreateBitCast(BasePtrsAlloca, OMPBuilder.VoidPtrPtr);
  OffloadingMappingArgs.Ptrs =
      OMPBuilder.Builder.CreateBitCast(PtrsAlloca, OMPBuilder.VoidPtrPtr);
  OffloadingMappingArgs.Sizes = OMPBuilder.Builder.CreateBitCast(
      SizesAlloca, OMPBuilder.SizeTy->getPointerTo());

  // OffloadingMappingArgs.BasePtrs = OMPBuilder.Builder.CreateInBoundsGEP(
  //     BasePtrsAlloca->getType()->getPointerElementType(), BasePtrsAlloca,
  //     {OMPBuilder.Builder.getInt32(0), OMPBuilder.Builder.getInt32(0)});
  // OffloadingMappingArgs.Ptrs = OMPBuilder.Builder.CreateInBoundsGEP(
  //     PtrsAlloca->getType()->getPointerElementType(), PtrsAlloca,
  //     {OMPBuilder.Builder.getInt32(0), OMPBuilder.Builder.getInt32(0)});
  // OffloadingMappingArgs.Sizes = OMPBuilder.Builder.CreateInBoundsGEP(
  //     SizesAlloca->getType()->getPointerElementType(), SizesAlloca,
  //     {OMPBuilder.Builder.getInt32(0), OMPBuilder.Builder.getInt32(0)});
}

void CGIntrinsicsOpenMP::emitOMPSingle(Function *Fn, BasicBlock *BBEntry,
                                       BasicBlock *AfterBB,
                                       BodyGenCallbackTy BodyGenCB,
                                       FinalizeCallbackTy FiniCB) {
  const DebugLoc DL = BBEntry->getTerminator()->getDebugLoc();
  BBEntry->getTerminator()->eraseFromParent();
  // Set the insertion location at the end of the BBEntry.
  OpenMPIRBuilder::LocationDescription Loc(
      InsertPointTy(BBEntry, BBEntry->end()), DL);

  InsertPointTy AfterIP =
      OMPBuilder.createSingle(Loc, BodyGenCB, FiniCB, /*DidIt*/ nullptr);
  BranchInst::Create(AfterBB, AfterIP.getBlock());
  LLVM_DEBUG(dbgs() << "=== Single Fn\n" << *Fn << "=== End of Single Fn\n");
}

void CGIntrinsicsOpenMP::emitOMPCritical(Function *Fn, BasicBlock *BBEntry,
                                         BasicBlock *AfterBB,
                                         BodyGenCallbackTy BodyGenCB,
                                         FinalizeCallbackTy FiniCB) {
  const DebugLoc DL = BBEntry->getTerminator()->getDebugLoc();
  BBEntry->getTerminator()->eraseFromParent();
  // Set the insertion location at the end of the BBEntry.
  OpenMPIRBuilder::LocationDescription Loc(
      InsertPointTy(BBEntry, BBEntry->end()), DL);

  InsertPointTy AfterIP = OMPBuilder.createCritical(Loc, BodyGenCB, FiniCB, "",
                                                    /*HintInst*/ nullptr);
  BranchInst::Create(AfterBB, AfterIP.getBlock());
  LLVM_DEBUG(dbgs() << "=== Critical Fn\n"
                    << *Fn << "=== End of Critical Fn\n");
}

void CGIntrinsicsOpenMP::emitOMPBarrier(Function *Fn, BasicBlock *BBEntry,
                                        Directive DK) {
  const DebugLoc DL = BBEntry->getTerminator()->getDebugLoc();
  // Set the insertion location at the end of the BBEntry.
  OpenMPIRBuilder::LocationDescription Loc(
      InsertPointTy(BBEntry, BBEntry->getTerminator()->getIterator()), DL);

  // TODO: check ForceSimpleCall usage.
  OMPBuilder.createBarrier(Loc, DK,
                           /*ForceSimpleCall*/ false,
                           /*CheckCancelFlag*/ true);
  LLVM_DEBUG(dbgs() << "=== Barrier Fn\n" << *Fn << "=== End of Barrier Fn\n");
}

void CGIntrinsicsOpenMP::emitOMPTaskwait(BasicBlock *BBEntry) {
  const DebugLoc DL = BBEntry->getTerminator()->getDebugLoc();
  // Set the insertion location at the end of the BBEntry.
  OpenMPIRBuilder::LocationDescription Loc(
      InsertPointTy(BBEntry, BBEntry->getTerminator()->getIterator()), DL);

  OMPBuilder.createTaskwait(Loc);
}

GlobalVariable *
CGIntrinsicsOpenMP::emitOffloadingGlobals(StringRef DevWrapperFuncName,
                                          ConstantDataArray *ELF) {
  GlobalVariable *OMPRegionId = nullptr;
  GlobalVariable *OMPOffloadEntries = nullptr;

  // TODO: assumes 1 target region, can we call tgt_register_lib
  // multiple times?
  OMPRegionId = new GlobalVariable(
      M, OMPBuilder.Int8, /* isConstant */ true, GlobalValue::WeakAnyLinkage,
      ConstantInt::get(OMPBuilder.Int8, 0), DevWrapperFuncName + ".region_id",
      nullptr, GlobalVariable::NotThreadLocal,
      /* AddressSpace */ 0);

  Constant *OMPOffloadEntry;
  CGIntrinsicsOpenMP::emitOMPOffloadingEntry(DevWrapperFuncName, OMPRegionId,
                                             OMPOffloadEntry);

  // TODO: do this at finalization when all entries have been
  // found.
  // TODO: assumes 1 device image, can we call tgt_register_lib
  // multiple times?
  auto *ArrayTy = ArrayType::get(TgtOffloadEntryTy, 1);
  OMPOffloadEntries =
      new GlobalVariable(M, ArrayTy,
                         /* isConstant */ true, GlobalValue::ExternalLinkage,
                         ConstantArray::get(ArrayTy, {OMPOffloadEntry}),
                         ".omp_offloading.entries");

  assert(OMPRegionId && "Expected non-null omp region id global");
  assert(OMPOffloadEntries &&
         "Expected non-null omp offloading entries constant");

  auto EmitOffloadingBinaryGlobals = [&]() {
    auto *GV = new GlobalVariable(M, ELF->getType(), /* isConstant */ true,
                                  GlobalValue::InternalLinkage, ELF,
                                  ".omp_offloading.device_image");
    GV->setUnnamedAddr(GlobalValue::UnnamedAddr::Global);

    StructType *TgtDeviceImageTy = StructType::create(
        {OMPBuilder.Int8Ptr, OMPBuilder.Int8Ptr,
         TgtOffloadEntryTy->getPointerTo(), TgtOffloadEntryTy->getPointerTo()},
        "struct.__tgt_device_image");

    StructType *TgtBinDescTy = StructType::create(
        {OMPBuilder.Int32, TgtDeviceImageTy->getPointerTo(),
         TgtOffloadEntryTy->getPointerTo(), TgtOffloadEntryTy->getPointerTo()},
        "struct.__tgt_bin_desc");

    auto *ArrayTy = ArrayType::get(TgtDeviceImageTy, 1);
    auto *Zero = ConstantInt::get(OMPBuilder.SizeTy, 0);
    auto *One = ConstantInt::get(OMPBuilder.SizeTy, 1);
    auto *Size = ConstantInt::get(OMPBuilder.SizeTy, ELF->getNumElements());
    Constant *ZeroZero[] = {Zero, Zero};
    Constant *ZeroOne[] = {Zero, One};
    Constant *ZeroSize[] = {Zero, Size};

    auto *ImageB =
        ConstantExpr::getGetElementPtr(GV->getValueType(), GV, ZeroZero);
    auto *ImageE =
        ConstantExpr::getGetElementPtr(GV->getValueType(), GV, ZeroSize);
    auto *EntriesB = ConstantExpr::getGetElementPtr(
        OMPOffloadEntries->getValueType(), OMPOffloadEntries, ZeroZero);
    auto *EntriesE = ConstantExpr::getGetElementPtr(
        OMPOffloadEntries->getValueType(), OMPOffloadEntries, ZeroOne);

    auto *DeviceImageEntry = ConstantStruct::get(TgtDeviceImageTy, ImageB,
                                                 ImageE, EntriesB, EntriesE);
    auto *DeviceImages =
        new GlobalVariable(M, ArrayTy,
                           /* isConstant */ true, GlobalValue::InternalLinkage,
                           ConstantArray::get(ArrayTy, {DeviceImageEntry}),
                           ".omp_offloading.device_images");

    auto *ImagesB = ConstantExpr::getGetElementPtr(DeviceImages->getValueType(),
                                                   DeviceImages, ZeroZero);
    auto *DescInit =
        ConstantStruct::get(TgtBinDescTy,
                            ConstantInt::get(OMPBuilder.Int32,
                                             /* number of images */ 1),
                            ImagesB, EntriesB, EntriesE);
    auto *BinDesc =
        new GlobalVariable(M, DescInit->getType(),
                           /* isConstant */ true, GlobalValue::InternalLinkage,
                           DescInit, ".omp_offloading.descriptor");

    // Add tgt_register_requires, tgt_register_lib,
    // tgt_unregister_lib.
    {
      // tgt_register_requires.
      auto *FuncTy = FunctionType::get(OMPBuilder.Void, /*isVarArg*/ false);
      auto *Func = Function::Create(FuncTy, GlobalValue::InternalLinkage,
                                    ".omp_offloading.requires_reg", &M);
      Func->setSection(".text.startup");

      // Get __tgt_register_lib function declaration.
      auto *RegFuncTy = FunctionType::get(OMPBuilder.Void, OMPBuilder.Int64,
                                          /*isVarArg*/ false);
      FunctionCallee RegFuncC =
          M.getOrInsertFunction("__tgt_register_requires", RegFuncTy);

      // Construct function body
      IRBuilder<> Builder(BasicBlock::Create(M.getContext(), "entry", Func));
      // TODO: fix to pass the requirements enum value.
      Builder.CreateCall(RegFuncC, ConstantInt::get(OMPBuilder.Int64, 1));
      Builder.CreateRetVoid();

      // Add this function to constructors.
      // Set priority to 1 so that __tgt_register_lib is executed
      // AFTER
      // __tgt_register_requires (we want to know what requirements
      // have been asked for before we load a libomptarget plugin so
      // that by the time the plugin is loaded it can report how
      // many devices there are which can satisfy these
      // requirements).
      appendToGlobalCtors(M, Func, /*Priority*/ 0);
    }
    {
      // ctor
      auto *FuncTy = FunctionType::get(OMPBuilder.Void, /*isVarArg*/ false);
      auto *Func = Function::Create(FuncTy, GlobalValue::InternalLinkage,
                                    ".omp_offloading.descriptor_reg", &M);
      Func->setSection(".text.startup");

      // Get __tgt_register_lib function declaration.
      auto *RegFuncTy =
          FunctionType::get(OMPBuilder.Void, TgtBinDescTy->getPointerTo(),
                            /*isVarArg*/ false);
      FunctionCallee RegFuncC =
          M.getOrInsertFunction("__tgt_register_lib", RegFuncTy);

      // Construct function body
      IRBuilder<> Builder(BasicBlock::Create(M.getContext(), "entry", Func));
      Builder.CreateCall(RegFuncC, BinDesc);
      Builder.CreateRetVoid();

      // Add this function to constructors.
      // Set priority to 1 so that __tgt_register_lib is executed
      // AFTER
      // __tgt_register_requires (we want to know what requirements
      // have been asked for before we load a libomptarget plugin so
      // that by the time the plugin is loaded it can report how
      // many devices there are which can satisfy these
      // requirements).
      appendToGlobalCtors(M, Func, /*Priority*/ 1);
    }
    {
      auto *FuncTy = FunctionType::get(OMPBuilder.Void, /*isVarArg*/ false);
      auto *Func = Function::Create(FuncTy, GlobalValue::InternalLinkage,
                                    ".omp_offloading.descriptor_unreg", &M);
      Func->setSection(".text.startup");

      // Get __tgt_unregister_lib function declaration.
      auto *UnRegFuncTy =
          FunctionType::get(OMPBuilder.Void, TgtBinDescTy->getPointerTo(),
                            /*isVarArg*/ false);
      FunctionCallee UnRegFuncC =
          M.getOrInsertFunction("__tgt_unregister_lib", UnRegFuncTy);

      // Construct function body
      IRBuilder<> Builder(BasicBlock::Create(M.getContext(), "entry", Func));
      Builder.CreateCall(UnRegFuncC, BinDesc);
      Builder.CreateRetVoid();

      // Add this function to global destructors.
      // Match priority of __tgt_register_lib
      appendToGlobalDtors(M, Func, /*Priority*/ 1);
    }
  };

  EmitOffloadingBinaryGlobals();

  return OMPRegionId;
}

void CGIntrinsicsOpenMP::emitOMPTarget(
    Function *Fn, BasicBlock *EntryBB, BasicBlock *StartBB, BasicBlock *EndBB,
    MapVector<Value *, DSAType> &DSAValueMap,
    MapVector<Value *, SmallVector<FieldMappingInfo, 4>> &StructMappingInfoMap,
    TargetInfoStruct &TargetInfo, bool IsDeviceTargetRegion) {
  if (IsDeviceTargetRegion)
    emitOMPTargetDevice(Fn, EntryBB, StartBB, EndBB, DSAValueMap,
                        StructMappingInfoMap, TargetInfo);
  else
    emitOMPTargetHost(Fn, EntryBB, StartBB, EndBB, DSAValueMap,
                  StructMappingInfoMap, TargetInfo);
}

void CGIntrinsicsOpenMP::emitOMPTargetHost(
    Function *Fn, BasicBlock *EntryBB, BasicBlock *StartBB, BasicBlock *EndBB,
    MapVector<Value *, DSAType> &DSAValueMap,
    MapVector<Value *, SmallVector<FieldMappingInfo, 4>> &StructMappingInfoMap,
    TargetInfoStruct &TargetInfo) {

  Twine DevWrapperFuncName = getDevWrapperFuncPrefix() + TargetInfo.DevFuncName;

  GlobalVariable *OMPRegionId =
      emitOffloadingGlobals(DevWrapperFuncName.str(), TargetInfo.ELF);

  const DebugLoc DL = EntryBB->getTerminator()->getDebugLoc();
  OpenMPIRBuilder::LocationDescription Loc(
      InsertPointTy(EntryBB, EntryBB->getTerminator()->getIterator()), DL);
  uint32_t SrcLocStrSize;
  Constant *SrcLocStr = OMPBuilder.getOrCreateSrcLocStr(Loc, SrcLocStrSize);
  Value *SrcLoc = OMPBuilder.getOrCreateIdent(SrcLocStr, SrcLocStrSize);

  // TODO: should we use target_mapper without teams or the more general
  // target_teams_mapper. Does the former buy us anything (less overhead?)
  //FunctionCallee TargetMapper =
  //    OMPBuilder.getOrCreateRuntimeFunction(M, OMPRTL___tgt_target_mapper);
  FunctionCallee TargetMapper =
      OMPBuilder.getOrCreateRuntimeFunction(M, OMPRTL___tgt_target_teams_mapper);
  OMPBuilder.Builder.SetInsertPoint(EntryBB->getTerminator());

  // Emit mappings.
  OffloadingMappingArgsTy OffloadingMappingArgs;
  InsertPointTy AllocaIP(&Fn->getEntryBlock(),
                         Fn->getEntryBlock().getFirstInsertionPt());
  emitOMPOffloadingMappings(AllocaIP, DSAValueMap, StructMappingInfoMap,
                            OffloadingMappingArgs, /* isTargetRegion */ true);

  auto *OffloadResult = OMPBuilder.Builder.CreateCall(
      TargetMapper,
      {SrcLoc, ConstantInt::get(OMPBuilder.Int64, -1),
       ConstantExpr::getBitCast(OMPRegionId, OMPBuilder.Int8Ptr),
       ConstantInt::get(OMPBuilder.Int32, OffloadingMappingArgs.Size),
       OffloadingMappingArgs.BasePtrs, OffloadingMappingArgs.Ptrs,
       OffloadingMappingArgs.Sizes, OffloadingMappingArgs.MapTypes,
       OffloadingMappingArgs.MapNames,
       // TODO: offload_mappers is null for now.
       Constant::getNullValue(OMPBuilder.VoidPtrPtr), TargetInfo.NumTeams,
       TargetInfo.ThreadLimit});
  auto *Failed = OMPBuilder.Builder.CreateIsNotNull(OffloadResult);
  OMPBuilder.Builder.CreateCondBr(Failed, StartBB, EndBB);
  EntryBB->getTerminator()->eraseFromParent();
}

void CGIntrinsicsOpenMP::emitOMPTargetDevice(
    Function *Fn, BasicBlock *EntryBB, BasicBlock *StartBB, BasicBlock *EndBB,
    MapVector<Value *, DSAType> &DSAValueMap,
    MapVector<Value *, SmallVector<FieldMappingInfo, 4>> &StructMappingInfoMap,
    TargetInfoStruct &TargetInfo) {
  // Emit the Numba wrapper offloading function.
  SmallVector<Type *, 8> WrapperArgsTypes;
  SmallVector<StringRef, 8> WrapperArgsNames;
  for (auto &It : DSAValueMap) {
    Value *V = It.first;
    DSAType DSA = It.second;

    LLVM_DEBUG(dbgs() << "V " << *V << " DSA " << DSA << "\n");
    switch (DSA) {
    case DSA_FIRSTPRIVATE:
      WrapperArgsTypes.push_back(V->getType()->getPointerElementType());
      WrapperArgsNames.push_back(V->getName());
      break;
    case DSA_PRIVATE:
      // do nothing
      break;
    default:
      WrapperArgsTypes.push_back(V->getType());
      WrapperArgsNames.push_back(V->getName());
    }
  }

  for(auto &Arg : WrapperArgsNames)
    LLVM_DEBUG(dbgs() << "[IOMP] Adding wrapper arg " << Arg << "\n");
  for(auto &Arg : WrapperArgsTypes) {
    LLVM_DEBUG(dbgs() << "[IOMP] Arg type ");
    LLVM_DEBUG(Arg->print(dbgs()));
    LLVM_DEBUG(dbgs() << "\n");
  }

  Twine DevWrapperFuncName = getDevWrapperFuncPrefix() + Fn->getName();
  FunctionType *NumbaWrapperFnTy =
      FunctionType::get(OMPBuilder.Void, WrapperArgsTypes,
                        /* isVarArg */ false);
  Function *NumbaWrapperFunc = Function::Create(
      NumbaWrapperFnTy, GlobalValue::ExternalLinkage, DevWrapperFuncName, M);

  // Name the wrapper arguments for readability.
  for (size_t I = 0; I < NumbaWrapperFunc->arg_size(); ++I)
    NumbaWrapperFunc->getArg(I)->setName(WrapperArgsNames[I]);

  IRBuilder<> Builder(
      BasicBlock::Create(M.getContext(), "entry", NumbaWrapperFunc));
  // Set up default arguments. Depends on the target architecture.
  FunctionCallee DevFuncCallee(Fn);
  SmallVector<Value *, 8> DevFuncArgs;
  Triple TargetTriple(M.getTargetTriple());

  for (auto &Arg : NumbaWrapperFunc->args())
    DevFuncArgs.push_back(&Arg);

  if (isOpenMPDeviceRuntime()) {
    OpenMPIRBuilder::LocationDescription Loc(Builder);
    auto IP = OMPBuilder.createTargetInit(Loc, /* IsSPMD */ false,
                                          /* RequiresFullRuntime */ true);
    Builder.restoreIP(IP);
  }

  Builder.CreateCall(DevFuncCallee, DevFuncArgs);

  if (isOpenMPDeviceRuntime()) {
    OpenMPIRBuilder::LocationDescription Loc(Builder);
    OMPBuilder.createTargetDeinit(Loc, /* IsSPMD */ false,
                                  /* RequiresFullRuntime */ true);
  }

  Builder.CreateRetVoid();

  if (isOpenMPDeviceRuntime()) {
    constexpr int OMP_TGT_GENERIC_EXEC_MODE = 1;
    // Emit OMP device globals and metadata.
    auto *ExecModeGV = new GlobalVariable(
        M, OMPBuilder.Int8, /* isConstant */ false, GlobalValue::WeakAnyLinkage,
        Builder.getInt8(OMP_TGT_GENERIC_EXEC_MODE),
        DevWrapperFuncName + "_exec_mode");
    appendToCompilerUsed(M, {ExecModeGV});

    // Get "nvvm.annotations" metadata node.
    // TODO: may need to adjust for AMD gpus.
    NamedMDNode *MD = M.getOrInsertNamedMetadata("nvvm.annotations");

    Metadata *MDVals[] = {
        ConstantAsMetadata::get(NumbaWrapperFunc),
        MDString::get(M.getContext(), "kernel"),
        ConstantAsMetadata::get(ConstantInt::get(OMPBuilder.Int32, 1))};
    // Append metadata to nvvm.annotations.
    MD->addOperand(MDNode::get(M.getContext(), MDVals));

    // Add a function attribute for the kernel.
    Fn->addFnAttr(Attribute::get(M.getContext(), "kernel"));

  } else {
    // Generating an offloading entry is required by the x86_64 plugin.
    Constant *OMPOffloadEntry;
    emitOMPOffloadingEntry(DevWrapperFuncName, NumbaWrapperFunc,
                           OMPOffloadEntry);
  }
  // Add llvm.module.flags for "openmp", "openmp-device" to enable
  // OpenMPOpt.
  M.addModuleFlag(llvm::Module::Max, "openmp", 50);
  M.addModuleFlag(llvm::Module::Max, "openmp-device", 50);
}

void CGIntrinsicsOpenMP::emitOMPTeamsDeviceRuntime(
    MapVector<Value *, DSAType> &DSAValueMap, ValueToValueMapTy *VMap,
    const DebugLoc &DL, Function *Fn, BasicBlock *BBEntry, BasicBlock *StartBB,
    BasicBlock *EndBB, BasicBlock *AfterBB, TeamsInfoStruct &TeamsInfo) {
  SmallVector<Value *, 16> CapturedVars;
  Function *OutlinedFn =
      createOutlinedFunction(DSAValueMap, VMap, Fn, BBEntry, StartBB, EndBB,
                             AfterBB, CapturedVars, ".omp_outlined_teams");
  // TODO: remove
  //OutlinedFn->addFnAttr(Attribute::NoInline);

  // Set up the call to the teams outlined function.
  BBEntry->getTerminator()->eraseFromParent();
  OpenMPIRBuilder::LocationDescription Loc(
      InsertPointTy(BBEntry, BBEntry->end()), DL);

  uint32_t SrcLocStrSize;
  Constant *SrcLocStr = OMPBuilder.getOrCreateSrcLocStr(Loc, SrcLocStrSize);
  OMPBuilder.Builder.restoreIP(Loc.IP);
  OMPBuilder.Builder.SetCurrentDebugLocation(Loc.DL);

  Value *Ident = OMPBuilder.getOrCreateIdent(SrcLocStr, SrcLocStrSize);
  Value *ThreadID = OMPBuilder.getOrCreateThreadID(Ident);

  assert(Ident && "Expected non-null Ident");
  assert(ThreadID && "Expected non-null ThreadID");

  // Create global_tid, bound_tid (zero) to pass to the teams outlined function.
  AllocaInst *ThreadIDAddr =
      OMPBuilder.Builder.CreateAlloca(OMPBuilder.Int32, nullptr, ".threadid.addr");
  AllocaInst *ZeroAddr =
      OMPBuilder.Builder.CreateAlloca(OMPBuilder.Int32, nullptr, ".zero.addr");
  OMPBuilder.Builder.CreateStore(ThreadID, ThreadIDAddr);
  OMPBuilder.Builder.CreateStore(Constant::getNullValue(OMPBuilder.Int32),
                                 ZeroAddr);

  FunctionCallee TeamsOutlinedFn(OutlinedFn);
  SmallVector<Value *, 8> Args;
  Args.append({ThreadIDAddr, ZeroAddr});

  for (size_t Idx = 0; Idx < CapturedVars.size(); ++Idx) {
      // Pass firstprivate scalar by value.
    if (DSAValueMap[CapturedVars[Idx]] == DSA_FIRSTPRIVATE &&
        CapturedVars[Idx]
            ->getType()
            ->getPointerElementType()
            ->isSingleValueType()) {
      Type *VPtrElemTy = CapturedVars[Idx]->getType()->getPointerElementType();
      Value *Load = OMPBuilder.Builder.CreateLoad(VPtrElemTy, CapturedVars[Idx]);
      Args.push_back(Load);

      continue;
    }
    Args.push_back(CapturedVars[Idx]);
  }

  OMPBuilder.Builder.CreateCall(TeamsOutlinedFn, Args);

  OMPBuilder.Builder.CreateBr(AfterBB);

  LLVM_DEBUG(dbgs() << "=== Dump OuterFn\n"
                    << *Fn << "=== End of Dump OuterFn\n");

  if (verifyFunction(*Fn, &errs()))
    report_fatal_error("Verification of OuterFn failed!");
}
void CGIntrinsicsOpenMP::emitOMPTeams(MapVector<Value *, DSAType> &DSAValueMap,
                                      ValueToValueMapTy *VMap,
                                      const DebugLoc &DL, Function *Fn,
                                      BasicBlock *BBEntry, BasicBlock *StartBB,
                                      BasicBlock *EndBB, BasicBlock *AfterBB,
                                      TeamsInfoStruct &TeamsInfo) {
  if (isOpenMPDeviceRuntime())
    emitOMPTeamsDeviceRuntime(DSAValueMap, nullptr, DL, Fn, BBEntry, StartBB,
                              EndBB, AfterBB, TeamsInfo);
  else
    emitOMPTeamsHostRuntime(DSAValueMap, nullptr, DL, Fn, BBEntry, StartBB,
                            EndBB, AfterBB, TeamsInfo);
}

void CGIntrinsicsOpenMP::emitOMPTeamsHostRuntime(MapVector<Value *, DSAType> &DSAValueMap,
                                      ValueToValueMapTy *VMap,
                                      const DebugLoc &DL, Function *Fn,
                                      BasicBlock *BBEntry, BasicBlock *StartBB,
                                      BasicBlock *EndBB, BasicBlock *AfterBB,
                                      TeamsInfoStruct &TeamsInfo) {
  SmallVector<Value *, 16> CapturedVars;
  Function *OutlinedFn = createOutlinedFunction(
      DSAValueMap, /*ValueToValueMapTy */ VMap, Fn, BBEntry, StartBB, EndBB,
      AfterBB, CapturedVars, ".omp_outlined_teams");
  // TODO: remove
  //OutlinedFn->addFnAttr(Attribute::NoInline);

  // Set up the call to the teams outlined function.
  BBEntry->getTerminator()->eraseFromParent();
  OpenMPIRBuilder::LocationDescription Loc(
      InsertPointTy(BBEntry, BBEntry->end()), DL);

  uint32_t SrcLocStrSize;
  Constant *SrcLocStr = OMPBuilder.getOrCreateSrcLocStr(Loc, SrcLocStrSize);
  OMPBuilder.Builder.restoreIP(Loc.IP);
  OMPBuilder.Builder.SetCurrentDebugLocation(Loc.DL);

  Value *Ident = OMPBuilder.getOrCreateIdent(SrcLocStr, SrcLocStrSize);
  Value *ThreadID = OMPBuilder.getOrCreateThreadID(Ident);

  assert(Ident && "Expected non-null Ident");

  // Emit call to set the number of teams and thread limit.
  if (TeamsInfo.NumTeams || TeamsInfo.ThreadLimit) {
    Value *NumTeams =
        (TeamsInfo.NumTeams ? TeamsInfo.NumTeams
                            : Constant::getNullValue(OMPBuilder.Int32));
    Value *ThreadLimit =
        (TeamsInfo.ThreadLimit ? TeamsInfo.ThreadLimit
                               : Constant::getNullValue(OMPBuilder.Int32));
    FunctionCallee KmpcPushNumTeams =
        OMPBuilder.getOrCreateRuntimeFunction(M, OMPRTL___kmpc_push_num_teams);
    OMPBuilder.Builder.CreateCall(KmpcPushNumTeams,
                                  {Ident, ThreadID, NumTeams, ThreadLimit});
  }

  FunctionCallee ForkTeams =
      OMPBuilder.getOrCreateRuntimeFunction(M, OMPRTL___kmpc_fork_teams);

  SmallVector<Value *, 8> Args;
  Value *NumCapturedVars = OMPBuilder.Builder.getInt32(CapturedVars.size());
  Args.append({Ident, NumCapturedVars,
               OMPBuilder.Builder.CreateBitCast(OutlinedFn,
                                                OMPBuilder.ParallelTaskPtr)});

    for (size_t Idx = 0; Idx < CapturedVars.size(); ++Idx) {
      // Pass firstprivate scalar by value.
    if (DSAValueMap[CapturedVars[Idx]] == DSA_FIRSTPRIVATE &&
        CapturedVars[Idx]
            ->getType()
            ->getPointerElementType()
            ->isSingleValueType()) {
      Type *VPtrElemTy = CapturedVars[Idx]->getType()->getPointerElementType();
      Value *Load = OMPBuilder.Builder.CreateLoad(VPtrElemTy, CapturedVars[Idx]);
      Args.push_back(Load);

      continue;
    }
    Args.push_back(CapturedVars[Idx]);
  }

  OMPBuilder.Builder.CreateCall(ForkTeams, Args);

  OMPBuilder.Builder.CreateBr(AfterBB);

  LLVM_DEBUG(dbgs() << "=== Dump OuterFn\n"
                    << *Fn << "=== End of Dump OuterFn\n");

  if (verifyFunction(*Fn, &errs()))
    report_fatal_error("Verification of OuterFn failed!");
}

void CGIntrinsicsOpenMP::emitOMPTargetEnterData(
    Function *Fn, BasicBlock *BBEntry, MapVector<Value *, DSAType> &DSAValueMap,
    MapVector<Value *, SmallVector<FieldMappingInfo, 4>>
        &StructMappingInfoMap) {

  const DebugLoc DL = BBEntry->getTerminator()->getDebugLoc();
  OpenMPIRBuilder::LocationDescription Loc(
      InsertPointTy(BBEntry, BBEntry->getTerminator()->getIterator()), DL);
  uint32_t SrcLocStrSize;
  Constant *SrcLocStr = OMPBuilder.getOrCreateSrcLocStr(Loc, SrcLocStrSize);
  Value *SrcLoc = OMPBuilder.getOrCreateIdent(SrcLocStr, SrcLocStrSize);

  FunctionCallee TargetDataBeginMapper =
      OMPBuilder.getOrCreateRuntimeFunction(M, OMPRTL___tgt_target_data_begin_mapper);
  OMPBuilder.Builder.SetInsertPoint(BBEntry->getTerminator());

  // Emit mappings.
  OffloadingMappingArgsTy OffloadingMappingArgs;
  InsertPointTy AllocaIP(&Fn->getEntryBlock(),
                         Fn->getEntryBlock().getFirstInsertionPt());
  emitOMPOffloadingMappings(AllocaIP, DSAValueMap, StructMappingInfoMap,
                            OffloadingMappingArgs, /* IsTargetRegion */ false);

  OMPBuilder.Builder.CreateCall(
      TargetDataBeginMapper,
      {SrcLoc, ConstantInt::get(OMPBuilder.Int64, -1),
       ConstantInt::get(OMPBuilder.Int32, OffloadingMappingArgs.Size),
       OffloadingMappingArgs.BasePtrs, OffloadingMappingArgs.Ptrs,
       OffloadingMappingArgs.Sizes, OffloadingMappingArgs.MapTypes,
       OffloadingMappingArgs.MapNames,
       // TODO: offload_mappers is null for now.
       Constant::getNullValue(OMPBuilder.VoidPtrPtr)});
}

void CGIntrinsicsOpenMP::emitOMPTargetExitData(
    Function *Fn, BasicBlock *BBEntry, MapVector<Value *, DSAType> &DSAValueMap,
    MapVector<Value *, SmallVector<FieldMappingInfo, 4>>
        &StructMappingInfoMap) {

  const DebugLoc DL = BBEntry->getTerminator()->getDebugLoc();
  OpenMPIRBuilder::LocationDescription Loc(
      InsertPointTy(BBEntry, BBEntry->getTerminator()->getIterator()), DL);
  uint32_t SrcLocStrSize;
  Constant *SrcLocStr = OMPBuilder.getOrCreateSrcLocStr(Loc, SrcLocStrSize);
  Value *SrcLoc = OMPBuilder.getOrCreateIdent(SrcLocStr, SrcLocStrSize);

  FunctionCallee TargetDataEndMapper =
      OMPBuilder.getOrCreateRuntimeFunction(M, OMPRTL___tgt_target_data_end_mapper);
  OMPBuilder.Builder.SetInsertPoint(BBEntry->getTerminator());

  // Emit mappings.
  OffloadingMappingArgsTy OffloadingMappingArgs;
  InsertPointTy AllocaIP(&Fn->getEntryBlock(),
                         Fn->getEntryBlock().getFirstInsertionPt());
  emitOMPOffloadingMappings(AllocaIP, DSAValueMap, StructMappingInfoMap,
                            OffloadingMappingArgs, /* IsTargetRegion */ false);

  OMPBuilder.Builder.CreateCall(
      TargetDataEndMapper,
      {SrcLoc, ConstantInt::get(OMPBuilder.Int64, -1),
       ConstantInt::get(OMPBuilder.Int32, OffloadingMappingArgs.Size),
       OffloadingMappingArgs.BasePtrs, OffloadingMappingArgs.Ptrs,
       OffloadingMappingArgs.Sizes, OffloadingMappingArgs.MapTypes,
       OffloadingMappingArgs.MapNames,
       // TODO: offload_mappers is null for now.
       Constant::getNullValue(OMPBuilder.VoidPtrPtr)});
}

void CGIntrinsicsOpenMP::emitOMPDistribute(
    MapVector<Value *, DSAType> &DSAValueMap, BasicBlock *StartBB,
    BasicBlock *ExitBB, OMPLoopInfoStruct &OMPLoopInfo, bool IsStandalone) {
  // TODO: de-duplicate code, there is large overlap with omp for code
  // generation.
  LLVM_DEBUG(dbgs() << "OMPLoopInfo.IV " << *OMPLoopInfo.IV << "\n");
  LLVM_DEBUG(dbgs() << "OMPLoopInfo.UB " << *OMPLoopInfo.UB << "\n");
  assert(OMPLoopInfo.IV && "Expected non-null IV");
  assert(OMPLoopInfo.UB && "Expected non-null UB");

  BasicBlock *PreHeader = StartBB;
  BasicBlock *LoopHeader = PreHeader->getUniqueSuccessor();
  BasicBlock *LoopExit = ExitBB;
  assert(LoopHeader && "Expected unique successor from PreHeader to LoopHeader");
  LLVM_DEBUG(dbgs() << "=== PreHeader\n"
                    << *PreHeader << "=== End of PreHeader\n");
  LLVM_DEBUG(dbgs() << "=== LoopHeader\n"
                    << *LoopHeader << "=== End of LoopHeader\n");
  LLVM_DEBUG(dbgs() << "=== LoopExit \n"
                    << *LoopExit << "=== End of LoopExit\n");
  Type *IVTy = OMPLoopInfo.IV->getType()->getPointerElementType();

  FunctionCallee KmpcForStaticInit = getKmpcForStaticInit(IVTy);
  FunctionCallee KmpcForStaticFini =
      OMPBuilder.getOrCreateRuntimeFunction(M, OMPRTL___kmpc_for_static_fini);

  const DebugLoc DL = PreHeader->getTerminator()->getDebugLoc();
  OpenMPIRBuilder::LocationDescription Loc(
      InsertPointTy(PreHeader, PreHeader->getTerminator()->getIterator()), DL);
  uint32_t SrcLocStrSize;
  Constant *SrcLocStr = OMPBuilder.getOrCreateSrcLocStr(Loc, SrcLocStrSize);
  Value *SrcLoc = OMPBuilder.getOrCreateIdent(SrcLocStr, SrcLocStrSize);

  // Create allocas for static init values.
  InsertPointTy AllocaIP(PreHeader, PreHeader->getFirstInsertionPt());
  Type *I32Type = Type::getInt32Ty(M.getContext());
  OMPBuilder.Builder.restoreIP(AllocaIP);
  Value *PLastIter =
      OMPBuilder.Builder.CreateAlloca(I32Type, nullptr, "omp.distribute.is_last");
  Value *PLowerBound = OMPBuilder.Builder.CreateAlloca(IVTy, nullptr, "omp.distribute.lb");
  Value *PStride = OMPBuilder.Builder.CreateAlloca(IVTy, nullptr, "omp.distribute.stride");
  Value *PUpperBound = OMPBuilder.Builder.CreateAlloca(IVTy, nullptr, "omp.distribute.ub");

  OpenMPIRBuilder::OutlineInfo OI;
  OI.EntryBB = PreHeader;
  OI.ExitBB = LoopExit;
  SmallPtrSet<BasicBlock *, 8> BlockSet;
  SmallVector<BasicBlock *, 8> BlockVector;
  OI.collectBlocks(BlockSet, BlockVector);

  // TODO: De-duplicate privatization code.
  auto Privatizer = [&]() {
    for (auto &It : DSAValueMap) {
      Value *Orig = It.first;
      DSAType DSA = It.second;
      Value *ReplacementValue = nullptr;
      Type *VTy = Orig->getType()->getPointerElementType();

      if (DSA == DSA_SHARED)
        continue;

      // Store previous uses to set them to the ReplacementValue after
      // privatization codegen.
      SetVector<Use *> Uses;
      for (Use &U : Orig->uses())
        if (auto *UserI = dyn_cast<Instruction>(U.getUser()))
          if (BlockSet.count(UserI->getParent()))
            Uses.insert(&U);

      OMPBuilder.Builder.restoreIP(AllocaIP);
      if (DSA == DSA_PRIVATE) {
        ReplacementValue = OMPBuilder.Builder.CreateAlloca(
            VTy, /*ArraySize */ nullptr, Orig->getName() + ".distribute.priv");
        OMPBuilder.Builder.CreateStore(Constant::getNullValue(VTy),
                                       ReplacementValue);
      } else if (DSA == DSA_FIRSTPRIVATE) {
        Value *V = OMPBuilder.Builder.CreateLoad(
            VTy, Orig, Orig->getName() + ".distribute.firstpriv.reload");
        ReplacementValue = OMPBuilder.Builder.CreateAlloca(
            VTy, /*ArraySize */ nullptr,
            Orig->getName() + ".distribute.firstpriv.copy");
        OMPBuilder.Builder.CreateStore(V, ReplacementValue);
        // ReplacementValue = Orig;
      } else
        assert(false && "Unsupported privatization");

      assert(ReplacementValue && "Expected non-null ReplacementValue");

      for (Use *UPtr : Uses)
        UPtr->set(ReplacementValue);
    }
  };

  OMPBuilder.Builder.SetInsertPoint(PreHeader->getTerminator());

  // Store the initial normalized upper bound to PUpperBound.
  Value *LoadUB = OMPBuilder.Builder.CreateLoad(IVTy, OMPLoopInfo.UB);
  OMPBuilder.Builder.CreateStore(LoadUB, PUpperBound);

  Constant *Zero = ConstantInt::get(IVTy, 0);
  Constant *One = ConstantInt::get(IVTy, 1);
  OMPBuilder.Builder.CreateStore(Zero, PLowerBound);
  OMPBuilder.Builder.CreateStore(One, PStride);

  // If Chunk is not specified (nullptr), default to one, complying with the
  // OpenMP specification.
  if (!OMPLoopInfo.Chunk)
    OMPLoopInfo.Chunk = One;
  Value *ChunkCast = OMPBuilder.Builder.CreateIntCast(OMPLoopInfo.Chunk, IVTy,
                                                      /*isSigned*/ false);

  Value *ThreadNum = OMPBuilder.getOrCreateThreadID(SrcLoc);

  // TODO: add more scheduling types.
  Constant *SchedulingType =
      ConstantInt::get(I32Type, static_cast<int>(OMPLoopInfo.DistSched));

  LLVM_DEBUG(dbgs() << "=== SchedulingType " << *SchedulingType << "\n");
  LLVM_DEBUG(dbgs() << "=== PLowerBound " << *PLowerBound << "\n");
  LLVM_DEBUG(dbgs() << "=== PUpperBound " << *PUpperBound << "\n");
  LLVM_DEBUG(dbgs() << "=== PStride " << *PStride << "\n");
  LLVM_DEBUG(dbgs() << "=== Incr " << *One << "\n");
  LLVM_DEBUG(dbgs() << "=== ChunkCast " << *ChunkCast << "\n");
  OMPBuilder.Builder.CreateCall(
      KmpcForStaticInit, {SrcLoc, ThreadNum, SchedulingType, PLastIter,
                          PLowerBound, PUpperBound, PStride, One, ChunkCast});
  // Load returned upper bound to UB.
  Value *LoadPUpperBound = OMPBuilder.Builder.CreateLoad(
      PUpperBound->getType()->getPointerElementType(), PUpperBound);
  OMPBuilder.Builder.CreateStore(LoadPUpperBound, OMPLoopInfo.UB);
  // Add lower bound to IV.
  Value *LowerBound = OMPBuilder.Builder.CreateLoad(IVTy, PLowerBound);
  //Value *LoadIV = OMPBuilder.Builder.CreateLoad(IVTy, OMPLoopInfo.IV);
  //Value *UpdateIV = OMPBuilder.Builder.CreateAdd(LoadIV, LowerBound);
  OMPBuilder.Builder.CreateStore(LowerBound, OMPLoopInfo.IV);

  // Add fini call after the loop exit block.
  BasicBlock *FiniBB = SplitBlock(LoopExit, &*LoopExit->getFirstInsertionPt());
  OMPBuilder.Builder.SetInsertPoint(FiniBB, FiniBB->getFirstInsertionPt());
  OMPBuilder.Builder.CreateCall(KmpcForStaticFini, {SrcLoc, ThreadNum});

  // Run the privatizer last to update values in the whole generated code.
  if (IsStandalone)
    Privatizer();
}

void CGIntrinsicsOpenMP::emitOMPDistributeParallelFor(
    MapVector<Value *, DSAType> &DSAValueMap, BasicBlock *StartBB,
    BasicBlock *ExitBB, OMPLoopInfoStruct &OMPLoopInfo,
    ParRegionInfoStruct &ParRegionInfo, bool IsStandalone) {

  LLVM_DEBUG(dbgs() << "OMPLoopInfo.IV " << *OMPLoopInfo.IV << "\n");
  LLVM_DEBUG(dbgs() << "OMPLoopInfo.Start " << *OMPLoopInfo.Start << "\n");
  LLVM_DEBUG(dbgs() << "OMPLoopInfo.LB " << *OMPLoopInfo.LB << "\n");
  LLVM_DEBUG(dbgs() << "OMPLoopInfo.UB " << *OMPLoopInfo.UB << "\n");
  assert(OMPLoopInfo.IV && "Expected non-null IV");
  assert(OMPLoopInfo.Start && "Expected non-null Start");
  assert(OMPLoopInfo.LB && "Expected non-null LB");
  assert(OMPLoopInfo.UB && "Expected non-null UB");

  BasicBlock *PreHeader = StartBB;
  BasicBlock *LoopHeader = PreHeader->getUniqueSuccessor();
  BasicBlock *LoopExit = ExitBB;
  assert(LoopHeader && "Expected unique successor from PreHeader to Header");
  LLVM_DEBUG(dbgs() << "=== PreHeader\n"
                    << *PreHeader << "=== End of PreHeader\n");
  LLVM_DEBUG(dbgs() << "=== LoopHeader\n"
                    << *LoopHeader << "=== End of LoopHeader\n");
  LLVM_DEBUG(dbgs() << "=== LoopExit \n"
                    << *LoopExit << "=== End of LoopExit\n");

  Type *IVTy = OMPLoopInfo.IV->getType()->getPointerElementType();
  Function *Fn = PreHeader->getParent();

  FunctionCallee KmpcForStaticInit = getKmpcForStaticInit(IVTy);
  FunctionCallee KmpcForStaticFini =
      OMPBuilder.getOrCreateRuntimeFunction(M, OMPRTL___kmpc_for_static_fini);

  const DebugLoc DL = PreHeader->getTerminator()->getDebugLoc();
  OpenMPIRBuilder::LocationDescription Loc(
      InsertPointTy(PreHeader, PreHeader->getTerminator()->getIterator()), DL);
  uint32_t SrcLocStrSize;
  Constant *SrcLocStr = OMPBuilder.getOrCreateSrcLocStr(Loc, SrcLocStrSize);
  Value *SrcLoc = OMPBuilder.getOrCreateIdent(SrcLocStr, SrcLocStrSize);

  // Create basic block structure.
  BasicBlock *ForEntry =
      SplitBlock(PreHeader, &*PreHeader->getFirstInsertionPt());
  ForEntry->setName("omp.inner.for.entry");
  BasicBlock *ForBegin =
      SplitBlock(ForEntry, &*ForEntry->getFirstInsertionPt());
  ForBegin->setName("omp.inner.for.begin");
  BasicBlock *CondTrue =
      SplitBlock(PreHeader, PreHeader->getTerminator());
  CondTrue->setName("omp.distribute.min.ub");
  BasicBlock *DistributePreheader =
      SplitBlock(CondTrue, CondTrue->getTerminator());
  DistributePreheader->setName("omp.distribute.preheader");
  BasicBlock *DistributeCond =
      SplitBlock(DistributePreheader, DistributePreheader->getTerminator());
  DistributeCond->setName("omp.distribute.cond");
  BasicBlock *ForEnd = splitBlockBefore(
      ExitBB, &*ExitBB->getFirstInsertionPt(), /*DomTreeUpdater*/ nullptr,
      /*LoopInfo*/ nullptr, /*MemorySSAUpdater*/ nullptr);
  ForEnd->setName("omp.inner.for.end");
  BasicBlock *ForExit = SplitBlock(ForEnd, ForEnd->getTerminator());
  ForExit->setName("omp.inner.for.exit");
  BasicBlock *DistributeInc =
      SplitBlock(ForExit, ForExit->getTerminator());
  DistributeInc->setName("omp.distribute.inc");
  BasicBlock *DistributeExit =
      SplitBlock(DistributeInc, DistributeInc->getTerminator());
  DistributeExit->setName("omp.distribute.exit");

  // Create allocas for distribute loop values.
  // TODO: Rethink AllocaIP, better if at Fn entry but breaks DSAValueMap if
  // the parent outlined function is not emitted first.
  // InsertPointTy AllocaIP(&Fn->getEntryBlock(),
  //                       Fn->getEntryBlock().getFirstInsertionPt());
  InsertPointTy AllocaIP(PreHeader, PreHeader->getFirstInsertionPt());
  Type *I32Type = Type::getInt32Ty(M.getContext());
  OMPBuilder.Builder.restoreIP(AllocaIP);
  Value *PDistributeIV =
      OMPBuilder.Builder.CreateAlloca(IVTy, nullptr, "omp.distribute.iv");
  Value *PLastIter =
      OMPBuilder.Builder.CreateAlloca(I32Type, nullptr, "omp.distribute.is_last");
  Value *PStride =
      OMPBuilder.Builder.CreateAlloca(IVTy, nullptr, "omp.distribute.stride");

  Value *PStart = nullptr;
  Value *PLowerBound = nullptr;
  Value *PUpperBound = nullptr;
  if (isOpenMPDeviceRuntime()) {
    // Create globalized allocation pointers for the start and distribute loop
    // bounds to be shareable with parallel threads.
    FunctionCallee KmpcAllocShared =
        OMPBuilder.getOrCreateRuntimeFunction(M, OMPRTL___kmpc_alloc_shared);
    Value *PStartAlloc = OMPBuilder.Builder.CreateCall(
        KmpcAllocShared,
        {ConstantInt::get(OMPBuilder.SizeTy,
                          M.getDataLayout().getTypeAllocSize(IVTy))});
    PStart = OMPBuilder.Builder.CreateBitCast(
        PStartAlloc, IVTy->getPointerTo(), "omp.distribute.start");
    Value *PLowerBoundAlloc = OMPBuilder.Builder.CreateCall(
        KmpcAllocShared,
        {ConstantInt::get(OMPBuilder.SizeTy,
                          M.getDataLayout().getTypeAllocSize(IVTy))});
    PLowerBound = OMPBuilder.Builder.CreateBitCast(
        PLowerBoundAlloc, IVTy->getPointerTo(), "omp.distribute.lb");
    // OMPBuilder.Builder.CreateAlloca(IVTy, nullptr, "omp.distribute.lb");
    Value *PUpperBoundAlloc = OMPBuilder.Builder.CreateCall(
        KmpcAllocShared,
        {ConstantInt::get(OMPBuilder.SizeTy,
                          M.getDataLayout().getTypeAllocSize(IVTy))});
    PUpperBound = OMPBuilder.Builder.CreateBitCast(
        PUpperBoundAlloc, IVTy->getPointerTo(), "omp.distribute.ub");
    // OMPBuilder.Builder.CreateAlloca(IVTy, nullptr, "omp.distribute.ub");
  } else {
    // Create globalized allocation pointers for the start and distribute loop
    // bounds to be shareable with parallel threads.
    PStart =
        OMPBuilder.Builder.CreateAlloca(IVTy, nullptr, "omp.distribute.start");
    PLowerBound =
        OMPBuilder.Builder.CreateAlloca(IVTy, nullptr, "omp.distribute.lb");
    PUpperBound =
        OMPBuilder.Builder.CreateAlloca(IVTy, nullptr, "omp.distribute.ub");
  }

  assert(PStart && "Expected non-null PStart");
  assert(PLowerBound && "Expected non-null PLowerBound");
  assert(PUpperBound && "Expected non-null PUpperBound");

  OMPBuilder.Builder.SetInsertPoint(PreHeader->getTerminator());
  // Store the start and initial normalized upper bound to PUpperBound.
  Value *LoadStart =
      OMPBuilder.Builder.CreateLoad(IVTy, OMPLoopInfo.Start);
  OMPBuilder.Builder.CreateStore(LoadStart, PStart);
  Value *LoadUB =
      OMPBuilder.Builder.CreateLoad(IVTy, OMPLoopInfo.UB);
  OMPBuilder.Builder.CreateStore(LoadUB, PUpperBound);

  Constant *Zero = ConstantInt::get(IVTy, 0);
  Constant *One = ConstantInt::get(IVTy, 1);
  OMPBuilder.Builder.CreateStore(Zero, PLowerBound);
  OMPBuilder.Builder.CreateStore(One, PStride);
  Constant *ZeroI32 = ConstantInt::get(I32Type, 0);
  OMPBuilder.Builder.CreateStore(ZeroI32, PLastIter);

  // If Chunk is not specified (nullptr), default to one, complying with the
  // OpenMP specification.
  if (!OMPLoopInfo.Chunk)
    OMPLoopInfo.Chunk = One;
  Value *ChunkCast =
      OMPBuilder.Builder.CreateIntCast(OMPLoopInfo.Chunk, IVTy, /*isSigned*/ false);

  Value *ThreadNum = OMPBuilder.getOrCreateThreadID(SrcLoc);

  // TODO: add more scheduling types.
  Constant *SchedulingType =
      ConstantInt::get(I32Type, static_cast<int>(OMPLoopInfo.DistSched));

  LLVM_DEBUG(dbgs() << "=== SchedulingType " << *SchedulingType << "\n");
  LLVM_DEBUG(dbgs() << "=== PLowerBound " << *PLowerBound << "\n");
  LLVM_DEBUG(dbgs() << "=== PUpperBound " << *PUpperBound << "\n");
  LLVM_DEBUG(dbgs() << "=== PStride " << *PStride << "\n");
  LLVM_DEBUG(dbgs() << "=== Incr " << *One << "\n");
  LLVM_DEBUG(dbgs() << "=== ChunkCast " << *ChunkCast << "\n");
  OMPBuilder.Builder.CreateCall(
      KmpcForStaticInit, {SrcLoc, ThreadNum, SchedulingType, PLastIter,
                          PLowerBound, PUpperBound, PStride, One, ChunkCast});

  LoadUB = OMPBuilder.Builder.CreateLoad(IVTy, PUpperBound);
  Value *LoadGlobalUB = OMPBuilder.Builder.CreateLoad(IVTy, OMPLoopInfo.UB);
  Value *Cond = OMPBuilder.Builder.CreateICmpSGT(LoadUB, LoadGlobalUB);
  OMPBuilder.Builder.CreateCondBr(Cond, CondTrue, DistributePreheader);
  PreHeader->getTerminator()->eraseFromParent();

  // Emit CondTrue and Set UB = min(UB, GlobalUB)
  OMPBuilder.Builder.SetInsertPoint(CondTrue->getTerminator());
  OMPBuilder.Builder.CreateStore(LoadGlobalUB, PUpperBound);

  // Add lower bound to the distribute loop IV.
  OMPBuilder.Builder.SetInsertPoint(DistributePreheader->getTerminator());
  Value *LoadLB = OMPBuilder.Builder.CreateLoad(IVTy, PLowerBound);
  OMPBuilder.Builder.CreateStore(LoadLB, PDistributeIV);

  // Emit Cond block for the distribute outer loop.
  OMPBuilder.Builder.SetInsertPoint(DistributeCond);
  DistributeCond->getTerminator()->eraseFromParent();
  Value *LoadIV = OMPBuilder.Builder.CreateLoad(IVTy, PDistributeIV);
  LoadUB = OMPBuilder.Builder.CreateLoad(IVTy, PUpperBound);
  Cond = OMPBuilder.Builder.CreateICmpSLE(LoadIV, LoadUB);
  OMPBuilder.Builder.CreateCondBr(Cond, ForEntry, DistributeExit);

  // Emit Inc block.
  OMPBuilder.Builder.SetInsertPoint(DistributeInc->getTerminator());
  LoadIV = OMPBuilder.Builder.CreateLoad(IVTy, PDistributeIV);
  Value *LoadStride = OMPBuilder.Builder.CreateLoad(IVTy, PStride);
  Value *UpdateIV = OMPBuilder.Builder.CreateAdd(LoadIV, LoadStride);
  OMPBuilder.Builder.CreateStore(UpdateIV, PDistributeIV);
  DistributeInc->getTerminator()->setSuccessor(0, DistributeCond);

  // Emit Exit block for the distribute outer loop.
  OMPBuilder.Builder.SetInsertPoint(DistributeExit->getTerminator());
  OMPBuilder.Builder.CreateCall(KmpcForStaticFini, {SrcLoc, ThreadNum});

  LLVM_DEBUG(dbgs() << "=== Dump Distribute DistributeParallelFor\n"
                    << *PreHeader->getParent()
                    << "=== End Distribute DistributeParallelFor\n");
  // TODO: Privatization runs by parallel for codegen.
  // Run the privatizer last to update values in the whole generated code.
  //if (IsStandalone)
  //  Privatizer();

  // Emit parallel for.

  // Replace uses of Start, LB, UB in the parallel for inner loop to the
  // globalized Start, distribute LB, UB.
  OpenMPIRBuilder::OutlineInfo OI;
  OI.EntryBB = ForBegin;
  OI.ExitBB = ForEnd;
  SmallPtrSet<BasicBlock *, 8> BlockSet;
  SmallVector<BasicBlock *, 8> BlockVector;
  OI.collectBlocks(BlockSet, BlockVector);

  auto ShouldReplace = [&BlockSet](Use &U) {
    if (auto *UserI = dyn_cast<Instruction>(U.getUser()))
      if (BlockSet.count(UserI->getParent()))
        return true;

    return false;
  };

  // Replace the inner, parallel for loop LB, UB.
  OMPLoopInfo.Start->replaceUsesWithIf(PStart, ShouldReplace);
  OMPLoopInfo.LB->replaceUsesWithIf(PLowerBound, ShouldReplace);
  OMPLoopInfo.UB->replaceUsesWithIf(PUpperBound, ShouldReplace);

  OMPLoopInfoStruct OMPInnerLoopInfo;
  OMPInnerLoopInfo.IV = OMPLoopInfo.IV;
  OMPInnerLoopInfo.Start = OMPLoopInfo.Start;
  OMPInnerLoopInfo.LB = PLowerBound;
  OMPInnerLoopInfo.UB = PUpperBound;
  // TODO: schedule, chunk.
  emitOMPFor(DSAValueMap, OMPInnerLoopInfo, ForBegin, ForEnd,
             /* IsStandalone */ false);

  LLVM_DEBUG(dbgs() << "=== Dump DistributeFor DistributeParallelFor\n"
                    << *PreHeader->getParent()
                    << "=== End DistributeFor DistributeParallelFor\n");

  BasicBlock *ParEntryBB = ForEntry;
  LLVM_DEBUG(dbgs() << "ParEntryBB " << ParEntryBB->getName() << "\n");
  BasicBlock *ParStartBB = ForBegin;
  LLVM_DEBUG(dbgs() << "ParStartBB " << ParStartBB->getName() << "\n");
  BasicBlock *ParEndBB = ForExit;
  LLVM_DEBUG(dbgs() << "ParEndBB " << ParEndBB->getName() << "\n");
  BasicBlock *ParAfterBB = DistributeInc;
  LLVM_DEBUG(dbgs() << "ParAfterBB " << ParAfterBB->getName() << "\n");

  // Prepare DSAValueMap for parallel
  DSAValueMap[PStart] = DSA_FIRSTPRIVATE;
  DSAValueMap[PLowerBound] = DSA_FIRSTPRIVATE;
  DSAValueMap[PUpperBound] = DSA_FIRSTPRIVATE;

  emitOMPParallel(
      DSAValueMap, nullptr, DL, Fn, ParEntryBB, ParStartBB, ParEndBB,
      ParAfterBB, [](auto) {}, ParRegionInfo);

  LLVM_DEBUG(dbgs() << "=== After Dump DistributeParallelFor\n"
                    << *PreHeader->getParent()
                    << "=== End DistributeParallelFor\n");

  // Delete further unused DSA entries.
  DSAValueMap.erase(PStart);
  DSAValueMap.erase(PLowerBound);
  DSAValueMap.erase(PUpperBound);

  if (verifyFunction(*Fn, &errs()))
    report_fatal_error("Verification of DistributeParallelFor lowering failed!");

  LLVM_DEBUG(dbgs() << "=== Dump DistributeParallelFor\n"
                    << *PreHeader->getParent()
                    << "=== End DistributeParallelFor\n");
}

void CGIntrinsicsOpenMP::emitOMPTargetTeamsDistributeParallelFor(
    MapVector<Value *, DSAType> &DSAValueMap, const DebugLoc &DL, Function *Fn,
    BasicBlock *EntryBB, BasicBlock *StartBB, BasicBlock *EndBB,
    BasicBlock *ExitBB, BasicBlock *AfterBB, OMPLoopInfoStruct &OMPLoopInfo,
    ParRegionInfoStruct &ParRegionInfo, TargetInfoStruct &TargetInfo,
    MapVector<Value *, SmallVector<FieldMappingInfo, 4>> &StructMappingInfoMap,
    bool IsDeviceTargetRegion) {
  emitOMPDistributeParallelFor(DSAValueMap, StartBB, ExitBB, OMPLoopInfo,
                               ParRegionInfo,
                               /* isStandalone */ false);
  // Lower target_teams.
  emitOMPTargetTeams(DSAValueMap, DL, Fn, EntryBB, StartBB, EndBB, AfterBB,
                     TargetInfo, StructMappingInfoMap, IsDeviceTargetRegion);

  // Alternative codegen, starting from top-down and renaming values using the
  // ValueToValueMap.
#if 0
  ValueToValueMapTy VMap;
  // Lower target_teams.
  if (IsDeviceTargetRegion) {
    emitOMPTeamsDevice(DSAValueMap, &VMap, DL, Fn, EntryBB, StartBB, EndBB, AfterBB);

    for(auto VV : VMap) {
      dbgs() << "VMap " << *VV.first << " => " << *VV.second << "\n";
    }
    emitOMPTargetDevice(Fn, DSAValueMap);
  } else {
    emitOMPTeams(DSAValueMap, &VMap, DL, Fn, EntryBB, StartBB, EndBB, AfterBB,
                 TargetInfo.NumTeams, TargetInfo.ThreadLimit);
    StartBB = SplitBlock(EntryBB, &*EntryBB->getFirstInsertionPt());
    EndBB = AfterBB;
    emitOMPTarget(TargetInfo.DevFuncName, TargetInfo.ELF, Fn, EntryBB, StartBB,
                  EndBB, DSAValueMap, StructMappingInfoMap, TargetInfo.NumTeams,
                  TargetInfo.ThreadLimit);
  }

  // Update DSAValueMap
  SmallVector<Value *, 8> ToDelete;
  for(auto &It : DSAValueMap) {
    Value *V = It.first;
    if(!VMap.count(V))
      continue;

    DSAValueMap[VMap[V]] = It.second;
    dbgs() << "Update DSAValueMap " << *VMap[V] << " ~> " << It.second << "\n";
    ToDelete.push_back(V);
  }
  for(auto *V : ToDelete) {
    dbgs() << "Update DSAValueMAp delete " << *V << "\n";
    DSAValueMap.erase(V);
  }

  // Update OMPLoopInfo
  OMPLoopInfo.IV = VMap[OMPLoopInfo.IV];
  OMPLoopInfo.Start = VMap[OMPLoopInfo.Start];
  OMPLoopInfo.LB = VMap[OMPLoopInfo.LB];
  OMPLoopInfo.UB = VMap[OMPLoopInfo.UB];

  emitOMPDistributeParallelFor(DSAValueMap, StartBB, ExitBB, OMPLoopInfo,
                               /* isStandalone */ false);
#endif
}

void CGIntrinsicsOpenMP::emitOMPTargetTeams(
    MapVector<Value *, DSAType> &DSAValueMap, const DebugLoc &DL, Function *Fn,
    BasicBlock *EntryBB, BasicBlock *StartBB, BasicBlock *EndBB,
    BasicBlock *AfterBB, TargetInfoStruct &TargetInfo,
    MapVector<Value *, SmallVector<FieldMappingInfo, 4>> &StructMappingInfoMap,
    bool IsDeviceTargetRegion) {

  BasicBlock *TeamsEntryBB = SplitBlock(EntryBB, EntryBB->getTerminator());
  TeamsEntryBB->setName("omp.teams.entry");
  BasicBlock *TeamsStartBB =
      splitBlockBefore(StartBB, &*StartBB->getFirstInsertionPt(), nullptr, nullptr,
                       nullptr, "omp.teams.start");
  BasicBlock *TeamsEndBB =
      splitBlockBefore(EndBB, &*EndBB->getFirstInsertionPt(), nullptr, nullptr,
                       nullptr, "omp.teams.end");
  // TargetInfo contains teams informations.
  TeamsInfoStruct TeamsInfo;
  TeamsInfo.NumTeams = TargetInfo.NumTeams;
  TeamsInfo.ThreadLimit = TargetInfo.ThreadLimit;
  emitOMPTeams(DSAValueMap, nullptr, DL, Fn, TeamsEntryBB, TeamsStartBB,
               TeamsEndBB, EndBB, TeamsInfo);

  emitOMPTarget(Fn, EntryBB, TeamsEntryBB, EndBB, DSAValueMap,
                StructMappingInfoMap, TargetInfo, IsDeviceTargetRegion);
}

bool CGIntrinsicsOpenMP::isOpenMPDeviceRuntime() {
  Triple TargetTriple(M.getTargetTriple());

  if (TargetTriple.isNVPTX())
    return true;

  return false;
}
