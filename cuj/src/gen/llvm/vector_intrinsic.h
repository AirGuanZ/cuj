#pragma once

#include <llvm/IR/IRBuilder.h>

#include <cuj/common.h>

CUJ_NAMESPACE_BEGIN(cuj::gen::detail)

inline llvm::Value *create_vector_store(
    llvm::IRBuilder<> &ir, const std::vector<llvm::Value *> &args)
{
    assert(args.size() >= 2);
    auto addr = args[0];

    auto elem_type = args[1]->getType();
    const int arr_size = static_cast<int>(args.size()) - 1;
    auto vec_type = llvm::FixedVectorType::get(elem_type, arr_size);

    llvm::Value *v = llvm::UndefValue::get(vec_type);
    for(int i = 0; i < arr_size; ++i)
        v = ir.CreateInsertElement(v, args[i + 1], i);

    return ir.CreateStore(
        v, ir.CreatePointerCast(
            addr, llvm::PointerType::get(
                vec_type, addr->getType()->getPointerAddressSpace())));
}

inline void create_vector_load(
    llvm::IRBuilder<> &ir, const std::vector<llvm::Value *> &args)
{
    assert(args.size() >= 2);
    assert(args[1]->getType()->isPointerTy());
    auto addr = args[0];

    auto elem_type = llvm::dyn_cast<llvm::PointerType>(args[1]->getType())
        ->getPointerElementType();
    const int arr_size = static_cast<int>(args.size()) - 1;
    auto vec_type = llvm::FixedVectorType::get(elem_type, arr_size);

    llvm::Value *vec = ir.CreateLoad(ir.CreatePointerCast(
        addr, llvm::PointerType::get(
            vec_type, addr->getType()->getPointerAddressSpace())));

    for(int i = 0; i < arr_size; ++i)
    {
        auto elem = ir.CreateExtractElement(vec, i);
        ir.CreateStore(elem, args[i + 1]);
    }
}

CUJ_NAMESPACE_END(cuj::gen::detail)
