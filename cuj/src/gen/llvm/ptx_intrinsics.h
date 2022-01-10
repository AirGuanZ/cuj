#pragma once

#include <llvm/IR/IRBuilder.h>

#include <cuj/core/expr.h>

CUJ_NAMESPACE_BEGIN(cuj::gen)

llvm::Value *process_ptx_intrinsics(
    llvm::Module                    &top_module,
    llvm::IRBuilder<>               &ir_builder,
    core::Intrinsic                  intrinsic_type,
    const std::vector<llvm::Value*> &args,
    bool                             approx_math_func);

CUJ_NAMESPACE_END(cuj::gen)
