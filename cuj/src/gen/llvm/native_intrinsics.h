#pragma once

#include <llvm/IR/IRBuilder.h>

#include <cuj/core/expr.h>

CUJ_NAMESPACE_BEGIN(cuj::gen)

llvm::Value *process_native_intrinsics(
    llvm::Module                    &top_module,
    llvm::IRBuilder<>               &ir_builder,
    core::Intrinsic                  intrinsic_type,
    const std::vector<llvm::Value*> &args);

CUJ_NAMESPACE_END(cuj::gen)
