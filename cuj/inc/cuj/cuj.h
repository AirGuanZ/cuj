#pragma once

#include <cuj/ast/ast.h>

#if CUJ_ENABLE_CUDA

#include <cuj/builtin/cuda/cuda.h>

#endif // #if CUJ_ENABLE_CUDA

#if CUJ_ENABLE_LLVM

#include <cuj/gen/llvm.h>

#endif // #if CUJ_ENABLE_LLVM

#if CUJ_ENABLE_CUDA && CUJ_ENABLE_LLVM

#include <cuj/gen/ptx.h>

#endif // #if CUJ_ENABLE_CUDA && CUJ_ENABLE_LLVM

#include <cuj/gen/printer.h>

#include <cuj/ir/builder.h>
