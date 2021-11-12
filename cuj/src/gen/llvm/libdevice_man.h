#pragma once

#if CUJ_ENABLE_CUDA

#include <cuj/builtin/math/basic.h>
#include <cuj/gen/llvm.h>

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable: 4141)
#pragma warning(disable: 4244)
#pragma warning(disable: 4624)
#pragma warning(disable: 4626)
#pragma warning(disable: 4996)
#endif

#include <llvm/IR/Module.h>

#ifdef _MSC_VER
#pragma warning(pop)
#endif

CUJ_NAMESPACE_BEGIN(cuj::gen::libdev)

enum IntrinsicParamType
{
    S32,
    S64,
    F32,
    F64
};

std::unique_ptr<llvm::Module> new_libdevice10_module(llvm::LLVMContext *context);

const char *get_libdevice_function_name(
    builtin::math::IntrinsicBasicMathFunctionType func, IntrinsicParamType type);

CUJ_NAMESPACE_END(cuj::gen::libdev)

#endif // #if CUJ_ENABLE_CUDA
