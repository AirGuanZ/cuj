#pragma once

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

std::unique_ptr<llvm::Module> new_libdevice10_module(llvm::LLVMContext *context);

const char *get_libdevice_function_name(core::Intrinsic);

CUJ_NAMESPACE_END(cuj::gen::libdev)
