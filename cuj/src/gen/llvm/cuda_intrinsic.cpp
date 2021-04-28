#if CUJ_ENABLE_CUDA

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable: 4141)
#pragma warning(disable: 4244)
#pragma warning(disable: 4624)
#pragma warning(disable: 4626)
#pragma warning(disable: 4996)
#endif

#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/IntrinsicsNVPTX.h>
#include <llvm/Linker/Linker.h>

#ifdef _MSC_VER
#pragma warning(pop)
#endif

#include <cuj/gen/llvm.h>

#include "./libdevice_man.h"

CUJ_NAMESPACE_BEGIN(cuj::gen)

void link_with_libdevice(
    llvm::LLVMContext *context,
    llvm::Module      *dest_module)
{
    auto libdev_module = libdev::new_libdevice10_module(context);

    std::vector<std::string> libdev_func_names;
    for(auto &f : *libdev_module)
    {
        if(!f.isDeclaration())
            libdev_func_names.push_back(f.getName().str());
    }
    
    libdev_module->setTargetTriple("nvptx64-nvidia-cuda");
    dest_module->setDataLayout(libdev_module->getDataLayout());
    
    if(llvm::Linker::linkModules(*dest_module, std::move(libdev_module)))
        throw CUJException("failed to link with libdevice");

    for(auto &name : libdev_func_names)
    {
        auto func = dest_module->getFunction(name);
        CUJ_ASSERT(func);
        func->setLinkage(llvm::GlobalValue::InternalLinkage);
    }
}

llvm::Value *process_cuda_intrinsic(
    llvm::Module                     *top_module,
    llvm::IRBuilder<>                &ir,
    const std::string                &name,
    const std::vector<llvm::Value *> &args)
{
#define CUJ_CUDA_INTRINSIC_SREG(NAME, ID)                                       \
    do {                                                                        \
        if(name == NAME)                                                        \
        {                                                                       \
            CUJ_ASSERT(args.empty());                                           \
            return ir.CreateIntrinsic(                                          \
                llvm::Intrinsic::nvvm_read_ptx_sreg_##ID, {}, {});              \
        }                                                                       \
    } while(false)

    CUJ_CUDA_INTRINSIC_SREG("cuda.thread_index_x", tid_x);
    CUJ_CUDA_INTRINSIC_SREG("cuda.thread_index_y", tid_y);
    CUJ_CUDA_INTRINSIC_SREG("cuda.thread_index_z", tid_z);
    CUJ_CUDA_INTRINSIC_SREG("cuda.block_index_x", ctaid_x);
    CUJ_CUDA_INTRINSIC_SREG("cuda.block_index_y", ctaid_y);
    CUJ_CUDA_INTRINSIC_SREG("cuda.block_index_z", ctaid_z);
    CUJ_CUDA_INTRINSIC_SREG("cuda.block_dim_x", ntid_x);
    CUJ_CUDA_INTRINSIC_SREG("cuda.block_dim_y", ntid_y);
    CUJ_CUDA_INTRINSIC_SREG("cuda.block_dim_z", ntid_z);

#undef CUJ_CUDA_INTRINSIC_SREG

#define CUJ_CALL_LIBDEVICE(TYPE, IS_F32)                                        \
    do {                                                                        \
        if(name == (IS_F32 ? ("math." #TYPE ".f32") : ("math." #TYPE ".f64")))  \
        {                                                                       \
            auto func_name = libdev::get_libdevice_function_name(               \
                builtin::math::IntrinsicBasicMathFunctionType::TYPE, IS_F32);   \
            auto func = top_module->getFunction(func_name);                     \
            CUJ_ASSERT(func);                                                   \
            return ir.CreateCall(func, args);                                   \
        }                                                                       \
    } while(false)

    CUJ_CALL_LIBDEVICE(abs,       true);
    CUJ_CALL_LIBDEVICE(mod,       true);
    CUJ_CALL_LIBDEVICE(remainder, true);
    CUJ_CALL_LIBDEVICE(exp,       true);
    CUJ_CALL_LIBDEVICE(exp2,      true);
    CUJ_CALL_LIBDEVICE(log,       true);
    CUJ_CALL_LIBDEVICE(log2,      true);
    CUJ_CALL_LIBDEVICE(log10,     true);
    CUJ_CALL_LIBDEVICE(pow,       true);
    CUJ_CALL_LIBDEVICE(sqrt,      true);
    CUJ_CALL_LIBDEVICE(sin,       true);
    CUJ_CALL_LIBDEVICE(cos,       true);
    CUJ_CALL_LIBDEVICE(tan,       true);
    CUJ_CALL_LIBDEVICE(asin,      true);
    CUJ_CALL_LIBDEVICE(acos,      true);
    CUJ_CALL_LIBDEVICE(atan,      true);
    CUJ_CALL_LIBDEVICE(atan2,     true);
    CUJ_CALL_LIBDEVICE(ceil,      true);
    CUJ_CALL_LIBDEVICE(floor,     true);
    CUJ_CALL_LIBDEVICE(trunc,     true);
    CUJ_CALL_LIBDEVICE(round,     true);
    CUJ_CALL_LIBDEVICE(isfinite,  true);
    CUJ_CALL_LIBDEVICE(isinf,     true);
    CUJ_CALL_LIBDEVICE(isnan,     true);
    
    CUJ_CALL_LIBDEVICE(abs,       false);
    CUJ_CALL_LIBDEVICE(mod,       false);
    CUJ_CALL_LIBDEVICE(remainder, false);
    CUJ_CALL_LIBDEVICE(exp,       false);
    CUJ_CALL_LIBDEVICE(exp2,      false);
    CUJ_CALL_LIBDEVICE(log,       false);
    CUJ_CALL_LIBDEVICE(log2,      false);
    CUJ_CALL_LIBDEVICE(log10,     false);
    CUJ_CALL_LIBDEVICE(pow,       false);
    CUJ_CALL_LIBDEVICE(sqrt,      false);
    CUJ_CALL_LIBDEVICE(sin,       false);
    CUJ_CALL_LIBDEVICE(cos,       false);
    CUJ_CALL_LIBDEVICE(tan,       false);
    CUJ_CALL_LIBDEVICE(asin,      false);
    CUJ_CALL_LIBDEVICE(acos,      false);
    CUJ_CALL_LIBDEVICE(atan,      false);
    CUJ_CALL_LIBDEVICE(atan2,     false);
    CUJ_CALL_LIBDEVICE(ceil,      false);
    CUJ_CALL_LIBDEVICE(floor,     false);
    CUJ_CALL_LIBDEVICE(trunc,     false);
    CUJ_CALL_LIBDEVICE(round,     false);
    CUJ_CALL_LIBDEVICE(isfinite,  false);
    CUJ_CALL_LIBDEVICE(isinf,     false);
    CUJ_CALL_LIBDEVICE(isnan,     false);

    return nullptr;
}

CUJ_NAMESPACE_END(cuj::gen)

#endif // #if CUJ_ENABLE_CUDA
