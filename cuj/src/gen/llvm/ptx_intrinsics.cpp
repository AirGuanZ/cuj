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

#include "libdevice_man.h"
#include "ptx_intrinsics.h"

CUJ_NAMESPACE_BEGIN(cuj::gen)

void link_with_libdevice(llvm::Module &dest_module)
{
    auto &context = dest_module.getContext();
    auto libdev_module = libdev::new_libdevice10_module(&context);

    std::vector<std::string> libdev_func_names;
    for(auto &f : *libdev_module)
    {
        if(!f.isDeclaration())
            libdev_func_names.push_back(f.getName().str());
    }

    libdev_module->setTargetTriple("nvptx64-nvidia-cuda");
    dest_module.setDataLayout(libdev_module->getDataLayout());

    if(llvm::Linker::linkModules(dest_module, std::move(libdev_module)))
        throw CujException("failed to link with libdevice");

    for(auto &name : libdev_func_names)
    {
        auto func = dest_module.getFunction(name);
        assert(func);
        func->setLinkage(llvm::GlobalValue::InternalLinkage);
    }
}

llvm::Value *process_ptx_intrinsics(
    llvm::Module                    &top_module,
    llvm::IRBuilder<>               &ir_builder,
    core::Intrinsic                  intrinsic_type,
    const std::vector<llvm::Value*> &args,
    bool                             approx_math_func)
{
    switch(intrinsic_type)
    {
    case core::Intrinsic::thread_idx_x:
        return ir_builder.CreateIntrinsic(
            llvm::Intrinsic::nvvm_read_ptx_sreg_tid_x, {}, args);
    case core::Intrinsic::thread_idx_y:
        return ir_builder.CreateIntrinsic(
            llvm::Intrinsic::nvvm_read_ptx_sreg_tid_y, {}, args);
    case core::Intrinsic::thread_idx_z:
        return ir_builder.CreateIntrinsic(
            llvm::Intrinsic::nvvm_read_ptx_sreg_tid_z, {}, args);
    case core::Intrinsic::block_idx_x:
        return ir_builder.CreateIntrinsic(
            llvm::Intrinsic::nvvm_read_ptx_sreg_ctaid_x, {}, args);
    case core::Intrinsic::block_idx_y:
        return ir_builder.CreateIntrinsic(
            llvm::Intrinsic::nvvm_read_ptx_sreg_ctaid_y, {}, args);
    case core::Intrinsic::block_idx_z:
        return ir_builder.CreateIntrinsic(
            llvm::Intrinsic::nvvm_read_ptx_sreg_ctaid_z, {}, args);
    case core::Intrinsic::block_dim_x:
        return ir_builder.CreateIntrinsic(
            llvm::Intrinsic::nvvm_read_ptx_sreg_ntid_x, {}, args);
    case core::Intrinsic::block_dim_y:
        return ir_builder.CreateIntrinsic(
            llvm::Intrinsic::nvvm_read_ptx_sreg_ntid_y, {}, args);
    case core::Intrinsic::block_dim_z:
        return ir_builder.CreateIntrinsic(
            llvm::Intrinsic::nvvm_read_ptx_sreg_ntid_z, {}, args);
    default:
        break;
    }

    if(approx_math_func)
    {
#define MAP_INTRINSIC(FROM, TO)                                                 \
        if(intrinsic_type == core::Intrinsic::FROM)                             \
        {                                                                       \
            return ir_builder.CreateIntrinsic(                                  \
                llvm::Intrinsic::TO, {}, args);                                 \
        }

        MAP_INTRINSIC(f32_sin,   nvvm_sin_approx_ftz_f)
        MAP_INTRINSIC(f32_cos,   nvvm_cos_approx_ftz_f)
        MAP_INTRINSIC(f32_sqrt,  nvvm_sqrt_approx_ftz_f)
        MAP_INTRINSIC(f32_rsqrt, nvvm_rsqrt_approx_ftz_f)
        MAP_INTRINSIC(f32_floor, nvvm_floor_ftz_f)
        MAP_INTRINSIC(f32_ceil,  nvvm_ceil_ftz_f)
        MAP_INTRINSIC(f32_round, nvvm_round_ftz_f)
        MAP_INTRINSIC(f32_trunc, nvvm_trunc_ftz_f)
            
        MAP_INTRINSIC(f64_rsqrt, nvvm_rsqrt_approx_d)
        MAP_INTRINSIC(f64_floor, nvvm_floor_d)
        MAP_INTRINSIC(f64_ceil,  nvvm_ceil_d)
        MAP_INTRINSIC(f64_round, nvvm_round_d)
        MAP_INTRINSIC(f64_trunc, nvvm_trunc_d)

#undef MAP_INTRINSIC

        if(intrinsic_type == core::Intrinsic::f32_min ||
           intrinsic_type == core::Intrinsic::f64_min)
        {
            auto comp = ir_builder.CreateFCmpOLT(args[0], args[1]);
            return ir_builder.CreateSelect(comp, args[0], args[1]);
        }

        if(intrinsic_type == core::Intrinsic::f32_max ||
           intrinsic_type == core::Intrinsic::f64_max)
        {
            auto comp = ir_builder.CreateFCmpOGT(args[0], args[1]);
            return ir_builder.CreateSelect(comp, args[0], args[1]);
        }
    }

    if(auto func_name = libdev::get_libdevice_function_name(intrinsic_type))
    {
        auto func = top_module.getFunction(func_name);
        assert(func);
        if(!func->hasFnAttribute(llvm::Attribute::ReadNone))
            func->addFnAttr(llvm::Attribute::ReadNone);
        return ir_builder.CreateCall(func, args);
    }

    throw CujException(
        std::string("unsupported intrinsic: ") +
        intrinsic_name(intrinsic_type));
}

CUJ_NAMESPACE_END(cuj::gen)

#ifdef _MSC_VER
#pragma warning(pop)
#endif
