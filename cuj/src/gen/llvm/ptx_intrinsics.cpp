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

#include "helper.h"
#include "libdevice_man.h"
#include "ptx_intrinsics.h"

CUJ_NAMESPACE_BEGIN(cuj::gen)

namespace
{

    llvm::Function *get_print_function(llvm::Module &top_module)
    {
        const char *name = "vprintf";
        auto func = top_module.getFunction(name);
        if(func)
            return func;

        auto &context = top_module.getContext();
        std::array<llvm::Type *, 2> arg_types = {
            llvm::PointerType::get(llvm::IntegerType::getInt8Ty(context), 0),
            llvm::PointerType::get(llvm::IntegerType::getInt64Ty(context), 0)
        };
        auto ret_type = llvm::IntegerType::getInt32Ty(context);
        auto func_type = llvm::FunctionType::get(ret_type, arg_types, false);

        func = llvm::Function::Create(
            func_type, llvm::GlobalValue::ExternalLinkage, name, &top_module);
        func->deleteBody();
        return func;
    }

    llvm::Function *get_assertfail_function(llvm::Module &top_module)
    {
        const char *name = "__assertfail";
        auto func = top_module.getFunction(name);
        if(func)
            return func;

        auto &context = top_module.getContext();
        std::array<llvm::Type *, 5> arg_types = {
            llvm::PointerType::get(llvm::IntegerType::getInt8Ty(context), 0),
            llvm::PointerType::get(llvm::IntegerType::getInt8Ty(context), 0),
            llvm::IntegerType::getInt32Ty(context),
            llvm::PointerType::get(llvm::IntegerType::getInt8Ty(context), 0),
            llvm::IntegerType::getInt64Ty(context)
        };
        auto ret_type = llvm::Type::getVoidTy(context);
        auto func_type = llvm::FunctionType::get(ret_type, arg_types, false);

        func = llvm::Function::Create(
            func_type, llvm::GlobalValue::ExternalLinkage, name, top_module);
        func->deleteBody();
        return func;
    }

} // namespace anonymous

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

        MAP_INTRINSIC(f32_sin,      nvvm_sin_approx_ftz_f)
        MAP_INTRINSIC(f32_cos,      nvvm_cos_approx_ftz_f)
        MAP_INTRINSIC(f32_sqrt,     nvvm_sqrt_approx_ftz_f)
        MAP_INTRINSIC(f32_rsqrt,    nvvm_rsqrt_approx_ftz_f)
        MAP_INTRINSIC(f32_floor,    nvvm_floor_ftz_f)
        MAP_INTRINSIC(f32_ceil,     nvvm_ceil_ftz_f)
        MAP_INTRINSIC(f32_round,    nvvm_round_ftz_f)
        MAP_INTRINSIC(f32_trunc,    nvvm_trunc_ftz_f)
        MAP_INTRINSIC(f32_saturate, nvvm_saturate_ftz_f)

        MAP_INTRINSIC(f64_rsqrt, nvvm_rsqrt_approx_d)
    }
    else
    {
        MAP_INTRINSIC(f32_floor,    nvvm_floor_f)
        MAP_INTRINSIC(f32_ceil,     nvvm_ceil_f)
        MAP_INTRINSIC(f32_round,    nvvm_round_f)
        MAP_INTRINSIC(f32_trunc,    nvvm_trunc_f)
        MAP_INTRINSIC(f32_saturate, nvvm_saturate_f)
    }

    MAP_INTRINSIC(f64_floor,    nvvm_floor_d)
    MAP_INTRINSIC(f64_ceil,     nvvm_ceil_d)
    MAP_INTRINSIC(f64_round,    nvvm_round_d)
    MAP_INTRINSIC(f64_trunc,    nvvm_trunc_d)
    MAP_INTRINSIC(f64_saturate, nvvm_saturate_d)

#undef MAP_INTRINSIC

    if(auto func_name = libdev::get_libdevice_function_name(intrinsic_type))
    {
        auto func = top_module.getFunction(func_name);
        assert(func);
        if(!func->hasFnAttribute(llvm::Attribute::ReadNone))
            func->addFnAttr(llvm::Attribute::ReadNone);
        return ir_builder.CreateCall(func, args);
    }

    if(intrinsic_type == core::Intrinsic::print)
    {
        auto func = get_print_function(top_module);

        std::vector<llvm::Type *> mem_types;
        for(size_t i = 1; i < args.size(); ++i)
            mem_types.push_back(args[i]->getType());
        auto valist_type =
            llvm::StructType::get(top_module.getContext(), mem_types);

        llvm::Value *valist = llvm::UndefValue::get(valist_type);
        for(size_t i = 1; i < args.size(); ++i)
        {
            const auto iu = static_cast<unsigned>(i - 1);
            valist = ir_builder.CreateInsertValue(valist, args[i], iu);
        }

        auto valist_alloc = ir_builder.CreateAlloca(valist_type, 0, nullptr);
        ir_builder.CreateStore(valist, valist_alloc);

        auto valist_ptr = ir_builder.CreatePointerCast(
            valist_alloc, llvm::PointerType::get(
                llvm::IntegerType::getInt64Ty(top_module.getContext()), 0));

        return ir_builder.CreateCall(func, { args[0], valist_ptr });
    }

    if(intrinsic_type == core::Intrinsic::assert_fail)
    {
        auto func = get_assertfail_function(top_module);
        auto one = llvm_helper::llvm_constant_num(
            top_module.getContext(), size_t(1));
        return ir_builder.CreateCall(
            func, { args[0], args[1], args[2], args[3], one });
    }

    throw CujException(
        std::string("unsupported intrinsic: ") +
        intrinsic_name(intrinsic_type));
}

CUJ_NAMESPACE_END(cuj::gen)

#ifdef _MSC_VER
#pragma warning(pop)
#endif
