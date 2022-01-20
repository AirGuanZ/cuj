#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable: 4141)
#pragma warning(disable: 4244)
#pragma warning(disable: 4624)
#pragma warning(disable: 4626)
#pragma warning(disable: 4996)
#endif

#include <llvm/IR/IRBuilder.h>

#include "native_intrinsics.h"

CUJ_NAMESPACE_BEGIN(cuj::gen)

namespace
{

    template<typename...LLVMArgTypes>
    llvm::Function *add_readnone_function(
        llvm::Module      &top_module,
        const std::string &symbol_name,
        llvm::Type        *ret_type,
        LLVMArgTypes   *...arg_types)
    {
        auto func_type = llvm::FunctionType::get(ret_type, { arg_types... }, false);
        auto func = llvm::Function::Create(
            func_type, llvm::GlobalValue::ExternalLinkage,
            symbol_name, &top_module);
        func->addFnAttr(llvm::Attribute::ReadNone);
        return func;
    }

    llvm::Function *get_intrinsics_function(
        llvm::Module      &top_module,
        core::Intrinsic    intrinsic_type)
    {
        const std::string name = intrinsic_name(intrinsic_type);
        if(auto func = top_module.getFunction(name))
            return func;

        auto &context = top_module.getContext();
        auto i32 = llvm::Type::getInt32Ty(context);
        auto f32 = llvm::Type::getFloatTy(context);
        auto f64 = llvm::Type::getDoubleTy(context);

#define REGISTER_INTRINSIC(NAME, ...)                                           \
        case core::Intrinsic::NAME:                                             \
            return add_readnone_function(top_module, name, __VA_ARGS__)

        switch(intrinsic_type)
        {
        case core::Intrinsic::None:
            throw CujException("intrinsic 'None' is invalid");

        REGISTER_INTRINSIC(f32_mod,      f32, f32, f32);
        REGISTER_INTRINSIC(f32_rem,      f32, f32, f32);
        REGISTER_INTRINSIC(f32_exp10,    f32, f32);
        REGISTER_INTRINSIC(f32_rsqrt,    f32, f32);
        REGISTER_INTRINSIC(f32_tan,      f32, f32);
        REGISTER_INTRINSIC(f32_asin,     f32, f32);
        REGISTER_INTRINSIC(f32_acos,     f32, f32);
        REGISTER_INTRINSIC(f32_atan,     f32, f32);
        REGISTER_INTRINSIC(f32_atan2,    f32, f32, f32);
        REGISTER_INTRINSIC(f32_isfinite, i32, f32);
        REGISTER_INTRINSIC(f32_isinf,    i32, f32);
        REGISTER_INTRINSIC(f32_isnan,    i32, f32);
        
        REGISTER_INTRINSIC(f64_mod,      f64, f64, f64);
        REGISTER_INTRINSIC(f64_rem,      f64, f64, f64);
        REGISTER_INTRINSIC(f64_exp10,    f64, f64);
        REGISTER_INTRINSIC(f64_rsqrt,    f64, f64);
        REGISTER_INTRINSIC(f64_tan,      f64, f64);
        REGISTER_INTRINSIC(f64_asin,     f64, f64);
        REGISTER_INTRINSIC(f64_acos,     f64, f64);
        REGISTER_INTRINSIC(f64_atan,     f64, f64);
        REGISTER_INTRINSIC(f64_atan2,    f64, f64, f64);
        REGISTER_INTRINSIC(f64_isfinite, i32, f64);
        REGISTER_INTRINSIC(f64_isinf,    i32, f64);
        REGISTER_INTRINSIC(f64_isnan,    i32, f64);

#undef REGISTER_INTRINSIC

        default:
            break;
        }

        throw CujException(
            std::string("unsupported intrinsic type: ") +
            intrinsic_name(intrinsic_type));
    }

} // namespace anonymous

llvm::Value *process_native_intrinsics(
    llvm::Module                    &top_module,
    llvm::IRBuilder<>               &ir_builder,
    core::Intrinsic                  intrinsic_type,
    const std::vector<llvm::Value*> &args)
{
    auto &context = top_module.getContext();
    auto f32 = llvm::Type::getFloatTy(context);
    auto f64 = llvm::Type::getDoubleTy(context);

    switch(intrinsic_type)
    {
#define CASE_LLVM_INTRINSICS(TYPE, LLVM_TYPE, TYPE_ARG)                         \
        case core::Intrinsic::TYPE_ARG##_##TYPE:                                \
            return ir_builder.CreateIntrinsic(                                  \
                llvm::Intrinsic::LLVM_TYPE, { TYPE_ARG }, args)                 \

    CASE_LLVM_INTRINSICS(abs,   fabs,    f32);
    CASE_LLVM_INTRINSICS(exp,   exp,     f32);
    CASE_LLVM_INTRINSICS(exp2,  exp2,    f32);
    CASE_LLVM_INTRINSICS(log,   log,     f32);
    CASE_LLVM_INTRINSICS(log2,  log2,    f32);
    CASE_LLVM_INTRINSICS(log10, log10,   f32);
    CASE_LLVM_INTRINSICS(pow,   pow,     f32);
    CASE_LLVM_INTRINSICS(sqrt,  sqrt,    f32);
    CASE_LLVM_INTRINSICS(sin,   sin,     f32);
    CASE_LLVM_INTRINSICS(cos,   cos,     f32);
    CASE_LLVM_INTRINSICS(ceil,  ceil,    f32);
    CASE_LLVM_INTRINSICS(floor, floor,   f32);
    CASE_LLVM_INTRINSICS(trunc, trunc,   f32);
    CASE_LLVM_INTRINSICS(round, round,   f32);
    CASE_LLVM_INTRINSICS(min,   minimum, f32);
    CASE_LLVM_INTRINSICS(max,   maximum, f32);

    CASE_LLVM_INTRINSICS(abs,   fabs,    f64);
    CASE_LLVM_INTRINSICS(exp,   exp,     f64);
    CASE_LLVM_INTRINSICS(exp2,  exp2,    f64);
    CASE_LLVM_INTRINSICS(log,   log,     f64);
    CASE_LLVM_INTRINSICS(log2,  log2,    f64);
    CASE_LLVM_INTRINSICS(log10, log10,   f64);
    CASE_LLVM_INTRINSICS(pow,   pow,     f64);
    CASE_LLVM_INTRINSICS(sqrt,  sqrt,    f64);
    CASE_LLVM_INTRINSICS(sin,   sin,     f64);
    CASE_LLVM_INTRINSICS(cos,   cos,     f64);
    CASE_LLVM_INTRINSICS(ceil,  ceil,    f64);
    CASE_LLVM_INTRINSICS(floor, floor,   f64);
    CASE_LLVM_INTRINSICS(trunc, trunc,   f64);
    CASE_LLVM_INTRINSICS(round, round,   f64);
    CASE_LLVM_INTRINSICS(min,   minimum, f64);
    CASE_LLVM_INTRINSICS(max,   maximum, f64);

#undef CASE_LLVM_INTRINSICS

    default:
        break;
    }

    if(intrinsic_type == core::Intrinsic::f32_saturate ||
       intrinsic_type == core::Intrinsic::f64_saturate)
    {
        auto x = args[0];
        auto minv = llvm::ConstantFP::get(x->getType(), 0.0f);
        auto maxv = llvm::ConstantFP::get(x->getType(), 1.0f);
        auto cmp_left = ir_builder.CreateFCmpOLT(minv, x);
        auto val_left = ir_builder.CreateSelect(cmp_left, x, minv);
        auto cmp_right = ir_builder.CreateFCmpOGT(val_left, maxv);
        return ir_builder.CreateSelect(cmp_right, maxv, val_left);
    }

    auto func = get_intrinsics_function(top_module, intrinsic_type);
    return ir_builder.CreateCall(func, args);
}

CUJ_NAMESPACE_END(cuj::gen)

#ifdef _MSC_VER
#pragma warning(pop)
#endif
