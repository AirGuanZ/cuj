#pragma once

#include <iostream>

#include <llvm/Support/CodeGen.h>
#include <llvm/Support/raw_ostream.h>

#include <cuj/core/expr.h>
#include <cuj/gen/option.h>
#include <cuj/utils/unreachable.h>

CUJ_NAMESPACE_BEGIN(cuj::gen::llvm_helper)

template<typename T>
void print(const T &obj)
{
    std::string ret;
    llvm::raw_string_ostream ss(ret);
    ss << obj;
    ss.flush();
    std::cerr << ret << std::endl;
}

inline llvm::Type *builtin_to_llvm_type(llvm::LLVMContext *ctx, core::Builtin builtin) {
    switch (builtin) {
        case core::Builtin::S8:
        case core::Builtin::U8:
            return llvm::Type::getInt8Ty(*ctx);
        case core::Builtin::S16:
        case core::Builtin::U16:
            return llvm::Type::getInt16Ty(*ctx);
        case core::Builtin::S32:
        case core::Builtin::U32:
            return llvm::Type::getInt32Ty(*ctx);
        case core::Builtin::S64:
        case core::Builtin::U64:
            return llvm::Type::getInt64Ty(*ctx);
        case core::Builtin::F32:
            return llvm::Type::getFloatTy(*ctx);
        case core::Builtin::F64:
            return llvm::Type::getDoubleTy(*ctx);
        case core::Builtin::Char:
            return llvm::Type::getInt8Ty(*ctx);
        case core::Builtin::Bool:
            return llvm::Type::getInt1Ty(*ctx);
        case core::Builtin::Void:
            return llvm::Type::getVoidTy(*ctx);
    }
    unreachable();
}

template<typename T> requires std::is_arithmetic_v<T>
auto llvm_constant_num(llvm::LLVMContext &ctx, T v)
{
    using namespace llvm;
    if constexpr(std::is_same_v<T, int8_t>)
        return ConstantInt::get(Type::getInt8Ty(ctx), v, true);
    if constexpr(std::is_same_v<T, int16_t>)
        return ConstantInt::get(Type::getInt16Ty(ctx), v, true);
    if constexpr(std::is_same_v<T, int32_t>)
        return ConstantInt::get(Type::getInt32Ty(ctx), v, true);
    if constexpr(std::is_same_v<T, int64_t>)
        return ConstantInt::get(Type::getInt64Ty(ctx), v, true);
    if constexpr(std::is_same_v<T, uint8_t>)
        return ConstantInt::get(Type::getInt8Ty(ctx), v, false);
    if constexpr(std::is_same_v<T, uint16_t>)
        return ConstantInt::get(Type::getInt16Ty(ctx), v, false);
    if constexpr(std::is_same_v<T, uint32_t>)
        return ConstantInt::get(Type::getInt32Ty(ctx), v, false);
    if constexpr(std::is_same_v<T, uint64_t>)
        return ConstantInt::get(Type::getInt64Ty(ctx), v, false);
    if constexpr(std::is_same_v<T, float>)
        return ConstantFP::get(Type::getFloatTy(ctx), v);
    if constexpr(std::is_same_v<T, double>)
        return ConstantFP::get(Type::getDoubleTy(ctx), v);
    if constexpr(std::is_same_v<T, char>)
        return ConstantInt::get(Type::getInt8Ty(ctx), v, std::is_signed_v<char>);
    if constexpr(std::is_same_v<T, bool>)
        return ConstantInt::get(Type::getInt1Ty(ctx), v, false);
}

inline llvm::CodeGenOpt::Level get_codegen_opt_level(
    OptimizationLevel opt_level)
{
    switch(opt_level)
    {
    case OptimizationLevel::O0: return llvm::CodeGenOpt::None;
    case OptimizationLevel::O1: return llvm::CodeGenOpt::Less;
    case OptimizationLevel::O2: return llvm::CodeGenOpt::Default;
    case OptimizationLevel::O3: return llvm::CodeGenOpt::Aggressive;
    }
    unreachable();
}

CUJ_NAMESPACE_END(cuj::gen::llvm_helper)


