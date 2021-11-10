#if CUJ_ENABLE_CUDA

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable: 4141)
#pragma warning(disable: 4244)
#pragma warning(disable: 4624)
#pragma warning(disable: 4626)
#pragma warning(disable: 4996)
#endif

#include <llvm/Bitcode/BitcodeReader.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Verifier.h>

#ifdef _MSC_VER
#pragma warning(pop)
#endif

#include "./libdevice_man.h"

CUJ_NAMESPACE_BEGIN(cuj::gen::libdev)

std::string get_libdevice_str();

std::unique_ptr<llvm::Module> new_libdevice10_module(llvm::LLVMContext *context)
{
    const std::string libdev_str = get_libdevice_str();

    auto parse_result = parseBitcodeFile(
        llvm::MemoryBufferRef(libdev_str, "libdevice"), *context);
    if(!parse_result)
    {
        auto err = parse_result.takeError();
        throw CUJException(toString(std::move(err)));
    }

    std::unique_ptr<llvm::Module> result;
    parse_result->swap(result);

    std::string err;
    llvm::raw_string_ostream err_ss(err);
    if(verifyModule(*result, &err_ss))
        throw CUJException(err);

    return result;
}

const char *get_libdevice_function_name(
    builtin::math::IntrinsicBasicMathFunctionType func, bool f32)
{
    using builtin::math::IntrinsicBasicMathFunctionType;

    if(f32)
    {
        switch(func)
        {
        case IntrinsicBasicMathFunctionType::abs:       return "__nv_fabsf";
        case IntrinsicBasicMathFunctionType::mod:       return "__nv_fmodf";
        case IntrinsicBasicMathFunctionType::remainder: return "__nv_remainderf";
        case IntrinsicBasicMathFunctionType::exp:       return "__nv_expf";
        case IntrinsicBasicMathFunctionType::exp2:      return "__nv_exp2f";
        case IntrinsicBasicMathFunctionType::exp10:     return "__nv_exp10f";
        case IntrinsicBasicMathFunctionType::log:       return "__nv_logf";
        case IntrinsicBasicMathFunctionType::log2:      return "__nv_log2f";
        case IntrinsicBasicMathFunctionType::log10:     return "__nv_log10f";
        case IntrinsicBasicMathFunctionType::pow:       return "__nv_powf";
        case IntrinsicBasicMathFunctionType::sqrt:      return "__nv_sqrtf";
        case IntrinsicBasicMathFunctionType::rsqrt:     return "__nv_rsqrtf";
        case IntrinsicBasicMathFunctionType::sin:       return "__nv_sinf";
        case IntrinsicBasicMathFunctionType::cos:       return "__nv_cosf";
        case IntrinsicBasicMathFunctionType::tan:       return "__nv_tanf";
        case IntrinsicBasicMathFunctionType::asin:      return "__nv_asinf";
        case IntrinsicBasicMathFunctionType::acos:      return "__nv_acosf";
        case IntrinsicBasicMathFunctionType::atan:      return "__nv_atanf";
        case IntrinsicBasicMathFunctionType::atan2:     return "__nv_atan2f";
        case IntrinsicBasicMathFunctionType::ceil:      return "__nv_ceilf";
        case IntrinsicBasicMathFunctionType::floor:     return "__nv_floorf";
        case IntrinsicBasicMathFunctionType::round:     return "__nv_roundf";
        case IntrinsicBasicMathFunctionType::trunc:     return "__nv_truncf";
        case IntrinsicBasicMathFunctionType::isfinite:  return "__nv_finitef";
        case IntrinsicBasicMathFunctionType::isinf:     return "__nv_isinff";
        case IntrinsicBasicMathFunctionType::isnan:     return "__nv_isnanf";
        }
    }
    else
    {
        switch(func)
        {
        case IntrinsicBasicMathFunctionType::abs:       return "__nv_fabs";
        case IntrinsicBasicMathFunctionType::mod:       return "__nv_fmod";
        case IntrinsicBasicMathFunctionType::remainder: return "__nv_remainder";
        case IntrinsicBasicMathFunctionType::exp:       return "__nv_exp";
        case IntrinsicBasicMathFunctionType::exp2:      return "__nv_exp2";
        case IntrinsicBasicMathFunctionType::exp10:     return "__nv_exp10";
        case IntrinsicBasicMathFunctionType::log:       return "__nv_log";
        case IntrinsicBasicMathFunctionType::log2:      return "__nv_log2";
        case IntrinsicBasicMathFunctionType::log10:     return "__nv_log10";
        case IntrinsicBasicMathFunctionType::pow:       return "__nv_pow";
        case IntrinsicBasicMathFunctionType::sqrt:      return "__nv_sqrt";
        case IntrinsicBasicMathFunctionType::rsqrt:     return "__nv_rsqrt";
        case IntrinsicBasicMathFunctionType::sin:       return "__nv_sin";
        case IntrinsicBasicMathFunctionType::cos:       return "__nv_cos";
        case IntrinsicBasicMathFunctionType::tan:       return "__nv_tan";
        case IntrinsicBasicMathFunctionType::asin:      return "__nv_asin";
        case IntrinsicBasicMathFunctionType::acos:      return "__nv_acos";
        case IntrinsicBasicMathFunctionType::atan:      return "__nv_atan";
        case IntrinsicBasicMathFunctionType::atan2:     return "__nv_atan2";
        case IntrinsicBasicMathFunctionType::ceil:      return "__nv_ceil";
        case IntrinsicBasicMathFunctionType::floor:     return "__nv_floor";
        case IntrinsicBasicMathFunctionType::round:     return "__nv_round";
        case IntrinsicBasicMathFunctionType::trunc:     return "__nv_trunc";
        case IntrinsicBasicMathFunctionType::isfinite:  return "__nv_isfinited";
        case IntrinsicBasicMathFunctionType::isinf:     return "__nv_isinfd";
        case IntrinsicBasicMathFunctionType::isnan:     return "__nv_isnand";
        }
    }
    unreachable();
}

CUJ_NAMESPACE_END(cuj::gen::libdev)

#endif // #if CUJ_ENABLE_CUDA
