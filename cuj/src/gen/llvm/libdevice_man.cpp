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
        throw CujException(toString(std::move(err)));
    }

    std::unique_ptr<llvm::Module> result;
    parse_result->swap(result);

    std::string err;
    llvm::raw_string_ostream err_ss(err);
    if(verifyModule(*result, &err_ss))
        throw CujException(err);

    return result;
}

const char *get_libdevice_function_name(core::Intrinsic intrinsic)
{
    using enum core::Intrinsic;
    switch(intrinsic)
    {
    case f32_abs:      return "__nv_fabsf";
    case f32_mod:      return "__nv_fmodf";
    case f32_rem:      return "__nv_remainderf";
    case f32_exp:      return "__nv_expf";
    case f32_exp2:     return "__nv_exp2f";
    case f32_exp10:    return "__nv_exp10f";
    case f32_log:      return "__nv_logf";
    case f32_log2:     return "__nv_log2f";
    case f32_log10:    return "__nv_log10f";
    case f32_pow:      return "__nv_powf";
    case f32_sqrt:     return "__nv_sqrtf";
    case f32_rsqrt:    return "__nv_rsqrtf";
    case f32_sin:      return "__nv_sinf";
    case f32_cos:      return "__nv_cosf";
    case f32_tan:      return "__nv_tanf";
    case f32_asin:     return "__nv_asinf";
    case f32_acos:     return "__nv_acosf";
    case f32_atan:     return "__nv_atanf";
    case f32_atan2:    return "__nv_atan2f";
    case f32_ceil:     return "__nv_ceilf";
    case f32_floor:    return "__nv_floorf";
    case f32_trunc:    return "__nv_truncf";
    case f32_round:    return "__nv_roundf";
    case f32_isfinite: return "__nv_finitef";
    case f32_isinf:    return "__nv_isinff";
    case f32_isnan:    return "__nv_isnanf";
    case f32_min:      return "__nv_fminf";
    case f32_max:      return "__nv_fmaxf";
    case f64_abs:      return "__nv_fabs";
    case f64_mod:      return "__nv_fmod";
    case f64_rem:      return "__nv_remainder";
    case f64_exp:      return "__nv_exp";
    case f64_exp2:     return "__nv_exp2";
    case f64_exp10:    return "__nv_exp10";
    case f64_log:      return "__nv_log";
    case f64_log2:     return "__nv_log2";
    case f64_log10:    return "__nv_log10";
    case f64_pow:      return "__nv_pow";
    case f64_sqrt:     return "__nv_sqrt";
    case f64_rsqrt:    return "__nv_rsqrt";
    case f64_sin:      return "__nv_sin";
    case f64_cos:      return "__nv_cos";
    case f64_tan:      return "__nv_tan";
    case f64_asin:     return "__nv_asin";
    case f64_acos:     return "__nv_acos";
    case f64_atan:     return "__nv_atan";
    case f64_atan2:    return "__nv_atan2";
    case f64_ceil:     return "__nv_ceil";
    case f64_floor:    return "__nv_floor";
    case f64_trunc:    return "__nv_trunc";
    case f64_round:    return "__nv_round";
    case f64_isfinite: return "__nv_isfinited";
    case f64_isinf:    return "__nv_isinfd";
    case f64_isnan:    return "__nv_isnand";
    case f64_min:      return "__nv_fmin";
    case f64_max:      return "__nv_fmax";
    default:           return nullptr;
    }
}

CUJ_NAMESPACE_END(cuj::gen::libdev)
