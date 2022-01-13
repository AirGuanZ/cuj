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
#include <llvm/Linker/Linker.h>

#ifdef _MSC_VER
#pragma warning(pop)
#endif

#include "./libdevice_man.h"

CUJ_NAMESPACE_BEGIN(cuj::gen::libdev)

namespace
{

    const std::set<std::string> keeped_libdevice_functions = {
        "__nv_fabsf",
        "__nv_fmodf",
        "__nv_remainderf",
        "__nv_expf",
        "__nv_exp2f",
        "__nv_exp10f",
        "__nv_logf",
        "__nv_log2f",
        "__nv_log10f",
        "__nv_powf",
        "__nv_sqrtf",
        "__nv_rsqrtf",
        "__nv_sinf",
        "__nv_cosf",
        "__nv_tanf",
        "__nv_asinf",
        "__nv_acosf",
        "__nv_atanf",
        "__nv_atan2f",
        "__nv_ceilf",
        "__nv_floorf",
        "__nv_truncf",
        "__nv_roundf",
        "__nv_finitef",
        "__nv_isinff",
        "__nv_isnanf",
        "__nv_fminf",
        "__nv_fmaxf",
        "__nv_fabs",
        "__nv_fmod",
        "__nv_remainder",
        "__nv_exp",
        "__nv_exp2",
        "__nv_exp10",
        "__nv_log",
        "__nv_log2",
        "__nv_log10",
        "__nv_pow",
        "__nv_sqrt",
        "__nv_rsqrt",
        "__nv_sin",
        "__nv_cos",
        "__nv_tan",
        "__nv_asin",
        "__nv_acos",
        "__nv_atan",
        "__nv_atan2",
        "__nv_ceil",
        "__nv_floor",
        "__nv_trunc",
        "__nv_round",
        "__nv_isfinited",
        "__nv_isinfd",
        "__nv_isnand",
        "__nv_fmin",
        "__nv_fmax",
    };

} // namespace anonymous

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

#if defined(DEBUG) || defined(_DEBUG)
    std::string err;
    llvm::raw_string_ostream err_ss(err);
    if(verifyModule(*result, &err_ss))
        throw CujException(err);
#endif

    return result;
}

void link_with_libdevice(llvm::Module &dest_module)
{
    auto &context = dest_module.getContext();
    auto libdev_module = libdev::new_libdevice10_module(&context);

    std::vector<std::string> useless_libdev_func_names;
    std::vector<std::string> used_libdev_func_names;
    for(auto &f : *libdev_module)
    {
        if(!f.hasName() || f.isDeclaration())
            continue;
        std::string name = f.getName().str();
        if(f.getNumUses() == 0 && !keeped_libdevice_functions.contains(name))
            useless_libdev_func_names.push_back(std::move(name));
        else
            used_libdev_func_names.push_back(std::move(name));
    }

    for(auto &name : useless_libdev_func_names)
    {
        auto func = libdev_module->getFunction(name);
        func->eraseFromParent();
    }

    libdev_module->setTargetTriple("nvptx64-nvidia-cuda");
    dest_module.setDataLayout(libdev_module->getDataLayout());

    if(llvm::Linker::linkModules(dest_module, std::move(libdev_module)))
        throw CujException("failed to link with libdevice");

    for(auto &name : used_libdev_func_names)
    {
        auto func = dest_module.getFunction(name);
        if(func)
            func->setLinkage(llvm::GlobalValue::InternalLinkage);
    }
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
