#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable: 4141)
#pragma warning(disable: 4244)
#pragma warning(disable: 4624)
#pragma warning(disable: 4626)
#pragma warning(disable: 4996)
#endif

#include <llvm/IR/IRBuilder.h>

#ifdef _MSC_VER
#pragma warning(pop)
#endif

#include <cuj/gen/llvm.h>

CUJ_NAMESPACE_BEGIN(cuj::gen)

namespace
{

    template<typename...Args>
    llvm::Function *add_func_prototype(
        llvm::Module *top_module,
        const char   *name,
        bool          readnone,
        llvm::Type   *ret,
        Args...       arg_types)
    {
        std::vector<llvm::Type *> args = { arg_types... };
        auto func_type = llvm::FunctionType::get(ret, args, false);
        auto result = llvm::Function::Create(
            func_type, llvm::GlobalValue::ExternalLinkage, name, top_module);
        if(readnone)
            result->addFnAttr(llvm::Attribute::ReadNone);
        return result;
    }
    
    llvm::Function *get_host_math_intrinsics(
        const std::string &name,
        llvm::LLVMContext *context,
        llvm::Module      *top_module)
    {
        if(auto func = top_module->getFunction(name))
            return func;

        auto i32 = llvm::Type::getInt32Ty(*context);
        auto f32 = llvm::Type::getFloatTy(*context);
        auto f64 = llvm::Type::getDoubleTy(*context);
    
#define TRY_REGISTER_FUNC(NAME, READNONE, ...)                                  \
    do {                                                                        \
        if(name == NAME)                                                        \
            return add_func_prototype(top_module, NAME, READNONE, __VA_ARGS__); \
    } while(false)
        
        TRY_REGISTER_FUNC("host.math.abs.f32",       true, f32, f32);
        TRY_REGISTER_FUNC("host.math.mod.f32",       true, f32, f32, f32);
        TRY_REGISTER_FUNC("host.math.remainder.f32", true, f32, f32, f32);
        TRY_REGISTER_FUNC("host.math.exp.f32",       true, f32, f32);
        TRY_REGISTER_FUNC("host.math.exp2.f32",      true, f32, f32);
        TRY_REGISTER_FUNC("host.math.log.f32",       true, f32, f32);
        TRY_REGISTER_FUNC("host.math.log2.f32",      true, f32, f32);
        TRY_REGISTER_FUNC("host.math.log10.f32",     true, f32, f32);
        TRY_REGISTER_FUNC("host.math.pow.f32",       true, f32, f32, f32);
        TRY_REGISTER_FUNC("host.math.sqrt.f32",      true, f32, f32);
        TRY_REGISTER_FUNC("host.math.sin.f32",       true, f32, f32);
        TRY_REGISTER_FUNC("host.math.cos.f32",       true, f32, f32);
        TRY_REGISTER_FUNC("host.math.tan.f32",       true, f32, f32);
        TRY_REGISTER_FUNC("host.math.asin.f32",      true, f32, f32);
        TRY_REGISTER_FUNC("host.math.acos.f32",      true, f32, f32);
        TRY_REGISTER_FUNC("host.math.atan.f32",      true, f32, f32);
        TRY_REGISTER_FUNC("host.math.atan2.f32",     true, f32, f32, f32);
        TRY_REGISTER_FUNC("host.math.ceil.f32",      true, f32, f32);
        TRY_REGISTER_FUNC("host.math.floor.f32",     true, f32, f32);
        TRY_REGISTER_FUNC("host.math.trunc.f32",     true, f32, f32);
        TRY_REGISTER_FUNC("host.math.round.f32",     true, f32, f32);
        TRY_REGISTER_FUNC("host.math.isfinite.f32",  true, i32, f32);
        TRY_REGISTER_FUNC("host.math.isinf.f32",     true, i32, f32);
        TRY_REGISTER_FUNC("host.math.isnan.f32",     true, i32, f32);
        
        TRY_REGISTER_FUNC("host.math.abs.f64",       true, f64, f64);
        TRY_REGISTER_FUNC("host.math.mod.f64",       true, f64, f64, f64);
        TRY_REGISTER_FUNC("host.math.remainder.f64", true, f64, f64, f64);
        TRY_REGISTER_FUNC("host.math.exp.f64",       true, f64, f64);
        TRY_REGISTER_FUNC("host.math.exp2.f64",      true, f64, f64);
        TRY_REGISTER_FUNC("host.math.log.f64",       true, f64, f64);
        TRY_REGISTER_FUNC("host.math.log2.f64",      true, f64, f64);
        TRY_REGISTER_FUNC("host.math.log10.f64",     true, f64, f64);
        TRY_REGISTER_FUNC("host.math.pow.f64",       true, f64, f64, f64);
        TRY_REGISTER_FUNC("host.math.sqrt.f64",      true, f64, f64);
        TRY_REGISTER_FUNC("host.math.sin.f64",       true, f64, f64);
        TRY_REGISTER_FUNC("host.math.cos.f64",       true, f64, f64);
        TRY_REGISTER_FUNC("host.math.tan.f64",       true, f64, f64);
        TRY_REGISTER_FUNC("host.math.asin.f64",      true, f64, f64);
        TRY_REGISTER_FUNC("host.math.acos.f64",      true, f64, f64);
        TRY_REGISTER_FUNC("host.math.atan.f64",      true, f64, f64);
        TRY_REGISTER_FUNC("host.math.atan2.f64",     true, f64, f64, f64);
        TRY_REGISTER_FUNC("host.math.ceil.f64",      true, f64, f64);
        TRY_REGISTER_FUNC("host.math.floor.f64",     true, f64, f64);
        TRY_REGISTER_FUNC("host.math.trunc.f64",     true, f64, f64);
        TRY_REGISTER_FUNC("host.math.round.f64",     true, f64, f64);
        TRY_REGISTER_FUNC("host.math.isfinite.f64",  true, i32, f64);
        TRY_REGISTER_FUNC("host.math.isinf.f64",     true, i32, f64);
        TRY_REGISTER_FUNC("host.math.isnan.f64",     true, i32, f64);
    
        throw CUJException("unknown host math function: " + name);
    }

} // namespace anonymous

bool process_host_intrinsic_stat(
    llvm::LLVMContext               *context,
    llvm::Module                    *top_module,
    llvm::IRBuilder<>               &ir,
    const std::string               &name,
    const std::vector<llvm::Value*> &args)
{
    if(name == "system.print")
    {
        const std::string func_name = "host." + name;
        llvm::Function *func = top_module->getFunction(func_name);
        if(!func)
        {
            auto void_type = llvm::Type::getVoidTy(*context);
            auto pchar_type = llvm::PointerType::get(
                llvm::Type::getInt8Ty(*context), 0);
            func = add_func_prototype(
                top_module, func_name.c_str(), false, void_type, pchar_type);
        }

        ir.CreateCall(func, args);
        return true;
    }

    if(name == "system.free")
    {
        const std::string func_name = "host." + name;
        llvm::Function *func = top_module->getFunction(func_name);
        if(!func)
        {
            auto void_type = llvm::Type::getVoidTy(*context);
            auto ptr_type = llvm::PointerType::get(
                llvm::Type::getInt8Ty(*context), 0);
            func = add_func_prototype(
                top_module, func_name.c_str(), false, void_type, ptr_type);
        }

        ir.CreateCall(func, args);
        return true;
    }

    if(name == "system.assertfail")
    {
        const std::string func_name = "host." + name;
        auto func = top_module->getFunction(func_name);
        if(!func)
        {
            auto void_type  = llvm::Type::getVoidTy(*context);
            auto i8_type    = llvm::IntegerType::getInt8Ty(*context);
            auto pchar_type = llvm::PointerType::get(i8_type, 0);
            auto u32_type   = llvm::IntegerType::getInt32Ty(*context);

            func = add_func_prototype(
                top_module, func_name.c_str(), false, void_type,
                pchar_type, pchar_type, u32_type, pchar_type);
        }

        ir.CreateCall(func, args);
        return true;
    }

    return false;
}

llvm::Value *process_host_intrinsic_op(
    llvm::LLVMContext                *context,
    llvm::Module                     *top_module,
    llvm::IRBuilder<>                &ir,
    const std::string                &name,
    const std::vector<llvm::Value *> &args)
{
#define CUJ_HOST_MATH_INTRINSIC(NAME)                                           \
    do {                                                                        \
        if(name == NAME)                                                        \
        {                                                                       \
            auto func_name = "host." NAME;                                      \
            auto func = get_host_math_intrinsics(                               \
                func_name, context, top_module);                                \
            CUJ_INTERNAL_ASSERT(func);                                          \
            return ir.CreateCall(func, args);                                   \
        }                                                                       \
    } while(false)

    CUJ_HOST_MATH_INTRINSIC("math.abs.f32");
    CUJ_HOST_MATH_INTRINSIC("math.mod.f32");
    CUJ_HOST_MATH_INTRINSIC("math.remainder.f32");
    CUJ_HOST_MATH_INTRINSIC("math.exp.f32");
    CUJ_HOST_MATH_INTRINSIC("math.exp2.f32");
    CUJ_HOST_MATH_INTRINSIC("math.log.f32");
    CUJ_HOST_MATH_INTRINSIC("math.log2.f32");
    CUJ_HOST_MATH_INTRINSIC("math.log10.f32");
    CUJ_HOST_MATH_INTRINSIC("math.pow.f32");
    CUJ_HOST_MATH_INTRINSIC("math.sqrt.f32");
    CUJ_HOST_MATH_INTRINSIC("math.sin.f32");
    CUJ_HOST_MATH_INTRINSIC("math.cos.f32");
    CUJ_HOST_MATH_INTRINSIC("math.tan.f32");
    CUJ_HOST_MATH_INTRINSIC("math.asin.f32");
    CUJ_HOST_MATH_INTRINSIC("math.acos.f32");
    CUJ_HOST_MATH_INTRINSIC("math.atan.f32");
    CUJ_HOST_MATH_INTRINSIC("math.atan2.f32");
    CUJ_HOST_MATH_INTRINSIC("math.ceil.f32");
    CUJ_HOST_MATH_INTRINSIC("math.floor.f32");
    CUJ_HOST_MATH_INTRINSIC("math.trunc.f32");
    CUJ_HOST_MATH_INTRINSIC("math.round.f32");
    CUJ_HOST_MATH_INTRINSIC("math.isfinite.f32");
    CUJ_HOST_MATH_INTRINSIC("math.isinf.f32");
    CUJ_HOST_MATH_INTRINSIC("math.isnan.f32");

    CUJ_HOST_MATH_INTRINSIC("math.abs.f64");
    CUJ_HOST_MATH_INTRINSIC("math.mod.f64");
    CUJ_HOST_MATH_INTRINSIC("math.remainder.f64");
    CUJ_HOST_MATH_INTRINSIC("math.exp.f64");
    CUJ_HOST_MATH_INTRINSIC("math.exp2.f64");
    CUJ_HOST_MATH_INTRINSIC("math.log.f64");
    CUJ_HOST_MATH_INTRINSIC("math.log2.f64");
    CUJ_HOST_MATH_INTRINSIC("math.log10.f64");
    CUJ_HOST_MATH_INTRINSIC("math.pow.f64");
    CUJ_HOST_MATH_INTRINSIC("math.sqrt.f64");
    CUJ_HOST_MATH_INTRINSIC("math.sin.f64");
    CUJ_HOST_MATH_INTRINSIC("math.cos.f64");
    CUJ_HOST_MATH_INTRINSIC("math.tan.f64");
    CUJ_HOST_MATH_INTRINSIC("math.asin.f64");
    CUJ_HOST_MATH_INTRINSIC("math.acos.f64");
    CUJ_HOST_MATH_INTRINSIC("math.atan.f64");
    CUJ_HOST_MATH_INTRINSIC("math.atan2.f64");
    CUJ_HOST_MATH_INTRINSIC("math.ceil.f64");
    CUJ_HOST_MATH_INTRINSIC("math.floor.f64");
    CUJ_HOST_MATH_INTRINSIC("math.trunc.f64");
    CUJ_HOST_MATH_INTRINSIC("math.round.f64");
    CUJ_HOST_MATH_INTRINSIC("math.isfinite.f64");
    CUJ_HOST_MATH_INTRINSIC("math.isinf.f64");
    CUJ_HOST_MATH_INTRINSIC("math.isnan.f64");

#undef CUJ_HOST_MATH_INTRINSIC

    if(name == "atomic.add.f32")
    {
        const std::string func_name = "host." + name;

        llvm::Function *func = top_module->getFunction(func_name);
        if(!func)
        {
            auto f32_type = ir.getFloatTy();
            auto pf32_type = llvm::PointerType::get(f32_type, 0);
            func = add_func_prototype(
                top_module, func_name.c_str(), false,
                f32_type, pf32_type, f32_type);
        }

        return ir.CreateCall(func, args);
    }

    if(name == "atomic.add.f64")
    {
        const std::string func_name = "host." + name;

        llvm::Function *func = top_module->getFunction(func_name);
        if(!func)
        {
            auto f64_type = ir.getDoubleTy();
            auto pf64_type = llvm::PointerType::get(f64_type, 0);
            func = add_func_prototype(
                top_module, func_name.c_str(), false,
                f64_type, pf64_type, f64_type);
        }

        return ir.CreateCall(func, args);
    }

    if(name == "system.malloc")
    {
        const std::string func_name = "host." + name;
        llvm::Function *func = top_module->getFunction(func_name);
        if(!func)
        {
            auto ptr_type = llvm::PointerType::get(
                llvm::Type::getInt8Ty(*context), 0);

            llvm::Type *size_type;
            if constexpr(sizeof(size_t) == 4)
                size_type = llvm::Type::getInt32Ty(*context);
            else
                size_type = llvm::Type::getInt64Ty(*context);

            func = add_func_prototype(
                top_module, func_name.c_str(), false, ptr_type, size_type);
        }

        return ir.CreateCall(func, args);
    }

    return nullptr;
}

CUJ_NAMESPACE_END(cuj::gen)
