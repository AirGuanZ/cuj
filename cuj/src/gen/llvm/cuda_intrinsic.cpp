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

namespace
{

    void create_sample_texture_intrinsic(
        llvm::LLVMContext               &ctx,
        llvm::IRBuilder<>               &ir,
        const std::vector<llvm::Value*> &args,
        llvm::Intrinsic::ID              id)
    {
        CUJ_INTERNAL_ASSERT(args.size() == 1 + 2 + 4);
        auto tex = args[0];
        auto u   = args[1];
        auto v   = args[2];
        auto r   = args[3];
        auto g   = args[4];
        auto b   = args[5];
        auto a   = args[6];

        auto call = ir.CreateIntrinsic(id, { }, { tex, u, v });

        auto raw_type = call->getType();
        CUJ_INTERNAL_ASSERT(raw_type->isStructTy());
        auto type = static_cast<llvm::StructType*>(call->getType());

        auto alloc = ir.CreateAlloca(type);
        ir.CreateStore(call, alloc);

        std::array<llvm::Value *, 2> indices = {};
        indices[0] = llvm::ConstantInt::get(ctx, llvm::APInt(32, 0));

        indices[1] = llvm::ConstantInt::get(ctx, llvm::APInt(32, 0));
        auto member_addr = ir.CreateGEP(type, alloc, indices);
        auto member = ir.CreateLoad(member_addr);
        ir.CreateStore(member, r);

        indices[1] = llvm::ConstantInt::get(ctx, llvm::APInt(32, 1));
        member_addr = ir.CreateGEP(type, alloc, indices);
        member = ir.CreateLoad(member_addr);
        ir.CreateStore(member, g);

        indices[1] = llvm::ConstantInt::get(ctx, llvm::APInt(32, 2));
        member_addr = ir.CreateGEP(type, alloc, indices);
        member = ir.CreateLoad(member_addr);
        ir.CreateStore(member, b);

        indices[1] = llvm::ConstantInt::get(ctx, llvm::APInt(32, 3));
        member_addr = ir.CreateGEP(type, alloc, indices);
        member = ir.CreateLoad(member_addr);
        ir.CreateStore(member, a);
    }

    llvm::Function *get_print_function(
        llvm::LLVMContext &ctx,
        llvm::Module      *top_module)
    {
        if(auto f = top_module->getFunction("cuda_system_print"))
            return f;

        llvm::Type *tp_type;
        if constexpr(sizeof(void *) == 8)
            tp_type = llvm::IntegerType::getInt64Ty(ctx);
        else
            tp_type = llvm::IntegerType::getInt32Ty(ctx);

        auto s32_type = llvm::IntegerType::getInt32Ty(ctx);
        auto char_type = llvm::IntegerType::getInt8Ty(ctx);
        auto pchar_type = llvm::PointerType::get(char_type, 0);
        auto void_type = llvm::Type::getVoidTy(ctx);

        auto raw_func_type = llvm::FunctionType::get(
            s32_type, { tp_type, tp_type }, true);
        auto raw_func = llvm::Function::Create(
            raw_func_type, llvm::GlobalValue::ExternalLinkage,
            "vprintf", top_module);

        auto func_type = llvm::FunctionType::get(
            void_type, { pchar_type }, false);
        auto func = llvm::Function::Create(
            func_type, llvm::GlobalValue::ExternalLinkage,
            "cuda_system_print", top_module);

        auto entry_block = llvm::BasicBlock::Create(ctx, "entry", func);

        llvm::IRBuilder<> ir(ctx);
        ir.SetInsertPoint(entry_block);

        // const string: "cuda.system.print.fmt"
        // "%s"
        llvm::Value *fmt_str;
        {
            constexpr int GLOBAL_ADDR_SPACE = 1;

            std::vector<llvm::Constant *> fmt_consts;
            fmt_consts.push_back(llvm::ConstantInt::get(char_type, '%'));
            fmt_consts.push_back(llvm::ConstantInt::get(char_type, 's'));
            fmt_consts.push_back(llvm::ConstantInt::get(char_type, '\0'));

            auto arr_type = llvm::ArrayType::get(char_type, 3);
            auto init_const = llvm::ConstantArray::get(arr_type, fmt_consts);

            auto global_var = new llvm::GlobalVariable(
                *top_module, arr_type, true,
                llvm::GlobalValue::InternalLinkage, init_const,
                "cuda.system.print.fmt", nullptr,
                llvm::GlobalValue::NotThreadLocal, GLOBAL_ADDR_SPACE);

            std::array<llvm::Value *, 2> indices;
            indices[0] = llvm::ConstantInt::get(ctx, llvm::APInt(32, 0, false));
            indices[1] = llvm::ConstantInt::get(ctx, llvm::APInt(32, 0, false));
            auto val = ir.CreateGEP(global_var, indices);

            auto src_type = llvm::PointerType::get(char_type, GLOBAL_ADDR_SPACE);
            auto dst_type = llvm::PointerType::get(char_type, 0);

            fmt_str = ir.CreateIntrinsic(
                llvm::Intrinsic::nvvm_ptr_global_to_gen,
                { dst_type, src_type }, { val });
        }

        auto var_buffer = ir.CreateAlloca(tp_type);
        ir.CreateStore(ir.CreatePtrToInt(&*func->arg_begin(), tp_type), var_buffer);

        auto arg1 = ir.CreatePtrToInt(fmt_str, tp_type);
        auto arg2 = ir.CreatePtrToInt(var_buffer, tp_type);
        ir.CreateCall(raw_func, { arg1, arg2 });

        ir.CreateRetVoid();

        func->addFnAttr(llvm::Attribute::AlwaysInline);
        return func;
    }

    llvm::Function *get_assertfail_function(
        llvm::LLVMContext &ctx,
        llvm::Module      *top_module)
    {
        if(auto f = top_module->getFunction("__assertfail"))
            return f;

        llvm::Type *tp_type;
        if constexpr(sizeof(void *) == 8)
            tp_type = llvm::IntegerType::getInt64Ty(ctx);
        else
            tp_type = llvm::IntegerType::getInt32Ty(ctx);

        auto b32_type = llvm::IntegerType::getInt32Ty(ctx);
        auto void_type = llvm::Type::getVoidTy(ctx);

        auto func_type = llvm::FunctionType::get(
            void_type, { tp_type, tp_type, b32_type, tp_type, tp_type }, false);
        auto func = llvm::Function::Create(
            func_type, llvm::GlobalValue::ExternalLinkage,
            "__assertfail", top_module);

        return func;
    }

} // namespace anonymous

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
        CUJ_INTERNAL_ASSERT(func);
        func->setLinkage(llvm::GlobalValue::InternalLinkage);
    }
}

bool process_cuda_intrinsic_stat(
    llvm::LLVMContext               &ctx,
    llvm::Module                    *top_module,
    llvm::IRBuilder<>               &ir,
    const std::string               &name,
    const std::vector<llvm::Value*> &args)
{
    if(name == "system.assertfail")
    {
        CUJ_INTERNAL_ASSERT(args.size() == 4);

        auto message   = args[0];
        auto file      = args[1];
        auto line      = args[2];
        auto func_name = args[3];

        auto tp_type = sizeof(void *) == 8 ?
            llvm::IntegerType::getInt64Ty(ctx) :
            llvm::IntegerType::getInt32Ty(ctx);

        message   = ir.CreatePtrToInt(message, tp_type);
        file      = ir.CreatePtrToInt(file, tp_type);
        func_name = ir.CreatePtrToInt(func_name, tp_type);
        auto char_size = llvm::ConstantInt::get(tp_type, 1);

        auto func = get_assertfail_function(ctx, top_module);
        ir.CreateCall(func, { message, file, line, func_name, char_size });
        return true;
    }

    if(name == "system.print")
    {
        CUJ_INTERNAL_ASSERT(args.size() == 1);
        auto func = get_print_function(ctx, top_module);
        ir.CreateCall(func, args);
        return true;
    }

    if(name == "cuda.thread_block_barrier")
    {
        CUJ_INTERNAL_ASSERT(args.empty());
        ir.CreateIntrinsic(llvm::Intrinsic::nvvm_barrier0, {}, {});
        return true;
    }

    if(name == "cuda.sample.2d.f32")
    {
        create_sample_texture_intrinsic(
            ctx, ir, args, llvm::Intrinsic::nvvm_tex_unified_2d_v4f32_f32);
        return true;
    }

    if(name == "cuda.sample.2d.i32")
    {
        create_sample_texture_intrinsic(
            ctx, ir, args, llvm::Intrinsic::nvvm_tex_unified_2d_v4s32_f32);
        return true;
    }

    return false;
}

llvm::Value *process_cuda_intrinsic_op(
    llvm::Module                     *top_module,
    llvm::IRBuilder<>                &ir,
    const std::string                &name,
    const std::vector<llvm::Value *> &args,
    bool                              approx_math_funcs)
{
#define CUJ_CUDA_INTRINSIC_SREG(NAME, ID)                                       \
    do {                                                                        \
        if(name == NAME)                                                        \
        {                                                                       \
            CUJ_INTERNAL_ASSERT(args.empty());                                  \
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
            CUJ_INTERNAL_ASSERT(func);                                          \
            if(!func->hasFnAttribute(llvm::Attribute::ReadNone))                \
                func->addFnAttr(llvm::Attribute::ReadNone);                     \
            return ir.CreateCall(func, args);                                   \
        }                                                                       \
    } while(false)

    if(approx_math_funcs && name == "math.sin.f32")
    {
        return ir.CreateIntrinsic(
            llvm::Intrinsic::nvvm_sin_approx_ftz_f, {}, args);
    }

    if(approx_math_funcs && name == "math.cos.f32")
    {
        return ir.CreateIntrinsic(
            llvm::Intrinsic::nvvm_cos_approx_ftz_f, {}, args);
    }

    if(approx_math_funcs && name == "math.sqrt.f32")
    {
        return ir.CreateIntrinsic(
            llvm::Intrinsic::nvvm_sqrt_approx_ftz_f, {}, args);
    }

    if(approx_math_funcs && name == "math.floor.f32")
    {
        return ir.CreateIntrinsic(
            llvm::Intrinsic::nvvm_floor_f, {}, args);
    }

    if(approx_math_funcs && name == "math.ceil.f32")
    {
        return ir.CreateIntrinsic(
            llvm::Intrinsic::nvvm_ceil_f, {}, args);
    }

    if(approx_math_funcs && name == "math.rsqrt.f32")
    {
        return ir.CreateIntrinsic(
            llvm::Intrinsic::nvvm_rsqrt_approx_ftz_f, {}, args);
    }

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
    CUJ_CALL_LIBDEVICE(rsqrt,     true);
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
    CUJ_CALL_LIBDEVICE(rsqrt,     false);
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

    if(name == "atomic.add.f32" || name == "atomic.add.f64")
    {
        return ir.CreateAtomicRMW(
            llvm::AtomicRMWInst::FAdd, args[0], args[1],
            llvm::AtomicOrdering::SequentiallyConsistent);
    }

    return nullptr;
}

CUJ_NAMESPACE_END(cuj::gen)

#endif // #if CUJ_ENABLE_CUDA
