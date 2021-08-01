#include <stack>

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable: 4141)
#pragma warning(disable: 4244)
#pragma warning(disable: 4624)
#pragma warning(disable: 4626)
#pragma warning(disable: 4996)
#endif

#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/Intrinsics.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Verifier.h>
#include <llvm/Transforms/InstCombine/InstCombine.h>
#include <llvm/Transforms/IPO/GlobalDCE.h>
#include <llvm/Transforms/IPO/PassManagerBuilder.h>
#include <llvm/Transforms/Scalar.h>
#include <llvm/Transforms/Scalar/GVN.h>
#include <llvm/Transforms/Utils.h>
#include <llvm/LinkAllPasses.h>

#if CUJ_ENABLE_CUDA
#include <llvm/IR/IntrinsicsNVPTX.h>
#endif // #if CUJ_ENABLE_CUDA

#ifdef _MSC_VER
#pragma warning(pop)
#endif

#include <cuj/gen/llvm.h>

CUJ_NAMESPACE_BEGIN(cuj::gen)

namespace
{

    Box<llvm::LLVMContext> llvm_ctx;

    std::map<const ir::Type*, llvm::Type *> llvm_types;

    bool is_builtin_integral(ir::BuiltinType t)
    {
        switch(t)
        {
        case ir::BuiltinType::U8:
        case ir::BuiltinType::U16:
        case ir::BuiltinType::U32:
        case ir::BuiltinType::U64:
        case ir::BuiltinType::S8:
        case ir::BuiltinType::S16:
        case ir::BuiltinType::S32:
        case ir::BuiltinType::S64:
        case ir::BuiltinType::Bool:
        case ir::BuiltinType::Char:
            return true;
        case ir::BuiltinType::Void:
        case ir::BuiltinType::F32:
        case ir::BuiltinType::F64:
            return false;
        }
        unreachable();
    }

    bool is_builtin_signed(ir::BuiltinType t)
    {
        switch(t)
        {
        case ir::BuiltinType::Void:
        case ir::BuiltinType::U8:
        case ir::BuiltinType::U16:
        case ir::BuiltinType::U32:
        case ir::BuiltinType::U64:
        case ir::BuiltinType::Bool:
            return false;
        case ir::BuiltinType::S8:
        case ir::BuiltinType::S16:
        case ir::BuiltinType::S32:
        case ir::BuiltinType::S64:
        case ir::BuiltinType::F32:
        case ir::BuiltinType::F64:
            return true;
        case ir::BuiltinType::Char:
            return std::is_signed_v<char>;
        }
        unreachable();
    }

    llvm::Value *get_builtin_zero(ir::BuiltinType type, llvm::IRBuilder<> &ir)
    {
        CUJ_ASSERT(type != ir::BuiltinType::Void);
        switch(type)
        {
        case ir::BuiltinType::Void:
            unreachable();
        case ir::BuiltinType::Char:
        case ir::BuiltinType::Bool:
        case ir::BuiltinType::U8:
        case ir::BuiltinType::S8:
            return ir.getInt8(0);
        case ir::BuiltinType::U16:
        case ir::BuiltinType::S16:
            return ir.getInt16(0);
        case ir::BuiltinType::U32:
        case ir::BuiltinType::S32:
            return ir.getInt32(0);
        case ir::BuiltinType::U64:
        case ir::BuiltinType::S64:
            return ir.getInt64(0);
        case ir::BuiltinType::F32:
            return llvm::ConstantFP::get(ir.getFloatTy(), 0);
        case ir::BuiltinType::F64:
            return llvm::ConstantFP::get(ir.getDoubleTy(), 0);
        }
        unreachable();
    }

    const char *get_target_name(LLVMIRGenerator::Target target)
    {
        switch(target)
        {
        case LLVMIRGenerator::Target::Host: return "host";
#if CUJ_ENABLE_CUDA
        case LLVMIRGenerator::Target::PTX:  return "ptx";
#endif
        }
        unreachable();
    }

} // namespace anonymous

void link_with_libdevice(
    llvm::LLVMContext *context,
    llvm::Module      *dest_module);

#if CUJ_ENABLE_CUDA
bool process_cuda_intrinsic_stat(
    llvm::LLVMContext              &ctx,
    llvm::IRBuilder<>              &ir,
    const std::string               name,
    const std::vector<llvm::Value*> args);

llvm::Value *process_cuda_intrinsic_op(
    llvm::Module                     *top_module,
    llvm::IRBuilder<>                &ir,
    const std::string                &name,
    const std::vector<llvm::Value *> &args);
#endif

bool process_host_intrinsic_stat(
    llvm::LLVMContext               *context,
    llvm::Module                    *top_module,
    llvm::IRBuilder<>               &ir,
    const std::string               &name,
    const std::vector<llvm::Value*> &args);

llvm::Value *process_host_intrinsic_op(
    llvm::LLVMContext                *context,
    llvm::Module                     *top_module,
    llvm::IRBuilder<>                &ir,
    const std::string                &name,
    const std::vector<llvm::Value *> &args);

struct LLVMIRGenerator::Data
{
    // global
    
    std::unique_ptr<llvm::IRBuilder<>> ir_builder;
    std::unique_ptr<llvm::Module>      top_module;
    
    std::map<std::string, llvm::Function *> functions;
    std::map<std::string, llvm::Function *> external_functions;
    
    std::map<
        std::pair<llvm::Type*, std::vector<unsigned char>>,
        llvm::GlobalVariable *> global_data_consts;

    // per function

    llvm::Value *func_ret_class_ptr_arg = nullptr;

    llvm::Function                   *function = nullptr;
    std::map<int, llvm::AllocaInst *> index_to_allocas;

    std::map<ir::BasicTempValue, llvm::Value *> temp_values_;

    std::stack<llvm::BasicBlock *> break_dests;
    std::stack<llvm::BasicBlock *> continue_dests;
};

LLVMIRGenerator::~LLVMIRGenerator()
{
    delete data_;
}

void LLVMIRGenerator::set_target(Target target)
{
    target_ = target;
}

void LLVMIRGenerator::generate(const ir::Program &prog, llvm::DataLayout *dl)
{
    if(!llvm_ctx)
        llvm_ctx = newBox<llvm::LLVMContext>();

    CUJ_ASSERT(!data_);
    data_ = new Data;

    CUJ_ASSERT(!dl_);
    dl_ = dl;
    
    data_->ir_builder = newBox<llvm::IRBuilder<>>(*llvm_ctx);
    data_->top_module = newBox<llvm::Module>("cuj", *llvm_ctx);

#if CUJ_ENABLE_CUDA
    if(target_ == Target::PTX)
    {
        data_->top_module->setTargetTriple("nvptx64-nvidia-cuda");
        link_with_libdevice(llvm_ctx.get(), data_->top_module.get());
    }
#endif

    if(dl_)
        data_->top_module->setDataLayout(*dl_);

    for(auto &p : prog.types)
        find_llvm_type(p.second.get());

    for(auto &p : prog.types)
        construct_struct_type_body(p.second.get());

    for(auto &f : prog.funcs)
    {
        f.match(
            [this](const RC<ir::Function> &func)
        {
            generate_func_decl(*func);
        },
            [this](const RC<ir::ImportedHostFunction> &func)
        {
            generate_func_decl(*func);
        });
    }

    std::set<llvm::Function *> all_funcs;
    for(auto &f : prog.funcs)
    {
        f.match(
            [&](const RC<ir::Function> &func)
        {
            auto llvm_func = generate_func(*func);
            all_funcs.insert(llvm_func);
        },
            [&](const RC<ir::ImportedHostFunction> &func)
        {
            if(func->context_data)
            {
                auto llvm_func = generate_func(*func);
                all_funcs.insert(llvm_func);
            }
        });
    }

    llvm::legacy::FunctionPassManager fpm(data_->top_module.get());
    fpm.add(llvm::createPromoteMemoryToRegisterPass());
    fpm.add(llvm::createSROAPass());
    fpm.add(llvm::createEarlyCSEPass());
    fpm.add(llvm::createBasicAAWrapperPass());
    fpm.add(llvm::createInstructionCombiningPass());
    fpm.add(llvm::createReassociatePass());
    fpm.add(llvm::createGVNPass());
    fpm.add(llvm::createDeadCodeEliminationPass());
    fpm.add(llvm::createCFGSimplificationPass());
    fpm.doInitialization();

    for(auto f : all_funcs)
        fpm.run(*f);

    // IMPROVE: module-level opt pipeline
    llvm::ModuleAnalysisManager mam;
    llvm::GlobalDCEPass().run(*data_->top_module, mam);
}

llvm::Module *LLVMIRGenerator::get_module() const
{
    return data_->top_module.get();
}

std::unique_ptr<llvm::Module> LLVMIRGenerator::get_module_ownership()
{
    return std::move(data_->top_module);
}

std::string LLVMIRGenerator::get_string() const
{
    std::string result;
    llvm::raw_string_ostream str_stream(result);

    str_stream << *data_->top_module;
    str_stream.flush();

    return result;
}

llvm::Type *LLVMIRGenerator::find_llvm_type(const ir::Type *type)
{
    const auto it = llvm_types.find(type);
    if(it != llvm_types.end())
        return it->second;
    
    auto result = type->match(
        [](      ir::BuiltinType    t) { return create_llvm_type_record(t); },
        [](const ir::ArrayType     &t) { return create_llvm_type_record(t); },
        [](const ir::IntrinsicType &t) { return create_llvm_type_record(t); },
        [](const ir::PointerType   &t) { return create_llvm_type_record(t); },
        [](const ir::StructType    &t) { return create_llvm_type_record(t); });

    CUJ_ASSERT(!llvm_types.count(type));
    llvm_types.insert({ type, result });

    return result;
}

llvm::Type *LLVMIRGenerator::create_llvm_type_record(ir::BuiltinType type)
{
    switch(type)
    {
    case ir::BuiltinType::Void:
        return llvm::Type::getVoidTy(*llvm_ctx);
    case ir::BuiltinType::Char:
    case ir::BuiltinType::U8:
    case ir::BuiltinType::S8:
    case ir::BuiltinType::Bool:
        return llvm::Type::getInt8Ty(*llvm_ctx);
    case ir::BuiltinType::U16:
    case ir::BuiltinType::S16:
        return llvm::Type::getInt16Ty(*llvm_ctx);
    case ir::BuiltinType::U32:
    case ir::BuiltinType::S32:
        return llvm::Type::getInt32Ty(*llvm_ctx);
    case ir::BuiltinType::U64:
    case ir::BuiltinType::S64:
        return llvm::Type::getInt64Ty(*llvm_ctx);
    case ir::BuiltinType::F32:
        return llvm::Type::getFloatTy(*llvm_ctx);
    case ir::BuiltinType::F64:
        return llvm::Type::getDoubleTy(*llvm_ctx);
    }
    unreachable();
}

llvm::Type *LLVMIRGenerator::create_llvm_type_record(const ir::ArrayType &type)
{
    auto elem_type = find_llvm_type(type.elem_type);
    auto result = llvm::ArrayType::get(elem_type, type.size);
    return result;
}

llvm::Type *LLVMIRGenerator::create_llvm_type_record(const ir::IntrinsicType &type)
{
    throw CUJException("unknown intrinsic type: " + type.name);
}

llvm::Type *LLVMIRGenerator::create_llvm_type_record(const ir::PointerType &type)
{
    llvm::Type *elem_type;
    if(auto builtin_type = type.pointed_type->as_if<ir::BuiltinType>();
       builtin_type && *builtin_type == ir::BuiltinType::Void)
        elem_type = llvm::Type::getInt8Ty(*llvm_ctx);
    else
        elem_type = find_llvm_type(type.pointed_type);

    auto result = llvm::PointerType::get(elem_type, 0);
    return result;
}

llvm::Type *LLVMIRGenerator::create_llvm_type_record(const ir::StructType &type)
{
    auto result = llvm::StructType::create(*llvm_ctx, type.name);
    return result;
}

void LLVMIRGenerator::construct_struct_type_body(const ir::Type *type)
{
    auto struct_type = type->as_if<ir::StructType>();
    if(!struct_type)
        return;

    auto llvm_type = reinterpret_cast<llvm::StructType *>(find_llvm_type(type));
    if(!llvm_type->isOpaque())
        return;

    std::vector<llvm::Type *> mem_types;
    for(auto &mem : struct_type->mem_types)
        mem_types.push_back(find_llvm_type(mem));

    llvm_type->setBody(mem_types);
}

void LLVMIRGenerator::generate_func_decl(const ir::Function &func)
{
    auto func_type = generate_func_type(func);

    auto llvm_func = llvm::Function::Create(
        func_type, llvm::GlobalValue::ExternalLinkage,
        func.name, data_->top_module.get());

    mark_func_type(func, llvm_func);

    data_->functions[func.name] = llvm_func;
}

void LLVMIRGenerator::generate_func_decl(const ir::ImportedHostFunction &func)
{
    if(target_ != Target::Host)
    {
        throw CUJException(
            "imported function is only available for 'host' target");
    }

    {
        auto func_type = generate_func_type(func, true);
        
        auto host_func_symbol_name = func.context_data ?
            ("_cuj_host_contexted_func_" + func.symbol_name) : func.symbol_name;

        auto llvm_func = llvm::Function::Create(
            func_type, llvm::Function::ExternalLinkage,
            host_func_symbol_name, data_->top_module.get());

        data_->functions[host_func_symbol_name] = llvm_func;
    }

    if(func.context_data)
    {
        auto func_type = generate_func_type(func, false);
        
        auto llvm_func = llvm::Function::Create(
            func_type, llvm::Function::ExternalLinkage,
            func.symbol_name, data_->top_module.get());

        data_->functions[func.symbol_name] = llvm_func;
    }
}

llvm::Function *LLVMIRGenerator::generate_func(const ir::Function &func)
{
    CUJ_ASSERT(data_);
    CUJ_ASSERT(!data_->function);

    auto func_it = data_->functions.find(func.name);
    CUJ_ASSERT(func_it != data_->functions.end());
    data_->function = func_it->second;

    if(func.ret_type->is<ir::StructType>() || func.ret_type->is<ir::ArrayType>())
        data_->func_ret_class_ptr_arg = &*data_->function->arg_begin();

    auto entry_block =
        llvm::BasicBlock::Create(*llvm_ctx, "entry", data_->function);
    data_->ir_builder->SetInsertPoint(entry_block);

    generate_func_allocs(func);

    copy_func_args(func);

    generate(*func.body);

    if(auto builtin_type = func.ret_type->as_if<ir::BuiltinType>())
    {
        if(*builtin_type == ir::BuiltinType::Void)
            data_->ir_builder->CreateRetVoid();
        else
        {
            auto zero = get_builtin_zero(*builtin_type, *data_->ir_builder);
            data_->ir_builder->CreateRet(zero);
        }
    }
    else if(func.ret_type->is<ir::PointerType>())
    {
        data_->ir_builder->CreateRet(
            llvm::ConstantPointerNull::get(
                static_cast<llvm::PointerType*>(find_llvm_type(func.ret_type))));
    }
    else
    {
        CUJ_ASSERT(func.ret_type->is<ir::StructType>() ||
                   func.ret_type->is<ir::ArrayType>());
        data_->ir_builder->CreateRetVoid();
    }

    std::string err_msg;
    llvm::raw_string_ostream err_stream(err_msg);
    if(verifyFunction(*data_->function, &err_stream))
        throw CUJException(err_msg);

    auto ret = data_->function;

    data_->function               = nullptr;
    data_->func_ret_class_ptr_arg = nullptr;

    data_->index_to_allocas.clear();
    data_->temp_values_.clear();

    CUJ_ASSERT(data_->break_dests.empty());
    CUJ_ASSERT(data_->continue_dests.empty());

    return ret;
}

llvm::Function *LLVMIRGenerator::generate_func(
    const ir::ImportedHostFunction &func)
{
    CUJ_ASSERT(func.context_data);

    auto func_it = data_->functions.find(func.symbol_name);
    CUJ_ASSERT(func_it != data_->functions.end());
    data_->function = func_it->second;

    auto entry_block =
        llvm::BasicBlock::Create(*llvm_ctx, "entry", data_->function);
    data_->ir_builder->SetInsertPoint(entry_block);

    std::vector<llvm::Value*> call_args;
    call_args.push_back(
        llvm::ConstantInt::get(
            llvm::IntegerType::getInt64Ty(*llvm_ctx),
            reinterpret_cast<uint64_t>(func.context_data->get<void>())));

    for(auto &arg : data_->function->args())
    {
        auto load_inst = data_->ir_builder->CreateLoad(&arg);
        call_args.push_back(load_inst);
    }

    auto callee = data_->top_module->getFunction(
        "_cuj_host_contexted_func_" + func.symbol_name);
    auto call_inst = data_->ir_builder->CreateCall(callee, call_args);

    if(callee->getReturnType() != llvm::Type::getVoidTy(*llvm_ctx))
        data_->ir_builder->CreateRet(call_inst);
    else
        data_->ir_builder->CreateRetVoid();

    std::string err_msg;
    llvm::raw_string_ostream err_stream(err_msg);
    if(verifyFunction(*data_->function, &err_stream))
        throw CUJException(err_msg);

    auto ret = data_->function;

    data_->function = nullptr;
    CUJ_ASSERT(data_->func_ret_class_ptr_arg == nullptr);
    CUJ_ASSERT(data_->index_to_allocas.empty());
    CUJ_ASSERT(data_->temp_values_.empty());

    return ret;
}

llvm::FunctionType *LLVMIRGenerator::generate_func_type(
    const ir::Function &func)
{
    const bool is_ret_struct_or_arr =
        func.ret_type->is<ir::StructType>() ||
        func.ret_type->is<ir::ArrayType>();

    std::vector<llvm::Type *> arg_types;

    if(is_ret_struct_or_arr)
    {
        auto llvm_ret_type = find_llvm_type(func.ret_type);
        auto arg_type = llvm::PointerType::get(llvm_ret_type, 0);
        arg_types.push_back(arg_type);
    }

    for(auto &arg : func.args)
    {
        auto it = func.index_to_allocs.find(arg.alloc_index);
        CUJ_ASSERT(it != func.index_to_allocs.end());
        auto alloc_type = it->second->type;

        if(alloc_type->is<ir::StructType>() || alloc_type->is<ir::ArrayType>())
        {
            auto llvm_deref_type = find_llvm_type(alloc_type);
            auto llvm_type = llvm::PointerType::get(llvm_deref_type, 0);
            arg_types.push_back(llvm_type);
        }
        else
        {
            auto llvm_arg_type = find_llvm_type(alloc_type);
            arg_types.push_back(llvm_arg_type);
        }
    }

    auto ret_type = is_ret_struct_or_arr ? data_->ir_builder->getVoidTy()
                                         : find_llvm_type(func.ret_type);
    return llvm::FunctionType::get(ret_type, arg_types, false);
}

llvm::FunctionType *LLVMIRGenerator::generate_func_type(
    const ir::ImportedHostFunction &func, bool consider_context)
{
    const bool is_ret_struct_or_arr =
        func.ret_type->is<ir::StructType>() ||
        func.ret_type->is<ir::ArrayType>();

    std::vector<llvm::Type *> arg_types;

    if(is_ret_struct_or_arr)
    {
        auto llvm_ret_type = find_llvm_type(func.ret_type);
        auto arg_type = llvm::PointerType::get(llvm_ret_type, 0);
        arg_types.push_back(arg_type);
    }

    if(consider_context && func.context_data)
        arg_types.push_back(llvm::IntegerType::getInt64Ty(*llvm_ctx));

    for(auto arg_type : func.arg_types)
    {
        if(arg_type->is<ir::StructType>() || arg_type->is<ir::ArrayType>())
        {
            auto llvm_deref_type = find_llvm_type(arg_type);
            auto llvm_type = llvm::PointerType::get(llvm_deref_type, 0);
            arg_types.push_back(llvm_type);
        }
        else
        {
            auto llvm_arg_type = find_llvm_type(arg_type);
            arg_types.push_back(llvm_arg_type);
        }
    }

    auto ret_type = is_ret_struct_or_arr ? data_->ir_builder->getVoidTy()
                                         : find_llvm_type(func.ret_type);
    return llvm::FunctionType::get(ret_type, arg_types, false);
}

void LLVMIRGenerator::mark_func_type(
    const ir::Function &func, llvm::Function *llvm_func)
{
    if(target_ == Target::Host)
    {
        if(func.type != ir::Function::Type::Default)
        {
            throw CUJException(
                "only default function is supported by host llvm ir generator");
        }
    }
#if CUJ_ENABLE_CUDA
    else if(target_ == Target::PTX)
    {
        if(func.type == ir::Function::Type::Kernel)
        {
            auto constant_1 = llvm::ConstantInt::get(
                *llvm_ctx, llvm::APInt(32, 1, true));

            llvm::Metadata *mds[] = {
                llvm::ValueAsMetadata::get(llvm_func),
                llvm::MDString::get(*llvm_ctx, "kernel"),
                llvm::ValueAsMetadata::get(constant_1)
            };

            auto md_node = llvm::MDNode::get(*llvm_ctx, mds);

            llvm_func->getParent()
                     ->getOrInsertNamedMetadata("nvvm.annotations")
                     ->addOperand(md_node);
        }
    }
#endif // #if CUJ_ENABLE_CUDA
    else
        throw CUJException("unknown target type");
}

void LLVMIRGenerator::generate_func_allocs(const ir::Function &func)
{
    constexpr int ADDRESS_SPACE = 0;

    for(auto &p : func.index_to_allocs)
    {
        auto llvm_type = find_llvm_type(p.second->type);
        auto alloca_inst = data_->ir_builder->CreateAlloca(
            llvm_type, ADDRESS_SPACE, nullptr,
            "local" + std::to_string(p.first));

        data_->index_to_allocas[p.first] = alloca_inst;
    }
}

void LLVMIRGenerator::copy_func_args(const ir::Function &func)
{
    auto it  = data_->function->arg_begin();
    auto end = data_->function->arg_end();

    if(func.ret_type->is<ir::StructType>() || func.ret_type->is<ir::ArrayType>())
        ++it;

    size_t arg_idx = 0;
    while(it != end)
    {
        auto &arg = *it++;

        const int alloc_index = func.args[arg_idx++].alloc_index;
        auto alloc = data_->index_to_allocas[alloc_index];

        auto alloc_it = func.index_to_allocs.find(alloc_index);
        CUJ_ASSERT(alloc_it != func.index_to_allocs.end());
        auto alloc_type = alloc_it->second->type;

        if(alloc_type->is<ir::StructType>() || alloc_type->is<ir::ArrayType>())
        {
            auto val = data_->ir_builder->CreateLoad(&arg);
            data_->ir_builder->CreateStore(val, alloc);
        }
        else
            data_->ir_builder->CreateStore(&arg, alloc);
    }
}

void LLVMIRGenerator::generate(const ir::Statement &s)
{
    s.match(
        [this](const ir::Store         &_s) { generate(_s); },
        [this](const ir::Assign        &_s) { generate(_s); },
        [this](const ir::Break         &_s) { generate(_s); },
        [this](const ir::Continue      &_s) { generate(_s); },
        [this](const ir::Block         &_s) { generate(_s); },
        [this](const ir::If            &_s) { generate(_s); },
        [this](const ir::While         &_s) { generate(_s); },
        [this](const ir::Return        &_s) { generate(_s); },
        [this](const ir::ReturnClass   &_s) { generate(_s); },
        [this](const ir::ReturnArray   &_s) { generate(_s); },
        [this](const ir::Call          &_s) { generate(_s); },
        [this](const ir::IntrinsicCall &_s) { generate(_s); });
}

void LLVMIRGenerator::generate(const ir::Store &store)
{
    llvm::Value *dst_ptr = get_value(store.dst_ptr);
    llvm::Value *src_val = get_value(store.src_val);
    data_->ir_builder->CreateStore(src_val, dst_ptr);
}

void LLVMIRGenerator::generate(const ir::Assign &assign)
{
    CUJ_ASSERT(!data_->temp_values_.count(assign.lhs));
    llvm::Value *rhs_val = get_value(assign.rhs);
    data_->temp_values_[assign.lhs] = rhs_val;
}

void LLVMIRGenerator::generate(const ir::Break &)
{
    if(data_->break_dests.empty())
        throw CUJException("invalid break statement: no outer loop");
    data_->ir_builder->CreateBr(data_->break_dests.top());

    auto new_block = llvm::BasicBlock::Create(
        *llvm_ctx, "after break", data_->function);
    data_->ir_builder->SetInsertPoint(new_block);
}

void LLVMIRGenerator::generate(const ir::Continue &)
{
    if(data_->continue_dests.empty())
        throw CUJException("invalid continue statement: no outer loop");
    data_->ir_builder->CreateBr(data_->continue_dests.top());

    auto new_block = llvm::BasicBlock::Create(
        *llvm_ctx, "after break", data_->function);
    data_->ir_builder->SetInsertPoint(new_block);
}

void LLVMIRGenerator::generate(const ir::Block &block)
{
    for(auto &s : block.stats)
        generate(*s);
}

void LLVMIRGenerator::generate(const ir::If &if_s)
{
    auto cond = data_->ir_builder->CreateZExtOrTrunc(
        get_value(if_s.cond), data_->ir_builder->getInt1Ty());

    auto then_block  = llvm::BasicBlock::Create(*llvm_ctx, "then");
    auto merge_block = llvm::BasicBlock::Create(*llvm_ctx, "merge");

    auto else_block = if_s.else_block ?
        llvm::BasicBlock::Create(*llvm_ctx, "else") : nullptr;

    data_->ir_builder->CreateCondBr(
        cond, then_block, else_block ? else_block : merge_block);

    data_->function->getBasicBlockList().push_back(then_block);
    data_->ir_builder->SetInsertPoint(then_block);
    generate(*if_s.then_block);
    data_->ir_builder->CreateBr(merge_block);

    if(else_block)
    {
        data_->function->getBasicBlockList().push_back(else_block);
        data_->ir_builder->SetInsertPoint(else_block);
        if(if_s.else_block)
            generate(*if_s.else_block);
        data_->ir_builder->CreateBr(merge_block);
    }

    data_->function->getBasicBlockList().push_back(merge_block);
    data_->ir_builder->SetInsertPoint(merge_block);
}

void LLVMIRGenerator::generate(const ir::While &while_s)
{
    auto cond_block  = llvm::BasicBlock::Create(*llvm_ctx, "cond");
    auto body_block  = llvm::BasicBlock::Create(*llvm_ctx, "body");
    auto merge_block = llvm::BasicBlock::Create(*llvm_ctx, "merge");

    data_->ir_builder->CreateBr(cond_block);

    data_->function->getBasicBlockList().push_back(cond_block);
    data_->ir_builder->SetInsertPoint(cond_block);
    generate(*while_s.calculate_cond);

    auto cond = data_->ir_builder->CreateZExtOrTrunc(
        get_value(while_s.cond), data_->ir_builder->getInt1Ty());
    data_->ir_builder->CreateCondBr(cond, body_block, merge_block);

    data_->continue_dests.push(cond_block);
    data_->break_dests.push(merge_block);

    data_->function->getBasicBlockList().push_back(body_block);
    data_->ir_builder->SetInsertPoint(body_block);
    generate(*while_s.body);
    data_->ir_builder->CreateBr(cond_block);

    data_->continue_dests.pop();
    data_->break_dests.pop();

    data_->function->getBasicBlockList().push_back(merge_block);
    data_->ir_builder->SetInsertPoint(merge_block);
}

void LLVMIRGenerator::generate(const ir::Return &return_s)
{
    if(return_s.value)
    {
        auto value = get_value(*return_s.value);
        data_->ir_builder->CreateRet(value);
    }
    else
        data_->ir_builder->CreateRetVoid();

    auto block = llvm::BasicBlock::Create(
        *llvm_ctx, "after_return", data_->function);
    data_->ir_builder->SetInsertPoint(block);
}

void LLVMIRGenerator::generate(const ir::ReturnClass &return_class)
{
    auto src_ptr = get_value(return_class.class_ptr);
    auto src_val = data_->ir_builder->CreateLoad(src_ptr);
    data_->ir_builder->CreateStore(src_val, data_->func_ret_class_ptr_arg);
}

void LLVMIRGenerator::generate(const ir::ReturnArray &return_array)
{
    auto src_ptr = get_value(return_array.array_ptr);
    auto src_val = data_->ir_builder->CreateLoad(src_ptr);
    data_->ir_builder->CreateStore(src_val, data_->func_ret_class_ptr_arg);
}

void LLVMIRGenerator::generate(const ir::Call &call)
{
    std::vector<llvm::Value *> args;
    for(auto &a : call.op.args)
        args.push_back(get_value(a));

    auto func = data_->top_module->getFunction(call.op.name);
    data_->ir_builder->CreateCall(func, args);
}

void LLVMIRGenerator::generate(const ir::IntrinsicCall &call)
{
    std::vector<llvm::Value *> args;
    for(auto &a : call.op.args)
        args.push_back(get_value(a));

#if CUJ_ENABLE_CUDA
    if(target_ == Target::PTX)
    {
        if(process_cuda_intrinsic_stat(
            *llvm_ctx, *data_->ir_builder, call.op.name, args))
            return;
    }
#endif

    if(target_ == Target::Host)
    {
        if(process_host_intrinsic_stat(
            llvm_ctx.get(), data_->top_module.get(),
            *data_->ir_builder, call.op.name, args))
            return;
    }

    throw CUJException(
        "unknown intrinsic: " + call.op.name +
        " on target " + get_target_name(target_));
}

llvm::Value *LLVMIRGenerator::get_value(const ir::Value &v)
{
    return v.match(
        [this](const ir::BasicValue      &v) { return get_value(v); },
        [this](const ir::BinaryOp        &v) { return get_value(v); },
        [this](const ir::UnaryOp         &v) { return get_value(v); },
        [this](const ir::LoadOp          &v) { return get_value(v); },
        [this](const ir::CallOp          &v) { return get_value(v); },
        [this](const ir::CastBuiltinOp   &v) { return get_value(v); },
        [this](const ir::CastPointerOp   &v) { return get_value(v); },
        [this](const ir::ArrayElemAddrOp &v) { return get_value(v); },
        [this](const ir::IntrinsicOp     &v) { return get_value(v); },
        [this](const ir::MemberPtrOp     &v) { return get_value(v); },
        [this](const ir::PointerOffsetOp &v) { return get_value(v); },
        [this](const ir::EmptyPointerOp  &v) { return get_value(v); },
        [this](const ir::PointerToUIntOp &v) { return get_value(v); },
        [this](const ir::PointerDiffOp   &v) { return get_value(v); });
}

llvm::Value *LLVMIRGenerator::get_value(const ir::BasicValue &v)
{
    return v.match(
        [this](const ir::BasicTempValue      &v) { return get_value(v); },
        [this](const ir::BasicImmediateValue &v) { return get_value(v); },
        [this](const ir::AllocAddress        &v) { return get_value(v); },
        [this](const ir::ConstData           &v) { return get_value(v); });
}

llvm::Value *LLVMIRGenerator::get_value(const ir::BinaryOp &v)
{
    auto lhs = get_value(v.lhs);
    auto rhs = get_value(v.rhs);
    
    const ir::BuiltinType operand_type = get_arithmetic_type(v.lhs);
    CUJ_ASSERT(operand_type == get_arithmetic_type(v.rhs));

    const bool is_integral = is_builtin_integral(operand_type);
    const bool is_signed   = is_builtin_signed  (operand_type);

    switch(v.type)
    {
    case ir::BinaryOp::Type::Add:
    {
        CUJ_ASSERT(operand_type != ir::BuiltinType::Bool);
        if(is_integral)
            return data_->ir_builder->CreateAdd(lhs, rhs);
        return data_->ir_builder->CreateFAdd(lhs, rhs);
    }
    case ir::BinaryOp::Type::Sub:
    {
        CUJ_ASSERT(operand_type != ir::BuiltinType::Bool);
        if(is_integral)
            return data_->ir_builder->CreateSub(lhs, rhs);
        return data_->ir_builder->CreateFSub(lhs, rhs);
    }
    case ir::BinaryOp::Type::Mul:
    {
        CUJ_ASSERT(operand_type != ir::BuiltinType::Bool);
        if(is_integral)
            return data_->ir_builder->CreateMul(lhs, rhs);
        return data_->ir_builder->CreateFMul(lhs, rhs);
    }
    case ir::BinaryOp::Type::Div:
    {
        CUJ_ASSERT(operand_type != ir::BuiltinType::Bool);
        if(is_integral)
        {
            if(is_signed)
                return data_->ir_builder->CreateSDiv(lhs, rhs);
            return data_->ir_builder->CreateUDiv(lhs, rhs);
        }
        return data_->ir_builder->CreateFDiv(lhs, rhs);
    }
    case ir::BinaryOp::Type::Mod:
    {
        CUJ_ASSERT(is_integral);
        if(is_signed)
            return data_->ir_builder->CreateSRem(lhs, rhs);
        return data_->ir_builder->CreateURem(lhs, rhs);
    }
    case ir::BinaryOp::Type::And:
    {
        CUJ_ASSERT(operand_type == ir::BuiltinType::Bool);
        return i1_to_bool(data_->ir_builder->CreateAnd(lhs, rhs));
    }
    case ir::BinaryOp::Type::Or:
    {
        CUJ_ASSERT(operand_type == ir::BuiltinType::Bool);
        return i1_to_bool(data_->ir_builder->CreateOr(lhs, rhs));
    }
    case ir::BinaryOp::Type::BitwiseAnd:
    {
        CUJ_ASSERT(is_integral);
        return data_->ir_builder->CreateAnd(lhs, rhs);
    }
    case ir::BinaryOp::Type::BitwiseOr:
    {
        CUJ_ASSERT(is_integral);
        return data_->ir_builder->CreateOr(lhs, rhs);
    }
    case ir::BinaryOp::Type::BitwiseXOr:
    {
        CUJ_ASSERT(is_integral);
        return data_->ir_builder->CreateXor(lhs, rhs);
    }
    case ir::BinaryOp::Type::Equal:
    {
        if(is_integral)
            return i1_to_bool(data_->ir_builder->CreateICmpEQ(lhs, rhs));
        return i1_to_bool(data_->ir_builder->CreateFCmpOEQ(lhs, rhs));
    }
    case ir::BinaryOp::Type::NotEqual:
    {
        if(is_integral)
            return i1_to_bool(data_->ir_builder->CreateICmpNE(lhs, rhs));
        return i1_to_bool(data_->ir_builder->CreateFCmpONE(lhs, rhs));
    }
    case ir::BinaryOp::Type::Less:
    {
        if(is_integral)
        {
            if(is_signed)
                return i1_to_bool(data_->ir_builder->CreateICmpSLT(lhs, rhs));
            return i1_to_bool(data_->ir_builder->CreateICmpULT(lhs, rhs));
        }
        return i1_to_bool(data_->ir_builder->CreateFCmpOLT(lhs, rhs));
    }
    case ir::BinaryOp::Type::LessEqual:
    {
        if(is_integral)
        {
            if(is_signed)
                return i1_to_bool(data_->ir_builder->CreateICmpSLE(lhs, rhs));
            return i1_to_bool(data_->ir_builder->CreateICmpULE(lhs, rhs));
        }
        return i1_to_bool(data_->ir_builder->CreateFCmpOLE(lhs, rhs));
    }
    case ir::BinaryOp::Type::Greater:
    {
        if(is_integral)
        {
            if(is_signed)
                return i1_to_bool(data_->ir_builder->CreateICmpSGT(lhs, rhs));
            return i1_to_bool(data_->ir_builder->CreateICmpUGT(lhs, rhs));
        }
        return i1_to_bool(data_->ir_builder->CreateFCmpOGT(lhs, rhs));
    }
    case ir::BinaryOp::Type::GreaterEqual:
    {
        if(is_integral)
        {
            if(is_signed)
                return i1_to_bool(data_->ir_builder->CreateICmpSGE(lhs, rhs));
            return i1_to_bool(data_->ir_builder->CreateICmpUGE(lhs, rhs));
        }
        return i1_to_bool(data_->ir_builder->CreateFCmpOGE(lhs, rhs));
    }
    }

    unreachable();
}

llvm::Value *LLVMIRGenerator::get_value(const ir::UnaryOp &v)
{
    auto input = get_value(v.input);
    auto input_type = get_arithmetic_type(v.input);

    switch(v.type)
    {
    case ir::UnaryOp::Type::Neg:
        switch(input_type)
        {
        case ir::BuiltinType::Void:
            std::terminate();
        case ir::BuiltinType::Char:
        case ir::BuiltinType::U8:
        case ir::BuiltinType::U16:
        case ir::BuiltinType::U32:
        case ir::BuiltinType::U64:
        case ir::BuiltinType::S8:
        case ir::BuiltinType::S16:
        case ir::BuiltinType::S32:
        case ir::BuiltinType::S64:
            return data_->ir_builder->CreateNeg(input);
        case ir::BuiltinType::F32:
        case ir::BuiltinType::F64:
            return data_->ir_builder->CreateFNeg(input);
        case ir::BuiltinType::Bool:
            unreachable();
        }
    case ir::UnaryOp::Type::Not:
        CUJ_ASSERT(input_type == ir::BuiltinType::Bool);
        return data_->ir_builder->CreateNot(input);
    }

    unreachable();
}

llvm::Value *LLVMIRGenerator::get_value(const ir::LoadOp &v)
{
    auto pointer = get_value(v.src_ptr);
    auto load_type = find_llvm_type(v.type);
    return data_->ir_builder->CreateLoad(load_type, pointer);
}

llvm::Value *LLVMIRGenerator::get_value(const ir::CallOp &v)
{
    std::vector<llvm::Value *> args;
    for(auto &a : v.args)
        args.push_back(get_value(a));

    auto func = data_->top_module->getFunction(v.name);
    return data_->ir_builder->CreateCall(func, args);
}

llvm::Value *LLVMIRGenerator::get_value(const ir::CastBuiltinOp &v)
{
    const auto from_type = get_arithmetic_type(v.val);
    const auto to_type   = v.to_type;
    auto from = get_value(v.val);
    return convert_arithmetic(from, from_type, to_type);
}

llvm::Value *LLVMIRGenerator::get_value(const ir::CastPointerOp &v)
{
    const auto to_type = find_llvm_type(v.to_type);
    auto from_val = get_value(v.from_val);
    return data_->ir_builder->CreatePointerCast(from_val, to_type);
}

llvm::Value *LLVMIRGenerator::get_value(const ir::ArrayElemAddrOp &v)
{
    auto arr = get_value(v.arr_alloc);
    
    std::vector<llvm::Value *> indices(2);
    indices[0] = llvm::ConstantInt::get(
        *llvm_ctx, llvm::APInt(32, 0, false));
    indices[1] = llvm::ConstantInt::get(
        *llvm_ctx, llvm::APInt(32, 0, false));

    return data_->ir_builder->CreateGEP(arr, indices, "array_element");
}

llvm::Value *LLVMIRGenerator::get_value(const ir::IntrinsicOp &v)
{
#if CUJ_ENABLE_CUDA

#define CUJ_CUDA_INTRINSIC_SREG(NAME, ID)                                       \
    do {                                                                        \
        if(v.name == NAME)                                                      \
        {                                                                       \
            if(target_ != Target::PTX)                                          \
            {                                                                   \
                throw CUJException(                                             \
                    "cuda intrinsic is not supported in non-ptx mode");         \
            }                                                                   \
            return data_->ir_builder->CreateIntrinsic(                          \
                llvm::Intrinsic::nvvm_read_ptx_sreg_##ID, {}, {});              \
        }                                                                       \
    } while(false)

    CUJ_CUDA_INTRINSIC_SREG("cuda.thread_index_x", tid_x);
    CUJ_CUDA_INTRINSIC_SREG("cuda.thread_index_y", tid_y);
    CUJ_CUDA_INTRINSIC_SREG("cuda.thread_index_z", tid_z);
    CUJ_CUDA_INTRINSIC_SREG("cuda.block_index_x",  ctaid_x);
    CUJ_CUDA_INTRINSIC_SREG("cuda.block_index_y",  ctaid_y);
    CUJ_CUDA_INTRINSIC_SREG("cuda.block_index_z",  ctaid_z);
    CUJ_CUDA_INTRINSIC_SREG("cuda.block_dim_x",    ntid_x);
    CUJ_CUDA_INTRINSIC_SREG("cuda.block_dim_y",    ntid_y);
    CUJ_CUDA_INTRINSIC_SREG("cuda.block_dim_z",    ntid_z);

#undef CUJ_CUDA_INTRINSIC_SREG

#endif // #if CUJ_ENABLE_CUDA

    std::vector<llvm::Value *> args;
    for(auto &a : v.args)
        args.push_back(get_value(a));

    if(target_ == Target::Host)
    {
        auto ret = process_host_intrinsic_op(
            llvm_ctx.get(), data_->top_module.get(),
            *data_->ir_builder, v.name, args);
        if(ret)
            return ret;
    }

#if CUJ_ENABLE_CUDA
    if(target_ == Target::PTX)
    {
        auto ret = process_cuda_intrinsic_op(
            data_->top_module.get(), *data_->ir_builder, v.name, args);
        if(ret)
            return ret;
    }
#endif

    throw CUJException(
        "unknown intrinsic " + v.name +
        " on target " + get_target_name(target_));
}

llvm::Value *LLVMIRGenerator::get_value(const ir::MemberPtrOp &v)
{
    auto struct_ptr = get_value(v.ptr);
    auto struct_type = find_llvm_type(v.ptr_type);
    
    std::array<llvm::Value *, 2> indices;
    indices[0] = llvm::ConstantInt::get(
        *llvm_ctx, llvm::APInt(32, 0));
    indices[1] = llvm::ConstantInt::get(
        *llvm_ctx, llvm::APInt(32, v.member_index));

    return data_->ir_builder->CreateGEP(struct_type, struct_ptr, indices);
}

llvm::Value *LLVMIRGenerator::get_value(const ir::PointerOffsetOp &v)
{
    auto ptr = get_value(v.ptr);
    auto idx = get_value(v.index);
    std::vector<llvm::Value *> indices = { idx };
    return data_->ir_builder->CreateGEP(ptr, indices);
}

llvm::Value *LLVMIRGenerator::get_value(const ir::EmptyPointerOp &v)
{
    auto type = find_llvm_type(v.ptr_type);
    CUJ_ASSERT(type->isPointerTy());
    return llvm::ConstantPointerNull::get(static_cast<llvm::PointerType*>(type));
}

llvm::Value *LLVMIRGenerator::get_value(const ir::PointerToUIntOp &v)
{
    auto ptr = get_value(v.ptr_val);
    llvm::Type *to_type;
    if constexpr(sizeof(size_t) == 4)
        to_type = data_->ir_builder->getInt32Ty();
    else
        to_type = data_->ir_builder->getInt64Ty();
    return data_->ir_builder->CreatePtrToInt(ptr, to_type);
}

llvm::Value *LLVMIRGenerator::get_value(const ir::PointerDiffOp &v)
{
    auto lhs = get_value(v.lhs);
    auto rhs = get_value(v.rhs);
    return data_->ir_builder->CreatePtrDiff(lhs, rhs);
}

llvm::Value *LLVMIRGenerator::get_value(const ir::BasicTempValue &v)
{
    CUJ_ASSERT(data_->temp_values_.count(v));
    return data_->temp_values_[v];
}

llvm::Value *LLVMIRGenerator::get_value(const ir::BasicImmediateValue &v)
{
#define CUJ_IMM_INT(NAME, BITS, SIGNED)                                         \
    [](NAME v) -> llvm::Value*                                                  \
    {                                                                           \
        return llvm::ConstantInt::get(                                          \
            *llvm_ctx,                                                          \
            llvm::APInt(BITS, static_cast<uint64_t>(v), SIGNED));               \
    }

    return v.value.match(
        CUJ_IMM_INT(char,     8,  std::is_signed_v<char>),
        CUJ_IMM_INT(uint8_t,  8,  false),
        CUJ_IMM_INT(uint16_t, 16, false),
        CUJ_IMM_INT(uint32_t, 32, false),
        CUJ_IMM_INT(uint64_t, 64, false),
        CUJ_IMM_INT(int8_t,   8,  true),
        CUJ_IMM_INT(int16_t,  16, true),
        CUJ_IMM_INT(int32_t,  32, true),
        CUJ_IMM_INT(int64_t,  64, true),
        [this](float v) -> llvm::Value *
    {
        return llvm::ConstantFP::get(
            data_->ir_builder->getFloatTy(), v);
    },
        [this](double v) -> llvm::Value *
    {
        return llvm::ConstantFP::get(
            data_->ir_builder->getDoubleTy(), v);
    },
        [this](bool v) -> llvm::Value *
    {
        return llvm::ConstantInt::get(
            data_->ir_builder->getInt8Ty(), v ? 1 : 0, false);
    });

#undef CUJ_IMM_INT
}

llvm::Value *LLVMIRGenerator::get_value(const ir::AllocAddress &v)
{
    CUJ_ASSERT(data_->index_to_allocas.count(v.alloc_index));
    return data_->index_to_allocas[v.alloc_index];
}

llvm::Value *LLVMIRGenerator::get_value(const ir::ConstData &v)
{
#if CUJ_ENABLE_CUDA
    const int GLOBAL_ADDR_SPACE = target_ == Target::PTX ? 1 : 0;
#else
    const int GLOBAL_ADDR_SPACE = 0;
#endif

    auto u8_type   = llvm::Type::getInt8Ty(*llvm_ctx);
    auto elem_type = find_llvm_type(v.elem_type);

    llvm::GlobalVariable *global_var;
    if(auto it = data_->global_data_consts.find({ elem_type, v.bytes });
       it == data_->global_data_consts.end())
    {
        auto arr_type = llvm::ArrayType::get(u8_type, v.bytes.size());

        std::vector<llvm::Constant *> byte_consts;
        byte_consts.reserve(v.bytes.size());
        for(auto b : v.bytes)
            byte_consts.push_back(llvm::ConstantInt::get(u8_type, b, false));
        auto init_const = llvm::ConstantArray::get(arr_type, byte_consts);

        global_var = new llvm::GlobalVariable(
            *data_->top_module, arr_type, true,
            llvm::GlobalValue::InternalLinkage, init_const,
            "", nullptr, llvm::GlobalValue::NotThreadLocal,
            GLOBAL_ADDR_SPACE);

        if(dl_)
            global_var->setAlignment(dl_->getPrefTypeAlign(elem_type));
        else
        {
            global_var->setAlignment(
                llvm::MaybeAlign(llvm::Align(alignof(void *))));
        }

        data_->global_data_consts[{ elem_type, v.bytes }] = global_var;
    }
    else
        global_var = it->second;

    std::vector<llvm::Value *> indices(2);
    indices[0] = llvm::ConstantInt::get(
        *llvm_ctx, llvm::APInt(32, 0, false));
    indices[1] = llvm::ConstantInt::get(
        *llvm_ctx, llvm::APInt(32, 0, false));

    auto val = data_->ir_builder->CreateGEP(global_var, indices);
#if CUJ_ENABLE_CUDA
    if(target_ == Target::PTX)
    {
        auto src_type = llvm::PointerType::get(u8_type, GLOBAL_ADDR_SPACE);
        auto dst_type = llvm::PointerType::get(u8_type, 0);

        val = data_->ir_builder->CreateIntrinsic(
            llvm::Intrinsic::nvvm_ptr_global_to_gen,
            { dst_type, src_type }, { val });
    }
#endif

    val = data_->ir_builder->CreatePointerCast(
        val, llvm::PointerType::get(elem_type, 0));

    return val;
}

llvm::Value *LLVMIRGenerator::convert_to_bool(
    llvm::Value *from, ir::BuiltinType from_type)
{
    if(from_type == ir::BuiltinType::Bool)
        return from;
    auto bool_type = data_->ir_builder->getInt8Ty();

    llvm::Value *cond_val;
    if(!is_builtin_integral(from_type))
    {
        cond_val = data_->ir_builder->CreateFCmpONE(
            from, llvm::ConstantFP::get(from->getType(), 0.0));
    }
    else
    {
        cond_val = data_->ir_builder->CreateICmpNE(
            from, llvm::ConstantInt::get(from->getType(), 0));
    }

    auto then_val = llvm::ConstantInt::get(bool_type, 1);
    auto else_val = llvm::ConstantInt::get(bool_type, 0);

    return data_->ir_builder->CreateSelect(cond_val, then_val, else_val);
}

llvm::Value *LLVMIRGenerator::convert_from_bool(
    llvm::Value *from, ir::BuiltinType to_type)
{
    CUJ_ASSERT(to_type != ir::BuiltinType::Void);
    CUJ_ASSERT(from->getType() == data_->ir_builder->getInt8Ty());

    if(to_type == ir::BuiltinType::Bool)
        return from;

    auto to_llvm_type = create_llvm_type_record(to_type);
    auto cmp_val = data_->ir_builder->CreateICmpNE(
        from, llvm::ConstantInt::get(from->getType(), 0));
    
    llvm::Value *then_val, *else_val;
    if(is_builtin_integral(to_type))
    {
        then_val = llvm::ConstantInt::get(to_llvm_type, 1);
        else_val = llvm::ConstantInt::get(to_llvm_type, 0);
    }
    else
    {
        then_val = llvm::ConstantFP::get(to_llvm_type, 1);
        else_val = llvm::ConstantFP::get(to_llvm_type, 0);
    }
    return data_->ir_builder->CreateSelect(cmp_val, then_val, else_val);
}

llvm::Value *LLVMIRGenerator::convert_arithmetic(
    llvm::Value *from, ir::BuiltinType from_type, ir::BuiltinType to_type)
{
    CUJ_ASSERT(from_type != ir::BuiltinType::Void);
    CUJ_ASSERT(to_type != ir::BuiltinType::Void);

    if(from_type == to_type)
        return from;
    if(from_type == ir::BuiltinType::Bool)
        return convert_from_bool(from, to_type);
    if(to_type == ir::BuiltinType::Bool)
        return convert_to_bool(from, from_type);
    
    const auto to_llvm_type   = create_llvm_type_record(to_type);

    const bool is_from_int = is_builtin_integral(from_type);
    const bool is_to_int   = is_builtin_integral(to_type);
    const bool is_from_s   = is_builtin_signed(from_type);
    const bool is_to_s     = is_builtin_signed(to_type);
    const bool is_from_flt = !is_from_int;
    const bool is_to_flt   = !is_to_int;
    
    if(is_from_int && is_to_int)
    {
        if(is_to_s)
            return data_->ir_builder->CreateSExtOrTrunc(from, to_llvm_type);
        return data_->ir_builder->CreateZExtOrTrunc(from, to_llvm_type);
    }

    if(is_from_int && is_to_flt)
    {
        if(is_from_s)
            return data_->ir_builder->CreateSIToFP(from, to_llvm_type);
        return data_->ir_builder->CreateUIToFP(from, to_llvm_type);
    }

    if(is_from_flt && is_to_int)
    {
        if(is_to_s)
            return data_->ir_builder->CreateFPToSI(from, to_llvm_type);
        return data_->ir_builder->CreateFPToUI(from, to_llvm_type);
    }

    if(is_from_flt && is_to_flt)
        return data_->ir_builder->CreateFPCast(from, to_llvm_type);

    unreachable();
}

ir::BuiltinType LLVMIRGenerator::get_arithmetic_type(const ir::BasicValue &v)
{
    return v.match(
        [](const ir::BasicTempValue &t)
    {
        return t.type->as<ir::BuiltinType>();
    },
        [](const ir::BasicImmediateValue &t)
    {
        return t.value.match(
            [](char c)   { return ir::BuiltinType::Char; },
            [](uint8_t)  { return ir::BuiltinType::U8;   },
            [](uint16_t) { return ir::BuiltinType::U16;  },
            [](uint32_t) { return ir::BuiltinType::U32;  },
            [](uint64_t) { return ir::BuiltinType::U64;  },
            [](int8_t)   { return ir::BuiltinType::S8;   },
            [](int16_t)  { return ir::BuiltinType::S16;  },
            [](int32_t)  { return ir::BuiltinType::S32;  },
            [](int64_t)  { return ir::BuiltinType::S64;  },
            [](float)    { return ir::BuiltinType::F32;  },
            [](double)   { return ir::BuiltinType::F64;  },
            [](bool)     { return ir::BuiltinType::Bool; });
    },
        [](const ir::AllocAddress &) -> ir::BuiltinType
    {
        unreachable();
    },
        [](const ir::ConstData &) -> ir::BuiltinType
    {
        unreachable();
    });
}

llvm::Value *LLVMIRGenerator::i1_to_bool(llvm::Value *val)
{
    return data_->ir_builder->CreateZExt(val, data_->ir_builder->getInt8Ty());
}

CUJ_NAMESPACE_END(cuj::gen)
