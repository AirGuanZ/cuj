#if CUJ_ENABLE_LLVM

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
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Verifier.h>
#include <llvm/Transforms/InstCombine/InstCombine.h>
#include <llvm/Transforms/Scalar.h>
#include <llvm/Transforms/Scalar/GVN.h>
#include "llvm/Transforms/Utils.h"

#ifdef _MSC_VER
#pragma warning(pop)
#endif

#if CUJ_ENABLE_CUDA
#include <llvm/IR/IntrinsicsNVPTX.h>
#endif // #if CUJ_ENABLE_CUDA

#include <cuj/gen/llvm.h>

CUJ_NAMESPACE_BEGIN(cuj::gen)

namespace
{

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

} // namespace anonymous

struct LLVMIRGenerator::Data
{
    // global

    std::unique_ptr<llvm::LLVMContext> context;
    std::unique_ptr<llvm::IRBuilder<>> ir_builder;
    std::unique_ptr<llvm::Module>      top_module;

    std::unique_ptr<llvm::legacy::FunctionPassManager> fpm;

    std::map<const ir::Type *, llvm::Type *> types;
    std::map<std::string, llvm::Function *>  functions;

    // per function

    llvm::Function                   *function = nullptr;
    std::map<int, llvm::AllocaInst *> index_to_allocas;

    std::map<ir::BasicTempValue, llvm::Value *> temp_values_;

    std::stack<llvm::BasicBlock *> break_dests;
    std::stack<llvm::BasicBlock *> continue_dests;
};

LLVMIRGenerator::~LLVMIRGenerator()
{
    if(data_)
        delete data_;
}

void LLVMIRGenerator::set_target(Target target)
{
    target_ = target;
}

void LLVMIRGenerator::set_machine(
    const llvm::DataLayout *data_layout, const char *target_triple)
{
    data_layout_   = data_layout;
    target_triple_ = target_triple;
}

void LLVMIRGenerator::generate(const ir::Program &prog)
{
    CUJ_ASSERT(!data_);
    data_ = new Data;

    data_->context    = newBox<llvm::LLVMContext>();
    data_->ir_builder = newBox<llvm::IRBuilder<>>(*data_->context);
    data_->top_module = newBox<llvm::Module>("cuj", *data_->context);

    if(data_layout_)
        data_->top_module->setDataLayout(*data_layout_);
    if(target_triple_)
        data_->top_module->setTargetTriple(target_triple_);

    data_->fpm = newBox<llvm::legacy::FunctionPassManager>(data_->top_module.get());
    data_->fpm->add(llvm::createPromoteMemoryToRegisterPass());
    data_->fpm->add(llvm::createSROAPass());
    data_->fpm->add(llvm::createEarlyCSEPass());
    data_->fpm->add(llvm::createInstructionCombiningPass());
    data_->fpm->add(llvm::createReassociatePass());
    data_->fpm->add(llvm::createGVNPass());
    data_->fpm->add(llvm::createCFGSimplificationPass());
    data_->fpm->doInitialization();

    for(auto &p : prog.types)
        find_llvm_type(p.second.get());

    for(auto &p : prog.types)
        construct_struct_type_body(p.second.get());

    for(auto &f : prog.funcs)
        generate_func_decl(*f);

    for(auto &f : prog.funcs)
        generate_func(*f);
}

llvm::Module *LLVMIRGenerator::get_module() const
{
    return data_->top_module.get();
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
    const auto it = data_->types.find(type);
    if(it != data_->types.end())
        return it->second;
    
    auto result = type->match(
        [this](      ir::BuiltinType    t) { return create_llvm_type_record(t); },
        [this](const ir::ArrayType     &t) { return create_llvm_type_record(t); },
        [this](const ir::IntrinsicType &t) { return create_llvm_type_record(t); },
        [this](const ir::PointerType   &t) { return create_llvm_type_record(t); },
        [this](const ir::StructType    &t) { return create_llvm_type_record(t); });

    CUJ_ASSERT(!data_->types.count(type));
    data_->types.insert({ type, result });

    return result;
}

llvm::Type *LLVMIRGenerator::create_llvm_type_record(ir::BuiltinType type)
{
    switch(type)
    {
    case ir::BuiltinType::Void:
        return data_->ir_builder->getVoidTy();
    case ir::BuiltinType::U8:
    case ir::BuiltinType::S8:
    case ir::BuiltinType::Bool:
        return data_->ir_builder->getInt8Ty();
    case ir::BuiltinType::U16:
    case ir::BuiltinType::S16:
        return data_->ir_builder->getInt16Ty();
    case ir::BuiltinType::U32:
    case ir::BuiltinType::S32:
        return data_->ir_builder->getInt32Ty();
    case ir::BuiltinType::U64:
    case ir::BuiltinType::S64:
        return data_->ir_builder->getInt64Ty();
    case ir::BuiltinType::F32:
        return data_->ir_builder->getFloatTy();
    case ir::BuiltinType::F64:
        return data_->ir_builder->getDoubleTy();
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
    throw std::runtime_error("unknown intrinsic type: " + type.name);
}

llvm::Type *LLVMIRGenerator::create_llvm_type_record(const ir::PointerType &type)
{
    auto elem_type = find_llvm_type(type.pointed_type);
    auto result = llvm::PointerType::get(elem_type, 0);
    return result;
}

llvm::Type *LLVMIRGenerator::create_llvm_type_record(const ir::StructType &type)
{
    auto result = llvm::StructType::create(*data_->context, type.name);
    return result;
}

void LLVMIRGenerator::construct_struct_type_body(const ir::Type *type)
{
    auto struct_type = type->as_if<ir::StructType>();
    if(!struct_type)
        return;

    std::vector<llvm::Type *> mem_types;
    for(auto &mem : struct_type->mem_types)
        mem_types.push_back(find_llvm_type(mem));
    
    auto llvm_type = reinterpret_cast<llvm::StructType *>(find_llvm_type(type));
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

void LLVMIRGenerator::generate_func(const ir::Function &func)
{
    CUJ_ASSERT(data_);
    CUJ_ASSERT(!data_->function);

    auto func_it = data_->functions.find(func.name);
    CUJ_ASSERT(func_it != data_->functions.end());
    data_->function = func_it->second;

    auto entry_block =
        llvm::BasicBlock::Create(*data_->context, "entry", data_->function);
    data_->ir_builder->SetInsertPoint(entry_block);

    generate_func_allocs(func);

    copy_func_args(func);

    generate(*func.body);

    const ir::BuiltinType ret_type = func.ret_type->as<ir::BuiltinType>();
    if(ret_type == ir::BuiltinType::Void)
        data_->ir_builder->CreateRetVoid();
    else
    {
        auto zero = get_builtin_zero(ret_type, *data_->ir_builder);
        data_->ir_builder->CreateRet(zero);
    }

    std::string err_msg;
    llvm::raw_string_ostream err_stream(err_msg);
    if(verifyFunction(*data_->function, &err_stream))
        throw std::runtime_error(err_msg);

    data_->fpm->run(*data_->function);

    data_->function = nullptr;
    data_->index_to_allocas.clear();
    data_->temp_values_.clear();

    CUJ_ASSERT(data_->break_dests.empty());
    CUJ_ASSERT(data_->continue_dests.empty());
}

llvm::FunctionType *LLVMIRGenerator::generate_func_type(const ir::Function &func)
{
    std::vector<llvm::Type *> arg_types;
    for(auto &arg : func.args)
    {
        auto it = func.index_to_allocs.find(arg.alloc_index);
        CUJ_ASSERT(it != func.index_to_allocs.end());
        auto &alloc = it->second;
        auto llvm_arg_type = find_llvm_type(alloc->type);
        arg_types.push_back(llvm_arg_type);
    }

    auto ret_type = find_llvm_type(func.ret_type);
    return llvm::FunctionType::get(ret_type, arg_types, false);
}

void LLVMIRGenerator::mark_func_type(
    const ir::Function &func, llvm::Function *llvm_func)
{
    if(target_ == Target::Host)
    {
        if(func.type != ir::Function::Type::Default)
        {
            throw std::runtime_error(
                "only default function is supported by host llvm ir generator");
        }
    }
#if CUJ_ENABLE_CUDA
    else if(target_ == Target::PTX)
    {
        if(func.type == ir::Function::Type::Kernel)
        {
            auto constant_1 = llvm::ConstantInt::get(
                *data_->context, llvm::APInt(32, 1, true));

            llvm::Metadata *mds[] = {
                llvm::ValueAsMetadata::get(llvm_func),
                llvm::MDString::get(*data_->context, "kernel"),
                llvm::ValueAsMetadata::get(constant_1)
            };

            auto md_node = llvm::MDNode::get(*data_->context, mds);

            llvm_func->getParent()
                     ->getOrInsertNamedMetadata("nvvm.annotations")
                     ->addOperand(md_node);
        }
    }
#endif // #if CUJ_ENABLE_CUDA
    else
        throw std::runtime_error("unknown target type");
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
    size_t arg_idx = 0;
    for(auto &arg : data_->function->args())
    {
        auto alloc = data_->index_to_allocas[func.args[arg_idx++].alloc_index];
        data_->ir_builder->CreateStore(&arg, alloc);
    }
}

void LLVMIRGenerator::generate(const ir::Statement &s)
{
    s.match(
        [this](const ir::Store    &_s) { generate(_s); },
        [this](const ir::Assign   &_s) { generate(_s); },
        [this](const ir::Break    &_s) { generate(_s); },
        [this](const ir::Continue &_s) { generate(_s); },
        [this](const ir::Block    &_s) { generate(_s); },
        [this](const ir::If       &_s) { generate(_s); },
        [this](const ir::While    &_s) { generate(_s); },
        [this](const ir::Return   &_s) { generate(_s); },
        [this](const ir::Call     &_s) { generate(_s); });
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
        throw std::runtime_error("invalid break statement: no outer loop");
    data_->ir_builder->CreateBr(data_->break_dests.top());
}

void LLVMIRGenerator::generate(const ir::Continue &)
{
    if(data_->continue_dests.empty())
        throw std::runtime_error("invalid continue statement: no outer loop");
    data_->ir_builder->CreateBr(data_->continue_dests.top());
}

void LLVMIRGenerator::generate(const ir::Block &block)
{
    for(auto &s : block.stats)
        generate(*s);
}

void LLVMIRGenerator::generate(const ir::If &if_s)
{
    auto cond = get_value(if_s.cond);

    auto then_block  = llvm::BasicBlock::Create(*data_->context, "then");
    auto merge_block = llvm::BasicBlock::Create(*data_->context, "merge");

    auto else_block = if_s.else_block ?
        llvm::BasicBlock::Create(*data_->context, "else") : nullptr;

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
    auto cond_block  = llvm::BasicBlock::Create(*data_->context, "cond");
    auto body_block  = llvm::BasicBlock::Create(*data_->context, "body");
    auto merge_block = llvm::BasicBlock::Create(*data_->context, "merge");

    data_->ir_builder->CreateBr(cond_block);

    data_->function->getBasicBlockList().push_back(cond_block);
    data_->ir_builder->SetInsertPoint(cond_block);
    generate(*while_s.calculate_cond);

    auto cond = get_value(while_s.cond);
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
        *data_->context, "after_return", data_->function);
    data_->ir_builder->SetInsertPoint(block);
}

void LLVMIRGenerator::generate(const ir::Call &call)
{
    std::vector<llvm::Value *> args;
    for(auto &a : call.op.args)
        args.push_back(get_value(a));

    auto func = data_->top_module->getFunction(call.op.name);
    data_->ir_builder->CreateCall(func, args);
}

llvm::Value *LLVMIRGenerator::get_value(const ir::Value &v)
{
    return v.match(
        [this](const ir::BasicValue      &v) { return get_value(v); },
        [this](const ir::BinaryOp        &v) { return get_value(v); },
        [this](const ir::UnaryOp         &v) { return get_value(v); },
        [this](const ir::LoadOp          &v) { return get_value(v); },
        [this](const ir::CallOp          &v) { return get_value(v); },
        [this](const ir::CastOp          &v) { return get_value(v); },
        [this](const ir::ArrayElemAddrOp &v) { return get_value(v); },
        [this](const ir::IntrinsicOp     &v) { return get_value(v); },
        [this](const ir::MemberPtrOp     &v) { return get_value(v); },
        [this](const ir::PointerOffsetOp &v) { return get_value(v); });
}

llvm::Value *LLVMIRGenerator::get_value(const ir::BasicValue &v)
{
    return v.match(
        [this](const ir::BasicTempValue      &v) { return get_value(v); },
        [this](const ir::BasicImmediateValue &v) { return get_value(v); },
        [this](const ir::AllocAddress        &v) { return get_value(v); });
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
    case ir::BinaryOp::Type::And:
    {
        CUJ_ASSERT(operand_type == ir::BuiltinType::Bool);
        return data_->ir_builder->CreateAnd(lhs, rhs);
    }
    case ir::BinaryOp::Type::Or:
    {
        CUJ_ASSERT(operand_type == ir::BuiltinType::Bool);
        return data_->ir_builder->CreateOr(lhs, rhs);
    }
    case ir::BinaryOp::Type::XOr:
    {
        CUJ_ASSERT(operand_type == ir::BuiltinType::Bool);
        return data_->ir_builder->CreateXor(lhs, rhs);
    }
    case ir::BinaryOp::Type::Equal:
    {
        if(is_integral)
            return data_->ir_builder->CreateICmpEQ(lhs, rhs);
        return data_->ir_builder->CreateFCmpOEQ(lhs, rhs);
    }
    case ir::BinaryOp::Type::NotEqual:
    {
        if(is_integral)
            return data_->ir_builder->CreateICmpNE(lhs, rhs);
        return data_->ir_builder->CreateFCmpONE(lhs, rhs);
    }
    case ir::BinaryOp::Type::Less:
    {
        if(is_integral)
        {
            if(is_signed)
                return data_->ir_builder->CreateICmpSLT(lhs, rhs);
            return data_->ir_builder->CreateICmpULT(lhs, rhs);
        }
        return data_->ir_builder->CreateFCmpOLT(lhs, rhs);
    }
    case ir::BinaryOp::Type::LessEqual:
    {
        if(is_integral)
        {
            if(is_signed)
                return data_->ir_builder->CreateICmpSLE(lhs, rhs);
            return data_->ir_builder->CreateICmpULE(lhs, rhs);
        }
        return data_->ir_builder->CreateFCmpOLE(lhs, rhs);
    }
    case ir::BinaryOp::Type::Greater:
    {
        if(is_integral)
        {
            if(is_signed)
                return data_->ir_builder->CreateICmpSGT(lhs, rhs);
            return data_->ir_builder->CreateICmpUGT(lhs, rhs);
        }
        return data_->ir_builder->CreateFCmpOGT(lhs, rhs);
    }
    case ir::BinaryOp::Type::GreaterEqual:
    {
        if(is_integral)
        {
            if(is_signed)
                return data_->ir_builder->CreateICmpSGE(lhs, rhs);
            return data_->ir_builder->CreateICmpUGE(lhs, rhs);
        }
        return data_->ir_builder->CreateFCmpOGE(lhs, rhs);
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

llvm::Value *LLVMIRGenerator::get_value(const ir::CastOp &v)
{
    const auto from_type = get_arithmetic_type(v.val);
    const auto to_type   = v.to_type;
    auto from = get_value(v.val);
    return convert_arithmetic(from, from_type, to_type);
}

llvm::Value *LLVMIRGenerator::get_value(const ir::ArrayElemAddrOp &v)
{
    auto arr = get_value(v.arr_alloc);
    
    std::vector<llvm::Value *> indices(2);
    indices[0] = llvm::ConstantInt::get(
        *data_->context, llvm::APInt(32, 0, false));
    indices[1] = llvm::ConstantInt::get(
        *data_->context, llvm::APInt(32, 0, false));

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
                throw std::runtime_error(                                       \
                    "cuda intrinsic is not supported in host mode");            \
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

    throw std::runtime_error("unknown intrinsic calling: " + v.name);
}

llvm::Value *LLVMIRGenerator::get_value(const ir::MemberPtrOp &v)
{
    auto struct_ptr = get_value(v.ptr);
    auto struct_type = find_llvm_type(v.ptr_type);
    
    std::vector<llvm::Value *> indices(2);
    indices[0] = llvm::ConstantInt::get(
        *data_->context, llvm::APInt(32, 0, true));
    indices[1] = llvm::ConstantInt::get(
        *data_->context, llvm::APInt(32, v.member_index, true));

    return data_->ir_builder->CreateGEP(struct_type, struct_ptr, indices);
}

llvm::Value *LLVMIRGenerator::get_value(const ir::PointerOffsetOp &v)
{
    auto ptr = get_value(v.ptr);
    auto idx = get_value(v.index);
    std::vector<llvm::Value *> indices = { idx };
    return data_->ir_builder->CreateGEP(ptr, indices);
}

llvm::Value *LLVMIRGenerator::get_value(const ir::BasicTempValue &v)
{
    CUJ_ASSERT(data_->temp_values_.count(v));
    return data_->temp_values_[v];
}

llvm::Value *LLVMIRGenerator::get_value(const ir::BasicImmediateValue &v)
{
#define CUJ_IMM_INT(NAME, BITS, SIGNED)                                         \
    [this](NAME v) -> llvm::Value*                                              \
    {                                                                           \
        return llvm::ConstantInt::get(                                          \
            *data_->context,                                                    \
            llvm::APInt(BITS, static_cast<uint64_t>(v), SIGNED));               \
    }

    return v.value.match(
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
            [](uint8_t)  { return ir::BuiltinType::U8; },
            [](uint16_t) { return ir::BuiltinType::U16; },
            [](uint32_t) { return ir::BuiltinType::U32; },
            [](uint64_t) { return ir::BuiltinType::U64; },
            [](int8_t)   { return ir::BuiltinType::S8; },
            [](int16_t)  { return ir::BuiltinType::S16; },
            [](int32_t)  { return ir::BuiltinType::S32; },
            [](int64_t)  { return ir::BuiltinType::S64; },
            [](float)    { return ir::BuiltinType::F32; },
            [](double)   { return ir::BuiltinType::F64; },
            [](bool)     { return ir::BuiltinType::Bool; });
    },
        [](const ir::AllocAddress &) -> ir::BuiltinType
    {
        unreachable();
    });
}

CUJ_NAMESPACE_END(cuj::gen)

#endif // #if CUJ_ENABLE_LLVM
