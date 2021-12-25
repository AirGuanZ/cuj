#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable: 4141)
#pragma warning(disable: 4244)
#pragma warning(disable: 4624)
#pragma warning(disable: 4626)
#pragma warning(disable: 4996)
#endif

#include <stack>

#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/Intrinsics.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IR/Verifier.h>
#include <llvm/Transforms/InstCombine/InstCombine.h>
#include <llvm/Transforms/IPO/GlobalDCE.h>
#include <llvm/Transforms/IPO/PassManagerBuilder.h>
#include <llvm/Transforms/Scalar.h>
#include <llvm/Transforms/Scalar/GVN.h>
#include <llvm/Transforms/Utils.h>
#include <llvm/LinkAllPasses.h>

#include <cuj/gen/llvm.h>
#include <cuj/utils/scope_guard.h>
#include <cuj/utils/unreachable.h>

#include "helper.h"
#include "native_intrinsics.h"
#include "ptx_intrinsics.h"

CUJ_NAMESPACE_BEGIN(cuj::gen)

struct LLVMIRGenerator::LLVMData
{
    // per module

    core::Prog             prog;
    Box<llvm::LLVMContext> context;

    std::map<std::type_index, llvm::Type *>       index_to_llvm_type;
    std::map<const core::Type *, std::type_index> type_to_index;

    Box<llvm::IRBuilder<>> ir_builder;
    Box<llvm::Module>      top_module;

    struct FunctionRecord
    {
        llvm::Function *llvm_function;
    };

    std::map<const core::Func *, FunctionRecord> llvm_functions_;

    // per function

    llvm::Function *current_function = nullptr;

    std::vector<llvm::AllocaInst *> local_allocas;
    std::vector<llvm::AllocaInst *> arg_allocas;

    std::stack<llvm::BasicBlock *> break_dsts;
    std::stack<llvm::BasicBlock *> continue_dsts;
};

LLVMIRGenerator::~LLVMIRGenerator()
{
    delete llvm_;
}

void LLVMIRGenerator::set_target(Target target)
{
    target_ = target;
}

void LLVMIRGenerator::use_fast_math()
{
    fast_math_ = true;
}

void LLVMIRGenerator::use_approx_math_func()
{
    approx_math_func_ = true;
}

void LLVMIRGenerator::disable_basic_optimizations()
{
    basic_optimizations_ = false;
}

void LLVMIRGenerator::set_data_layout(llvm::DataLayout *data_layout)
{
    data_layout_ = data_layout;
}

void LLVMIRGenerator::generate(const dsl::Module &mod)
{
    assert(!llvm_);
    llvm_ = new LLVMData;
    llvm_->context = newBox<llvm::LLVMContext>();

    llvm_->ir_builder = newBox<llvm::IRBuilder<>>(*llvm_->context);
    llvm_->top_module = newBox<llvm::Module>("cuj", *llvm_->context);

    if(fast_math_)
    {
        llvm::FastMathFlags flags;
        flags.setFast(true);
        llvm_->ir_builder->setFastMathFlags(flags);
    }

    if(target_ == Target::PTX)
    {
        llvm_->top_module->setTargetTriple("nvptx64-nvidia-cuda");
        link_with_libdevice(*llvm_->top_module);

        if(fast_math_)
        {
            llvm_->top_module->addModuleFlag(
                llvm::Module::Override, "nvvm-reflect-ftz", 1);
        }
    }

    if(data_layout_)
        llvm_->top_module->setDataLayout(*data_layout_);
    
    llvm_->prog = mod._generate_prog();
    auto &prog = llvm_->prog;

    // build llvm types

    {
        auto index_to_type = build_type_to_index(prog);

        for(auto &[_, type] : index_to_type)
            build_llvm_type(type);

        for(auto &[_, type] : index_to_type)
            build_llvm_struct_body(type);
    }

    // build functions

    std::set<llvm::Function *> all_functions;

    for(auto &f : prog.funcs)
        declare_function(f.get());

    for(auto &f : prog.funcs)
        define_function(f.get());

    llvm::legacy::FunctionPassManager fpm(llvm_->top_module.get());
    fpm.add(llvm::createPromoteMemoryToRegisterPass());
    fpm.add(llvm::createSROAPass());

    if(basic_optimizations_)
    {
        fpm.add(llvm::createEarlyCSEPass());
        fpm.add(llvm::createBasicAAWrapperPass());
        fpm.add(llvm::createInstructionCombiningPass());
        fpm.add(llvm::createReassociatePass());
        fpm.add(llvm::createGVNPass());
        fpm.add(llvm::createDeadCodeEliminationPass());
        fpm.add(llvm::createCFGSimplificationPass());
    }

    fpm.doInitialization();
    for(auto &[_, f] : llvm_->llvm_functions_)
        fpm.run(*f.llvm_function);
    fpm.doFinalization();

    if(basic_optimizations_)
    {
        llvm::ModuleAnalysisManager mam;
        llvm::GlobalDCEPass().run(*llvm_->top_module, mam);
    }
}

llvm::Module *LLVMIRGenerator::get_llvm_module() const
{
    assert(llvm_);
    return llvm_->top_module.get();
}

std::pair<Box<llvm::LLVMContext>, Box<llvm::Module>>
    LLVMIRGenerator::get_data_ownership()
{
    assert(llvm_);
    return { std::move(llvm_->context), std::move(llvm_->top_module) };
}

std::string LLVMIRGenerator::get_llvm_string() const
{
    std::string result;
    llvm::raw_string_ostream ss(result);
    ss << *llvm_->top_module;
    ss.flush();
    return result;
}

std::map<std::type_index, const core::Type *>
    LLVMIRGenerator::build_type_to_index(const core::Prog &prog)
{
    std::map<std::type_index, const core::Type *> ret;
    auto handle_type_set = [&](const core::TypeSet &set)
    {
        for(auto &[type, index] : set.type_to_index)
        {
            llvm_->type_to_index.try_emplace(type, index);
            ret.try_emplace(index, type);
        }
    };

    handle_type_set(*prog.global_type_set);
    for(auto &func : prog.funcs)
    {
        if(auto set = func->type_set)
            handle_type_set(*set);
    }

    return ret;
}

llvm::Type *LLVMIRGenerator::build_llvm_type(const core::Type *type)
{
    const auto index = llvm_->type_to_index.at(type);
    if(auto it = llvm_->index_to_llvm_type.find(index);
       it != llvm_->index_to_llvm_type.end())
        return it->second;

    auto llvm_type = type->match(
        [&](core::Builtin t) -> llvm::Type*
    {
        return llvm_helper::builtin_to_llvm_type(llvm_->context.get(), t);
    },
        [&](const core::Struct &t) -> llvm::Type *
    {
        return llvm::StructType::create(*llvm_->context);
    },
        [&](const core::Array &t) -> llvm::Type *
    {
        auto element = build_llvm_type(t.element);
        return llvm::ArrayType::get(element, t.size);
    },
        [&](const core::Pointer &t) -> llvm::Type *
    {
        if(auto pt = t.pointed->as_if<core::Builtin>();
           pt && *pt == core::Builtin::Void)
        {
            return llvm::PointerType::get(
                llvm::Type::getInt8Ty(*llvm_->context), 0);
        }
        auto pointed = build_llvm_type(t.pointed);
        return llvm::PointerType::get(pointed, 0);
    });

    llvm_->index_to_llvm_type.insert({ index, llvm_type });
    return llvm_type;
}

void LLVMIRGenerator::build_llvm_struct_body(const core::Type *type)
{
    auto struct_type = type->as_if<core::Struct>();
    if(!struct_type)
        return;

    auto llvm_type = get_llvm_type(type);
    auto llvm_struct_type = llvm::dyn_cast<llvm::StructType>(llvm_type);
    assert(llvm_struct_type->isOpaque());

    std::vector<llvm::Type *> member_types;
    for(auto member : struct_type->members)
        member_types.push_back(get_llvm_type(member));

    llvm_struct_type->setBody(member_types);
}

llvm::Type *LLVMIRGenerator::get_llvm_type(const core::Type *type) const
{
    const auto index = llvm_->type_to_index.at(type);
    return llvm_->index_to_llvm_type.at(index);
}

llvm::FunctionType *LLVMIRGenerator::get_function_type(const core::Func &func)
{
    if(func.type == core::Func::Kernel)
    {
        auto ret_type = func.return_type.type->as_if<core::Builtin>();
        if(!ret_type || *ret_type != core::Builtin::Void)
            throw CujException("kernel function must return void");
    }

    llvm::Type *ret_type = get_llvm_type(func.return_type.type);
    if(func.return_type.is_reference)
        ret_type = llvm::PointerType::get(ret_type, 0);

    std::vector<llvm::Type *> arg_types;
    for(auto &arg : func.argument_types)
    {
        llvm::Type *arg_type = get_llvm_type(arg.type);
        if(arg.is_reference)
            arg_type = llvm::PointerType::get(arg_type, 0);
        arg_types.push_back(arg_type);
    }

    return llvm::FunctionType::get(ret_type, arg_types, false);
}

void LLVMIRGenerator::declare_function(const core::Func *func)
{
    std::string symbol_name = func->name;
    if(symbol_name.empty())
    {
        symbol_name = "_cuj_function_"
            + std::to_string(llvm_->llvm_functions_.size());
    }
    if(llvm_->top_module->getFunction(symbol_name))
    {
        throw CujException(
            "multiple definitions of function " + symbol_name);
    }

    auto func_type = get_function_type(*func);
    auto llvm_func = llvm::Function::Create(
        func_type, llvm::GlobalValue::ExternalLinkage,
        symbol_name, llvm_->top_module.get());

    if(target_ == Target::PTX)
        llvm_func->addFnAttr("nvptx-f32ftz", "true");

    if(func->type == core::Func::Kernel)
    {
        if(target_ != Target::PTX)
            throw CujException("only ptx target supports kernel function");

        auto one = llvm_helper::llvm_constant_num(*llvm_->context, 1);
        llvm::Metadata *mds[] = {
            llvm::ValueAsMetadata::get(llvm_func),
            llvm::MDString::get(*llvm_->context, "kernel"),
            llvm::ValueAsMetadata::get(one)
        };
        auto md_node = llvm::MDNode::get(*llvm_->context, mds);

        llvm_func->getParent()
            ->getOrInsertNamedMetadata("nvvm.annotations")
            ->addOperand(md_node);
    }

    llvm_->llvm_functions_.insert({ func, { llvm_func } });
}

void LLVMIRGenerator::define_function(const core::Func *func)
{
    clear_temp_function_data();
    llvm_->current_function = llvm_->llvm_functions_.at(func).llvm_function;
    CUJ_SCOPE_EXIT{ llvm_->current_function = nullptr; };

    auto entry_block = llvm::BasicBlock::Create(
        *llvm_->context, "entry", llvm_->current_function);
    llvm_->ir_builder->SetInsertPoint(entry_block);

    generate_local_allocs(func);

    generate(*func->root_block);

    generate_default_ret(func);

    std::string err_msg;
    llvm::raw_string_ostream err_stream(err_msg);
    if(verifyFunction(*llvm_->current_function, &err_stream))
        throw CujException(err_msg);
}

void LLVMIRGenerator::clear_temp_function_data()
{
    llvm_->current_function = nullptr;
    llvm_->local_allocas.clear();
    llvm_->arg_allocas.clear();
    llvm_->break_dsts = {};
    llvm_->continue_dsts = {};
}

void LLVMIRGenerator::generate_local_allocs(const core::Func *func)
{
    constexpr int LOCAL_ALLOCA_ADDRESS_SPACE = 0;

    for(size_t i = 0; i < func->local_alloc_types.size(); ++i)
    {
        auto llvm_type = get_llvm_type(func->local_alloc_types[i]);
        auto alloca_inst = llvm_->ir_builder->CreateAlloca(
            llvm_type, LOCAL_ALLOCA_ADDRESS_SPACE,
            nullptr, "var" + std::to_string(i));
        llvm_->local_allocas.push_back(alloca_inst);
    }

    for(auto &arg : llvm_->current_function->args())
    {
        auto alloca_inst = llvm_->ir_builder->CreateAlloca(
            arg.getType(), LOCAL_ALLOCA_ADDRESS_SPACE, nullptr);
        llvm_->arg_allocas.push_back(alloca_inst);
        llvm_->ir_builder->CreateStore(&arg, alloca_inst);
    }
}

void LLVMIRGenerator::generate_default_ret(const core::Func *func)
{
    if(func->return_type.is_reference)
    {
        auto ret_type = get_llvm_type(func->return_type.type);
        llvm_->ir_builder->CreateRet(
            llvm::ConstantPointerNull::get(
                llvm::PointerType::get(ret_type, 0)));
    }
    else
    {
        auto llvm_type = get_llvm_type(func->return_type.type);
        func->return_type.type->match(
            [&](core::Builtin t)
        {
            if(t == core::Builtin::F32)
            {
                llvm_->ir_builder->CreateRet(
                        llvm_helper::llvm_constant_num(*llvm_->context, 0.0f));
            }
            else if(t == core::Builtin::F64)
            {
                llvm_->ir_builder->CreateRet(
                        llvm_helper::llvm_constant_num(*llvm_->context, 0.0));
            }
            else if(t == core::Builtin::Void)
            {
                llvm_->ir_builder->CreateRetVoid();
            }
            else
            {
                llvm_->ir_builder->CreateRet(
                    llvm::ConstantInt::get(llvm_type, 0));
            }
        },
            [&](const core::Struct &)
        {
            auto alloc = llvm_->ir_builder->CreateAlloca(llvm_type);
            llvm_->ir_builder->CreateRet(
                llvm_->ir_builder->CreateLoad(alloc));
        },
            [&](const core::Array &)
        {
            auto alloc = llvm_->ir_builder->CreateAlloca(llvm_type);
            llvm_->ir_builder->CreateRet(
                llvm_->ir_builder->CreateLoad(alloc));
        },
            [&](const core::Pointer &)
        {
            llvm_->ir_builder->CreateRet(
                llvm::ConstantPointerNull::get(
                    llvm::dyn_cast<llvm::PointerType>(llvm_type)));
        });
    }
}

void LLVMIRGenerator::generate(const core::Stat &stat)
{
    stat.match([&](auto &_s) { generate(_s); });
}

void LLVMIRGenerator::generate(const core::Store &store)
{
    auto dst_addr = generate(store.dst_addr);
    auto val = generate(store.val);
    llvm_->ir_builder->CreateStore(val, dst_addr);
}

void LLVMIRGenerator::generate(const core::Block &block)
{
    for(auto &s : block.stats)
        generate(*s);
}

void LLVMIRGenerator::generate(const core::Return &ret)
{
    if(auto builtin = ret.return_type->as_if<core::Builtin>();
        builtin && *builtin == core::Builtin::Void)
        llvm_->ir_builder->CreateRetVoid();
    else
    {
        auto val = generate(ret.val);
        llvm_->ir_builder->CreateRet(val);
    }

    auto block = llvm::BasicBlock::Create(
        *llvm_->context, "after_return", llvm_->current_function);
    llvm_->ir_builder->SetInsertPoint(block);
}

void LLVMIRGenerator::generate(const core::If &if_s)
{
    auto cond = generate(if_s.cond);
    auto then_block = llvm::BasicBlock::Create(*llvm_->context, "then");
    auto exit_block = llvm::BasicBlock::Create(*llvm_->context, "exit_if");

    llvm::BasicBlock *else_block = nullptr;
    if(if_s.else_body)
        else_block = llvm::BasicBlock::Create(*llvm_->context, "else");

    llvm_->ir_builder->CreateCondBr(
        cond, then_block, else_block ? else_block : exit_block);

    llvm_->current_function->getBasicBlockList().push_back(then_block);
    llvm_->ir_builder->SetInsertPoint(then_block);
    generate(*if_s.then_body);
    llvm_->ir_builder->CreateBr(exit_block);

    if(else_block)
    {
        llvm_->current_function->getBasicBlockList().push_back(else_block);
        llvm_->ir_builder->SetInsertPoint(else_block);
        generate(*if_s.else_body);
        llvm_->ir_builder->CreateBr(exit_block);
    }

    llvm_->current_function->getBasicBlockList().push_back(exit_block);
    llvm_->ir_builder->SetInsertPoint(exit_block);
}

void LLVMIRGenerator::generate(const core::Loop &loop)
{
    auto body_block = llvm::BasicBlock::Create(*llvm_->context, "loop");
    auto exit_block = llvm::BasicBlock::Create(*llvm_->context, "exit_loop");

    llvm_->break_dsts.push(exit_block);
    llvm_->continue_dsts.push(body_block);

    llvm_->ir_builder->CreateBr(body_block);
    llvm_->current_function->getBasicBlockList().push_back(body_block);
    llvm_->ir_builder->SetInsertPoint(body_block);

    generate(*loop.body);
    llvm_->ir_builder->CreateBr(body_block);

    llvm_->break_dsts.pop();
    llvm_->continue_dsts.pop();

    llvm_->current_function->getBasicBlockList().push_back(exit_block);
    llvm_->ir_builder->SetInsertPoint(exit_block);
}

void LLVMIRGenerator::generate(const core::Break &break_s)
{
    assert(!llvm_->break_dsts.empty());
    llvm_->ir_builder->CreateBr(llvm_->break_dsts.top());

    auto after_break = llvm::BasicBlock::Create(
        *llvm_->context, "after_break");
    llvm_->current_function->getBasicBlockList().push_back(after_break);
    llvm_->ir_builder->SetInsertPoint(after_break);
}

void LLVMIRGenerator::generate(const core::Continue &continue_s)
{
    assert(!llvm_->continue_dsts.empty());
    llvm_->ir_builder->CreateBr(llvm_->continue_dsts.top());

    auto after_continue = llvm::BasicBlock::Create(
        *llvm_->context, "after_continue");
    llvm_->current_function->getBasicBlockList().push_back(after_continue);
    llvm_->ir_builder->SetInsertPoint(after_continue);
}

void LLVMIRGenerator::generate(const core::CallFuncStat &call)
{
    generate(call.call_expr);
}

llvm::Value *LLVMIRGenerator::generate(const core::Expr &expr)
{
    return expr.match([&](auto &_e) { return generate(_e); });
}

llvm::Value *LLVMIRGenerator::generate(const core::FuncArgAddr &expr)
{
    return llvm_->arg_allocas[expr.arg_index];
}

llvm::Value *LLVMIRGenerator::generate(const core::LocalAllocAddr &expr)
{
    return llvm_->local_allocas[expr.alloc_index];
}

llvm::Value *LLVMIRGenerator::generate(const core::Load &expr)
{
    auto ptr = generate(*expr.src_addr);
    return llvm_->ir_builder->CreateLoad(ptr);
}

llvm::Value *LLVMIRGenerator::generate(const core::Immediate &expr)
{
    return expr.value.match(
        [&](auto v) -> llvm::Value *
    {
        return llvm_helper::llvm_constant_num(*llvm_->context, v);
    });
}

llvm::Value *LLVMIRGenerator::generate(const core::NullPtr &expr)
{
    assert(expr.ptr_type->is<core::Pointer>());
    return llvm::ConstantPointerNull::get(
        llvm::dyn_cast<llvm::PointerType>(get_llvm_type(expr.ptr_type)));
}

llvm::Value *LLVMIRGenerator::generate(const core::ArithmeticCast &expr)
{
    auto src_builtin_type = expr.src_type->as<core::Builtin>();
    auto dst_builtin_type = expr.dst_type->as<core::Builtin>();

    const bool is_src_int = !is_floating_point(src_builtin_type);
    const bool is_dst_int = !is_floating_point(dst_builtin_type);
    const bool is_src_signed = is_signed(src_builtin_type);
    const bool is_dst_signed = is_signed(dst_builtin_type);

    auto src_val = generate(*expr.src_val);
    if(src_builtin_type == dst_builtin_type)
        return src_val;
    auto dst_type = get_llvm_type(expr.dst_type);

    if(src_builtin_type == core::Builtin::Bool)
    {
        if(is_dst_int)
        {
            return llvm_->ir_builder->CreateSelect(
                src_val,
                llvm::ConstantInt::get(dst_type, 1, false),
                llvm::ConstantInt::get(dst_type, 0, false));
        }
        return llvm_->ir_builder->CreateSelect(
            src_val,
            llvm::ConstantFP::get(dst_type, 1),
            llvm::ConstantFP::get(dst_type, 0));
    }

    if(is_src_int && is_dst_int)
    {
        if(is_dst_signed)
            return llvm_->ir_builder->CreateSExtOrTrunc(src_val, dst_type);
        return llvm_->ir_builder->CreateZExtOrTrunc(src_val, dst_type);
    }

    if(is_src_int && !is_dst_int)
    {
        if(is_src_signed)
            return llvm_->ir_builder->CreateSIToFP(src_val, dst_type);
        return llvm_->ir_builder->CreateUIToFP(src_val, dst_type);
    }

    if(!is_src_int && is_dst_int)
    {
        if(is_dst_signed)
            return llvm_->ir_builder->CreateFPToSI(src_val, dst_type);
        return llvm_->ir_builder->CreateFPToUI(src_val, dst_type);
    }

    if(!is_src_int && !is_dst_int)
        return llvm_->ir_builder->CreateFPCast(src_val, dst_type);

    unreachable();
}

llvm::Value *LLVMIRGenerator::generate(const core::PointerOffset &expr)
{
    auto ptr = generate(*expr.ptr_val);
    auto offset = generate(*expr.offset_val);
    if(expr.negative)
        offset = llvm_->ir_builder->CreateNeg(offset);
    llvm::Value *indices[1] = { offset };
    return llvm_->ir_builder->CreateGEP(ptr, indices);
}

llvm::Value *LLVMIRGenerator::generate(const core::ClassPointerToMemberPointer &expr)
{
    auto class_ptr = generate(*expr.class_ptr);
    std::array<llvm::Value *, 2> indices;
    indices[0] = llvm_helper::llvm_constant_num(*llvm_->context, uint32_t(0));
    indices[1] = llvm_helper::llvm_constant_num(*llvm_->context, uint32_t(expr.member_index));
    return llvm_->ir_builder->CreateGEP(class_ptr, indices);
}

llvm::Value *LLVMIRGenerator::generate(const core::DerefClassPointer &expr)
{
    auto class_ptr = generate(*expr.class_ptr);
    return llvm_->ir_builder->CreateLoad(class_ptr);
}

llvm::Value *LLVMIRGenerator::generate(const core::DerefArrayPointer &expr)
{
    auto array_ptr = generate(*expr.array_ptr);
    return llvm_->ir_builder->CreateLoad(array_ptr);
}

llvm::Value *LLVMIRGenerator::generate(const core::SaveClassIntoLocalAlloc &expr)
{
    auto class_val = generate(*expr.class_val);
    auto alloc = llvm_->ir_builder->CreateAlloca(class_val->getType());
    llvm_->ir_builder->CreateStore(class_val, alloc);
    return alloc;
}

llvm::Value *LLVMIRGenerator::generate(const core::SaveArrayIntoLocalAlloc &expr)
{
    auto array_val = generate(*expr.array_val);
    auto alloc = llvm_->ir_builder->CreateAlloca(array_val->getType());
    llvm_->ir_builder->CreateStore(array_val, alloc);
    return alloc;
}

llvm::Value *LLVMIRGenerator::generate(const core::ArrayAddrToFirstElemAddr &expr)
{
    auto arr = generate(*expr.array_ptr);
    std::array<llvm::Value *, 2> indices;
    indices[0] = llvm_helper::llvm_constant_num(*llvm_->context, uint32_t(0));
    indices[1] = llvm_helper::llvm_constant_num(*llvm_->context, uint32_t(0));
    return llvm_->ir_builder->CreateGEP(arr, indices);
}

llvm::Value *LLVMIRGenerator::generate(const core::Binary &expr)
{
    auto lhs = generate(*expr.lhs);
    auto rhs = generate(*expr.rhs);

    auto lhs_type = expr.lhs_type->as<core::Builtin>();
    auto rhs_type = expr.rhs_type->as<core::Builtin>();
    assert(lhs_type == rhs_type);

    switch(expr.op)
    {
    case core::Binary::Op::Add:
    {
        assert(lhs_type != core::Builtin::Bool);
        if(lhs->getType()->isFloatingPointTy())
            return llvm_->ir_builder->CreateFAdd(lhs, rhs);
        return llvm_->ir_builder->CreateAdd(lhs, rhs);
    }
    case core::Binary::Op::Sub:
    {
        assert(lhs_type != core::Builtin::Bool);
        if(lhs->getType()->isFloatingPointTy())
            return llvm_->ir_builder->CreateFSub(lhs, rhs);
        return llvm_->ir_builder->CreateSub(lhs, rhs);
    }
    case core::Binary::Op::Mul:
    {
        assert(lhs_type != core::Builtin::Bool);
        if(lhs->getType()->isFloatingPointTy())
            return llvm_->ir_builder->CreateFMul(lhs, rhs);
        return llvm_->ir_builder->CreateMul(lhs, rhs);
    }
    case core::Binary::Op::Div:
    {
        assert(lhs_type != core::Builtin::Bool);
        if(is_floating_point(lhs_type))
            return llvm_->ir_builder->CreateFDiv(lhs, rhs);
        if(is_signed(lhs_type))
            return llvm_->ir_builder->CreateSDiv(lhs, rhs);
        return llvm_->ir_builder->CreateUDiv(lhs, rhs);
    }
    case core::Binary::Op::Mod:
    {
        assert(!is_floating_point(lhs_type));
        if(is_signed(lhs_type))
            return llvm_->ir_builder->CreateSRem(lhs, rhs);
        return llvm_->ir_builder->CreateURem(lhs, rhs);
    }
    case core::Binary::Op::Equal:
    {
        if(lhs->getType()->isFloatingPointTy())
            return llvm_->ir_builder->CreateFCmpOEQ(lhs, rhs);
        return llvm_->ir_builder->CreateICmpEQ(lhs, rhs);
    }
    case core::Binary::Op::NotEqual:
    {
        if(lhs->getType()->isFloatingPointTy())
            return llvm_->ir_builder->CreateFCmpONE(lhs, rhs);
        return llvm_->ir_builder->CreateICmpNE(lhs, rhs);
    }
    case core::Binary::Op::Less:
    {
        if(lhs->getType()->isFloatingPointTy())
            return llvm_->ir_builder->CreateFCmpOLT(lhs, rhs);
        if(is_signed(lhs_type))
            return llvm_->ir_builder->CreateICmpSLT(lhs, rhs);
        return llvm_->ir_builder->CreateICmpULT(lhs, rhs);
    }
    case core::Binary::Op::LessEqual:
    {
        if(lhs->getType()->isFloatingPointTy())
            return llvm_->ir_builder->CreateFCmpOLE(lhs, rhs);
        if(is_signed(lhs_type))
            return llvm_->ir_builder->CreateICmpSLE(lhs, rhs);
        return llvm_->ir_builder->CreateICmpULE(lhs, rhs);
    }
    case core::Binary::Op::Greater:
    {
        if(lhs->getType()->isFloatingPointTy())
            return llvm_->ir_builder->CreateFCmpOGT(lhs, rhs);
        if(is_signed(lhs_type))
            return llvm_->ir_builder->CreateICmpSGT(lhs, rhs);
        return llvm_->ir_builder->CreateICmpUGT(lhs, rhs);
    }
    case core::Binary::Op::GreaterEqual:
    {
        if(lhs->getType()->isFloatingPointTy())
            return llvm_->ir_builder->CreateFCmpOGE(lhs, rhs);
        if(is_signed(lhs_type))
            return llvm_->ir_builder->CreateICmpSGE(lhs, rhs);
        return llvm_->ir_builder->CreateICmpUGE(lhs, rhs);
    }
    case core::Binary::Op::LeftShift:
    {
        assert(!is_floating_point(lhs_type));
        return llvm_->ir_builder->CreateShl(lhs, rhs);
    }
    case core::Binary::Op::RightShift:
    {
        assert(!is_signed(lhs_type));
        assert(!is_floating_point(lhs_type));
        return llvm_->ir_builder->CreateLShr(lhs, rhs);
    }
    case core::Binary::Op::BitwiseAnd:
    {
        assert(!is_floating_point(lhs_type));
        return llvm_->ir_builder->CreateAnd(lhs, rhs);
    }
    case core::Binary::Op::BitwiseOr:
    {
        assert(!is_floating_point(lhs_type));
        return llvm_->ir_builder->CreateOr(lhs, rhs);
    }
    case core::Binary::Op::BitwiseXOr:
    {
        assert(!is_floating_point(lhs_type));
        return llvm_->ir_builder->CreateXor(lhs, rhs);
    }
    }

    unreachable();
}

llvm::Value *LLVMIRGenerator::generate(const core::Unary &expr)
{
    auto val = generate(*expr.val);
    auto val_type = expr.val_type->as<core::Builtin>();

    switch(expr.op)
    {
    case core::Unary::Op::Neg:
    {
        assert(val_type != core::Builtin::Bool);
        if(is_floating_point(val_type))
            return llvm_->ir_builder->CreateFNeg(val);
        return llvm_->ir_builder->CreateNeg(val);
    }
    case core::Unary::Op::Not:
    {
        assert(val_type == core::Builtin::Bool);
        return llvm_->ir_builder->CreateNot(val);
    }
    case core::Unary::Op::BitwiseNot:
    {
        assert(!is_floating_point(val_type));
        return llvm_->ir_builder->CreateNot(val);
    }
    }

    unreachable();
}

llvm::Value *LLVMIRGenerator::generate(const core::CallFunc &expr)
{
    std::vector<llvm::Value *> args;
    for(auto &a : expr.args)
        args.push_back(generate(*a));

    if(expr.contextless_func)
    {
        auto func =
            llvm_->llvm_functions_.at(expr.contextless_func.get()).llvm_function;
        return llvm_->ir_builder->CreateCall(func, args);
    }
    
    if(expr.intrinsic != core::Intrinsic::None)
        return process_intrinsic_call(expr, args);

    auto core_func = llvm_->prog.funcs[expr.contexted_func_index].get();
    auto func = llvm_->llvm_functions_.at(core_func).llvm_function;
    return llvm_->ir_builder->CreateCall(func, args);
}

llvm::Value *LLVMIRGenerator::process_intrinsic_call(
    const core::CallFunc &call, const std::vector<llvm::Value*> &args)
{
    if(target_ == Target::Native)
    {
        return process_native_intrinsics(
            *llvm_->top_module, *llvm_->ir_builder, call.intrinsic, args);
    }
    assert(target_ == Target::PTX);
    return process_ptx_intrinsics(
        *llvm_->top_module, *llvm_->ir_builder,
        call.intrinsic, args, approx_math_func_);
}

CUJ_NAMESPACE_END(cuj::gen)

#ifdef _MSC_VER
#pragma warning(pop)
#endif
