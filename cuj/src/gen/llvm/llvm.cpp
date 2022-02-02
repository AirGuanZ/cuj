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
#include <llvm/IR/InlineAsm.h>
#include <llvm/IR/IntrinsicsNVPTX.h>
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
#include "libdevice_man.h"
#include "native_intrinsics.h"
#include "ptx_intrinsics.h"
#include "type_manager.h"
#include "vector_intrinsic.h"

CUJ_NAMESPACE_BEGIN(cuj::gen)

struct LLVMIRGenerator::LLVMData
{
    // per module

    core::Prog             prog;
    Box<llvm::LLVMContext> context;

    Box<llvm::IRBuilder<>> ir_builder;
    Box<llvm::Module>      top_module;

    struct FunctionRecord
    {
        llvm::Function *llvm_function;
    };

    std::map<const core::Func *, FunctionRecord> llvm_functions_;

    std::map<const core::GlobalVar *, llvm::GlobalVariable *> global_vars_;

    std::map<std::vector<unsigned char>, llvm::GlobalVariable *> global_const_vars_;

    llvm_helper::TypeManager type_manager;

    // per function

    llvm::Function *current_function = nullptr;

    std::vector<llvm::AllocaInst *> local_allocas;
    std::vector<llvm::AllocaInst *> arg_allocas;

    std::stack<llvm::BasicBlock *> break_dsts;
    std::stack<llvm::BasicBlock *> continue_dsts;
    std::stack<llvm::BasicBlock *> scope_exits;
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

void LLVMIRGenerator::disable_assert()
{
    enable_assert_ = false;
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
        libdev::link_with_libdevice(*llvm_->top_module);

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
        std::map<const core::Type *, std::type_index> types;
        auto handle_type_set = [&](const core::TypeSet &set)
        {
            for(auto &[type, index] : set.type_to_index)
                types.try_emplace(type, index);
        };

        handle_type_set(*prog.global_type_set);
        for(auto &func : prog.funcs)
        {
            if(func->type_set)
                handle_type_set(*func->type_set);
        }

        llvm_->type_manager.initialize(
            llvm_->context.get(), data_layout_, std::move(types));
    }

    // generate global variables

    generate_global_variables();

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

void LLVMIRGenerator::generate_global_variables()
{
    for(auto &pv : llvm_->prog.global_vars)
    {
        auto &var = *pv;

        // address space

        unsigned int address_space;
        if(var.memory_type == core::GlobalVar::MemoryType::Regular)
        {
            if(target_ == Target::Native)
                address_space = 0;
            else
            {
                assert(target_ == Target::PTX);
                address_space = 1;
            }
        }
        else
        {
            if(target_ == Target::Native)
                address_space = 0;
            else
            {
                assert(target_ == Target::PTX);
                address_space = 4;
            }
        }

        // allocate

        auto llvm_type = llvm_->type_manager.get_llvm_type(var.type);
        auto llvm_global_var = new llvm::GlobalVariable(
            *llvm_->top_module, llvm_type, false,
            llvm::GlobalValue::ExternalLinkage, nullptr,
            var.symbol_name, nullptr,
            llvm::GlobalValue::NotThreadLocal, address_space);
        if(const size_t align = llvm_->type_manager.get_custom_alignment(var.type))
        {
            llvm_global_var->setAlignment(llvm::Align(align));
        }
        llvm_global_var->setInitializer(llvm::Constant::getNullValue(llvm_type));

        llvm_->global_vars_.insert({ pv.get(), llvm_global_var });
    }
}

llvm::FunctionType *LLVMIRGenerator::get_function_type(const core::Func &func)
{
    if(func.type == core::Func::Kernel)
    {
        auto ret_type = func.return_type.type->as_if<core::Builtin>();
        if(!ret_type || *ret_type != core::Builtin::Void)
            throw CujException("kernel function must return void");
    }

    llvm::Type *ret_type = llvm_->type_manager.get_llvm_type(func.return_type.type);
    if(func.return_type.is_reference)
        ret_type = llvm::PointerType::get(ret_type, 0);

    std::vector<llvm::Type *> arg_types;
    for(auto &arg : func.argument_types)
    {
        llvm::Type *arg_type = llvm_->type_manager.get_llvm_type(arg.type);
        if(arg.is_reference)
            arg_type = llvm::PointerType::get(arg_type, 0);
        arg_types.push_back(arg_type);
    }

    return llvm::FunctionType::get(ret_type, arg_types, false);
}

void LLVMIRGenerator::declare_function(const core::Func *func)
{
    std::string symbol_name = func->name;
    assert(!symbol_name.empty());

    if(auto existing_llvm_func = llvm_->top_module->getFunction(symbol_name))
    {
        if(func->is_declaration)
        {
            if(existing_llvm_func->getFunctionType() != get_function_type(*func))
            {
                throw CujException(
                    "multiple function declaration with different signatures: " + symbol_name);
            }
            return;
        }
        throw CujException(
            "multiple definitions of function " + symbol_name);
    }

    auto func_type = get_function_type(*func);
    llvm::Function *llvm_func;
    if(func->is_declaration)
    {
        llvm_func = llvm::cast<llvm::Function>(llvm_->top_module->getOrInsertFunction(
            symbol_name, func_type).getCallee());
    }
    else
    {
        llvm_func = llvm::Function::Create(
            func_type, llvm::GlobalValue::ExternalLinkage,
            symbol_name, llvm_->top_module.get());
    }

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
    if(func->is_declaration)
        return;

    clear_temp_function_data();
    llvm_->current_function = llvm_->top_module->getFunction(func->name);
    CUJ_SCOPE_EXIT{ llvm_->current_function = nullptr; };

    auto entry_block = llvm::BasicBlock::Create(
        *llvm_->context, "entry", llvm_->current_function);
    llvm_->ir_builder->SetInsertPoint(entry_block);

    generate_local_allocs(func);

    generate(*func->root_block);

    generate_default_ret(func);

#if defined(DEBUG) || defined(_DEBUG)
    std::string err_msg;
    llvm::raw_string_ostream err_stream(err_msg);
    if(verifyFunction(*llvm_->current_function, &err_stream))
        throw CujException(err_msg);
#endif
}

void LLVMIRGenerator::clear_temp_function_data()
{
    llvm_->current_function = nullptr;
    llvm_->local_allocas.clear();
    llvm_->arg_allocas.clear();
    llvm_->break_dsts = {};
    llvm_->continue_dsts = {};
    llvm_->scope_exits = {};
}

void LLVMIRGenerator::generate_local_allocs(const core::Func *func)
{
    constexpr int LOCAL_ALLOCA_ADDRESS_SPACE = 0;

    for(size_t i = 0; i < func->local_alloc_types.size(); ++i)
    {
        auto type = func->local_alloc_types[i];
        auto llvm_type = llvm_->type_manager.get_llvm_type(type);
        auto alloca_inst = llvm_->ir_builder->CreateAlloca(
            llvm_type, LOCAL_ALLOCA_ADDRESS_SPACE,
            nullptr, "var" + std::to_string(i));
        if(const size_t align = llvm_->type_manager.get_custom_alignment(type))
            alloca_inst->setAlignment(llvm::Align(align));
        llvm_->local_allocas.push_back(alloca_inst);
    }

    for(size_t i = 0; i < func->argument_types.size(); ++i)
    {
        auto arg = llvm_->current_function->getArg(static_cast<unsigned>(i));

        auto alloca_inst = llvm_->ir_builder->CreateAlloca(
            arg->getType(), LOCAL_ALLOCA_ADDRESS_SPACE, nullptr);
        if(const size_t align = llvm_->type_manager.get_custom_alignment(
                func->local_alloc_types[i]))
            alloca_inst->setAlignment(llvm::Align(align));

        llvm_->arg_allocas.push_back(alloca_inst);
        llvm_->ir_builder->CreateStore(arg, alloca_inst);
    }
}

void LLVMIRGenerator::generate_default_ret(const core::Func *func)
{
    if(func->return_type.is_reference)
    {
        auto ret_type = llvm_->type_manager.get_llvm_type(func->return_type.type);
        llvm_->ir_builder->CreateRet(
            llvm::ConstantPointerNull::get(
                llvm::PointerType::get(ret_type, 0)));
    }
    else
    {
        auto llvm_type = llvm_->type_manager.get_llvm_type(func->return_type.type);
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

void LLVMIRGenerator::generate(const core::Copy &copy)
{
    /*if(data_layout_)
    {
        auto src_addr = generate(copy.src_addr);
        auto dst_addr = generate(copy.dst_addr);
        auto src_elem_type = llvm::dyn_cast<llvm::PointerType>(
            src_addr->getType())->getElementType();
        const auto size = data_layout_->getTypeStoreSize(src_elem_type);
        auto src_align = data_layout_->getABITypeAlign(src_elem_type);
        llvm_->ir_builder->CreateMemCpy(
            dst_addr, src_align, src_addr, src_align, size.getFixedSize());
    }
    else*/
    {
        auto src_addr = generate(copy.src_addr);
        auto dst_addr = generate(copy.dst_addr);
        auto src_val = llvm_->ir_builder->CreateLoad(src_addr);
        llvm_->ir_builder->CreateStore(src_val, dst_addr);
    }
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
    auto then_block = llvm::BasicBlock::Create(*llvm_->context, "then");
    auto exit_block = llvm::BasicBlock::Create(*llvm_->context, "exit_if");

    llvm::BasicBlock *else_block = nullptr;
    if(if_s.else_body)
        else_block = llvm::BasicBlock::Create(*llvm_->context, "else");

    generate(*if_s.calc_cond);
    auto cond = generate(if_s.cond);
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

void LLVMIRGenerator::generate(const core::Switch &switch_s)
{
    auto end_block = llvm::BasicBlock::Create(*llvm_->context, "exit_switch");

    auto default_body_block = end_block;
    if(switch_s.default_body)
    {
        default_body_block =
            llvm::BasicBlock::Create(*llvm_->context, "default");
    }

    std::vector<llvm::BasicBlock *> case_body_blocks;
    case_body_blocks.reserve(switch_s.branches.size());
    for(auto &_ : switch_s.branches)
    {
        case_body_blocks.push_back(
            llvm::BasicBlock::Create(*llvm_->context, "case"));
    }

    auto function = llvm_->current_function;
    for(auto c : case_body_blocks)
        function->getBasicBlockList().push_back(c);
    function->getBasicBlockList().push_back(default_body_block);
    if(end_block != default_body_block)
        function->getBasicBlockList().push_back(end_block);

    auto value = generate(switch_s.value);
    if(!value->getType()->isIntegerTy())
        throw CujException("switch statement requires an integer value");

    auto inst = llvm_->ir_builder->CreateSwitch(value, default_body_block);
    size_t case_index = 0;
    for(auto &b : switch_s.branches)
    {
        auto body_block = case_body_blocks[case_index++];
        auto cond = b.cond.value.match(
            [&]<typename T>(T v)
        {
            if(!std::is_integral_v<T>)
                throw CujException("switch statement requires an integer cond");
            auto val = llvm_helper::llvm_constant_num(*llvm_->context, v);
            auto cval = llvm::dyn_cast<llvm::ConstantInt>(val);
            assert(cval);
            return cval;
        });

        assert(cond->getType() == value->getType());
        inst->addCase(cond, body_block);

        llvm_->ir_builder->SetInsertPoint(body_block);
        generate(*b.body);

        llvm::BasicBlock *case_end = end_block;
        if(b.fallthrough)
        {
            case_end = case_index < case_body_blocks.size() ?
                case_body_blocks[case_index] : default_body_block;
        }
        llvm_->ir_builder->CreateBr(case_end);
    }

    if(default_body_block != end_block)
    {
        llvm_->ir_builder->SetInsertPoint(default_body_block);
        generate(*switch_s.default_body);
        llvm_->ir_builder->CreateBr(end_block);
    }
    llvm_->ir_builder->SetInsertPoint(end_block);
}

void LLVMIRGenerator::generate(const core::CallFuncStat &call)
{
    generate(call.call_expr);
}

void LLVMIRGenerator::generate(const core::MakeScope &make_scope)
{
    auto exit_block = llvm::BasicBlock::Create(*llvm_->context, "exit_scope");

    llvm_->scope_exits.push(exit_block);
    generate(*make_scope.body);
    llvm_->scope_exits.pop();

    llvm_->ir_builder->CreateBr(exit_block);
    llvm_->current_function->getBasicBlockList().push_back(exit_block);
    llvm_->ir_builder->SetInsertPoint(exit_block);
}

void LLVMIRGenerator::generate(const core::ExitScope &exit_scope)
{
    assert(!llvm_->scope_exits.empty());
    llvm_->ir_builder->CreateBr(llvm_->scope_exits.top());

    auto after_exit = llvm::BasicBlock::Create(
        *llvm_->context, "after_exit_scope");
    llvm_->current_function->getBasicBlockList().push_back(after_exit);
    llvm_->ir_builder->SetInsertPoint(after_exit);
}

void LLVMIRGenerator::generate(const core::InlineAsm &inline_asm)
{
    std::vector<llvm::Value *> input_values;
    for(auto &iv : inline_asm.input_values)
        input_values.push_back(generate(iv));

    std::vector<llvm::Value *> output_addresses;
    for(auto &oa : inline_asm.output_addresses)
        output_addresses.push_back(generate(oa));

    std::vector<llvm::Type *> llvm_output_types;
    for(auto oa : output_addresses)
    {
        auto oat = llvm::dyn_cast<llvm::PointerType>(oa->getType());
        llvm_output_types.push_back(oat->getElementType());
    }
    llvm::Type *llvm_output_type;
    if(llvm_output_types.size() == 0)
        llvm_output_type = llvm_->ir_builder->getVoidTy();
    else if(llvm_output_types.size() == 1)
        llvm_output_type = llvm_output_types[0];
    else
    {
        llvm_output_type =
            llvm::StructType::get(*llvm_->context, llvm_output_types);
    }

    std::vector<llvm::Type *> llvm_input_types;
    for(auto iv : input_values)
        llvm_input_types.push_back(iv->getType());

    auto asm_func_type =
        llvm::FunctionType::get(llvm_output_type, llvm_input_types, false);

    std::string constraints = inline_asm.output_constraints;
    if(!constraints.empty() && !inline_asm.input_constraints.empty())
        constraints += ",";
    constraints += inline_asm.input_constraints;
    if(!constraints.empty() && !inline_asm.clobber_constraints.empty())
        constraints += ",";
    constraints += inline_asm.clobber_constraints;
    auto asm_callee = llvm::InlineAsm::get(
        asm_func_type, inline_asm.asm_string,
        constraints, inline_asm.side_effects);

    auto output_value = llvm_->ir_builder->CreateCall(asm_callee, input_values);
    if(output_addresses.size() == 1)
    {
        auto ptr = output_addresses[0];
        llvm_->ir_builder->CreateStore(output_value, ptr);
    }
    else
    {
        for(size_t i = 0; i < output_addresses.size(); ++i)
        {
            unsigned idx = static_cast<unsigned int>(i);
            auto val = llvm_->ir_builder->CreateExtractValue(output_value, idx);
            auto ptr = output_addresses[i];
            llvm_->ir_builder->CreateStore(val, ptr);
        }
    }
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
        llvm::dyn_cast<llvm::PointerType>(
            llvm_->type_manager.get_llvm_type(expr.ptr_type)));
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
    auto dst_type = llvm_->type_manager.get_llvm_type(expr.dst_type);

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

llvm::Value *LLVMIRGenerator::generate(const core::BitwiseCast &expr)
{
    auto src_type = llvm_->type_manager.get_llvm_type(expr.src_type);
    auto dst_type = llvm_->type_manager.get_llvm_type(expr.dst_type);
    auto src_val = generate(*expr.src_val);

    if(src_type->isPointerTy())
    {
        if(dst_type->isPointerTy()) // ptr to ptr
            return llvm_->ir_builder->CreatePointerCast(src_val, dst_type);

        // ptr to integer
        return llvm_->ir_builder->CreatePtrToInt(src_val, dst_type);
    }

    if(dst_type->isPointerTy()) // integer to ptr
        return llvm_->ir_builder->CreateIntToPtr(src_val, dst_type);

    // arithmetic to arithmetic
    return llvm_->ir_builder->CreateBitCast(src_val, dst_type);
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
    const int member_index = llvm_->type_manager.get_struct_member_index(
        expr.class_ptr_type->as<core::Pointer>().pointed, expr.member_index);
    std::array<llvm::Value *, 2> indices;
    indices[0] = llvm_helper::llvm_constant_num(*llvm_->context, uint32_t(0));
    indices[1] = llvm_helper::llvm_constant_num(*llvm_->context, uint32_t(member_index));
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
        auto func = llvm_->top_module->getFunction(expr.contextless_func->name);
        return llvm_->ir_builder->CreateCall(func, args);
    }
    
    if(expr.intrinsic != core::Intrinsic::None)
        return process_intrinsic_call(expr, args);

    auto core_func = llvm_->prog.funcs[expr.contexted_func_index].get();
    auto func = llvm_->llvm_functions_.at(core_func).llvm_function;
    return llvm_->ir_builder->CreateCall(func, args);
}

llvm::Value *LLVMIRGenerator::generate(const core::GlobalVarAddr &expr)
{
    llvm::Value *ptr = llvm_->global_vars_.at(expr.var.get());
    if(target_ == Target::PTX)
    {
        auto var_type = llvm_->type_manager.get_llvm_type(expr.var->type);
        auto dst_type = llvm::PointerType::get(var_type, 0);
        if(expr.var->memory_type == core::GlobalVar::MemoryType::Regular)
        {
            auto src_type = llvm::PointerType::get(var_type, 1);
            ptr = llvm_->ir_builder->CreateIntrinsic(
                llvm::Intrinsic::nvvm_ptr_global_to_gen,
                { dst_type, src_type }, ptr);
        }
        else
        {
            auto src_type = llvm::PointerType::get(var_type, 4);
            ptr = llvm_->ir_builder->CreateIntrinsic(
                llvm::Intrinsic::nvvm_ptr_constant_to_gen,
                { dst_type, src_type }, ptr);
        }
    }
    return ptr;
}

llvm::Value *LLVMIRGenerator::generate(const core::GlobalConstAddr &expr)
{
    const int GLOBAL_ADDR_SPACE = target_ == Target::PTX ? 1 : 0;

    auto llvm_u8   = llvm_->ir_builder->getInt8Ty();
    auto llvm_elem = llvm_->type_manager.get_llvm_type(expr.pointed_type);

    llvm::GlobalVariable *global_var;
    if(auto it = llvm_->global_const_vars_.find(expr.data);
       it != llvm_->global_const_vars_.end())
    {
        global_var = it->second;
        size_t alignment = global_var->getAlign().valueOrOne().value();
        alignment = (std::max)(alignment, expr.alignment);
        alignment = (std::max)(
            alignment, llvm_->type_manager.get_custom_alignment(expr.pointed_type));
        global_var->setAlignment(llvm::Align(alignment));
    }
    else
    {
        auto arr_type = llvm::ArrayType::get(llvm_u8, expr.data.size());

        std::vector<llvm::Constant *> byte_consts;
        byte_consts.reserve(expr.data.size());
        for(auto b : expr.data)
            byte_consts.push_back(llvm::ConstantInt::get(llvm_u8, b, false));
        auto const_init = llvm::ConstantArray::get(arr_type, byte_consts);

        global_var = new llvm::GlobalVariable(
            *llvm_->top_module, arr_type, true,
            llvm::GlobalValue::InternalLinkage, const_init,
            "", nullptr, llvm::GlobalValue::NotThreadLocal,
            GLOBAL_ADDR_SPACE);

        size_t alignment = expr.alignment;
        alignment = (std::max)(
            alignment, llvm_->type_manager.get_custom_alignment(expr.pointed_type));
        if(alignment)
            global_var->setAlignment(llvm::Align(alignment));

        llvm_->global_const_vars_.insert({ expr.data, global_var });
    }

    std::array<llvm::Value *, 2> indices = {
        llvm_helper::llvm_constant_num(*llvm_->context, 0u),
        llvm_helper::llvm_constant_num(*llvm_->context, 0u)
    };
    auto val = llvm_->ir_builder->CreateGEP(global_var, indices);

    if(target_ == Target::PTX)
    {
        auto src_type = llvm::PointerType::get(llvm_u8, GLOBAL_ADDR_SPACE);
        auto dst_type = llvm::PointerType::get(llvm_u8, 0);
        val = llvm_->ir_builder->CreateIntrinsic(
            llvm::Intrinsic::nvvm_ptr_global_to_gen,
            { dst_type, src_type }, { val });
    }

    if(llvm_elem != llvm_u8)
    {
        val = llvm_->ir_builder->CreatePointerCast(
            val, llvm::PointerType::get(llvm_elem, 0));
    }

    return val;
}

llvm::Value *LLVMIRGenerator::process_intrinsic_call(
    const core::CallFunc &call, const std::vector<llvm::Value*> &args)
{
    if(call.intrinsic == core::Intrinsic::store_f32x4 ||
       call.intrinsic == core::Intrinsic::store_f32x3 ||
       call.intrinsic == core::Intrinsic::store_f32x2 ||
       call.intrinsic == core::Intrinsic::store_u32x4 ||
       call.intrinsic == core::Intrinsic::store_u32x3 ||
       call.intrinsic == core::Intrinsic::store_u32x2 ||
       call.intrinsic == core::Intrinsic::store_i32x4 ||
       call.intrinsic == core::Intrinsic::store_i32x3 ||
       call.intrinsic == core::Intrinsic::store_i32x2)
    {
        return detail::create_vector_store(
            *llvm_->ir_builder, args);
    }
    
    if(call.intrinsic == core::Intrinsic::load_f32x4 ||
       call.intrinsic == core::Intrinsic::load_f32x3 ||
       call.intrinsic == core::Intrinsic::load_f32x2 ||
       call.intrinsic == core::Intrinsic::load_u32x4 ||
       call.intrinsic == core::Intrinsic::load_u32x3 ||
       call.intrinsic == core::Intrinsic::load_u32x2 ||
       call.intrinsic == core::Intrinsic::load_i32x4 ||
       call.intrinsic == core::Intrinsic::load_i32x3 ||
       call.intrinsic == core::Intrinsic::load_i32x2)
    {
        detail::create_vector_load(
            *llvm_->ir_builder, args);
        return nullptr;
    }

    if(call.intrinsic == core::Intrinsic::atomic_add_f32)
    {
        return llvm_->ir_builder->CreateAtomicRMW(
            llvm::AtomicRMWInst::FAdd, args[0], args[1],
            llvm::AtomicOrdering::SequentiallyConsistent);
    }

    if(call.intrinsic == core::Intrinsic::atomic_add_i32 ||
       call.intrinsic == core::Intrinsic::atomic_add_u32)
    {
        return llvm_->ir_builder->CreateAtomicRMW(
            llvm::AtomicRMWInst::Add, args[0], args[1],
            llvm::AtomicOrdering::SequentiallyConsistent);
    }

    if(call.intrinsic == core::Intrinsic::f32_min ||
       call.intrinsic == core::Intrinsic::f64_min)
    {
        auto comp = llvm_->ir_builder->CreateFCmpOLT(args[0], args[1]);
        return llvm_->ir_builder->CreateSelect(comp, args[0], args[1]);
    }

    if(call.intrinsic == core::Intrinsic::f32_max ||
       call.intrinsic == core::Intrinsic::f64_max)
    {
        auto comp = llvm_->ir_builder->CreateFCmpOGT(args[0], args[1]);
        return llvm_->ir_builder->CreateSelect(comp, args[0], args[1]);
    }

    if(call.intrinsic == core::Intrinsic::i32_min ||
       call.intrinsic == core::Intrinsic::i64_min)
    {
        auto comp = llvm_->ir_builder->CreateICmpSLT(args[0], args[1]);
        return llvm_->ir_builder->CreateSelect(comp, args[0], args[1]);
    }

    if(call.intrinsic == core::Intrinsic::i32_max ||
       call.intrinsic == core::Intrinsic::i64_max)
    {
        auto comp = llvm_->ir_builder->CreateICmpSGT(args[0], args[1]);
        return llvm_->ir_builder->CreateSelect(comp, args[0], args[1]);
    }

    if(call.intrinsic == core::Intrinsic::assert_fail && !enable_assert_)
        return nullptr;

    if(call.intrinsic == core::Intrinsic::unreachable)
    {
        auto ret = llvm_->ir_builder->CreateUnreachable();
        auto after_block =
            llvm::BasicBlock::Create(*llvm_->context, "after_unreachable");
        llvm_->current_function->getBasicBlockList().push_back(after_block);
        llvm_->ir_builder->SetInsertPoint(after_block);
        return ret;
    }

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
