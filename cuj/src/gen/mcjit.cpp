#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable: 4141)
#pragma warning(disable: 4244)
#pragma warning(disable: 4624)
#pragma warning(disable: 4626)
#pragma warning(disable: 4996)
#endif

#include <cmath>

#include <llvm/Analysis/TargetTransformInfo.h>
#include <llvm/ExecutionEngine/ExecutionEngine.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/Support/Host.h>
#include <llvm/Support/TargetRegistry.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Target/TargetMachine.h>
#include <llvm/Target/TargetOptions.h>
#include <llvm/Transforms/IPO/PassManagerBuilder.h>
#include <llvm/Transforms/IPO.h>

#include <cuj/gen/llvm.h>
#include <cuj/gen/mcjit.h>

#include "llvm/helper.h"

CUJ_NAMESPACE_BEGIN(cuj::gen)

namespace
{

    struct LLVMModuleData
    {
        Box<llvm::LLVMContext>  llvm_context;
        Box<llvm::Module>       llvm_module;
        llvm::TargetMachine    *machine;
        llvm::CodeGenOpt::Level codegen_opt;
    };

    llvm::TargetMachine *get_native_target_machine(
        llvm::CodeGenOpt::Level codegen_opt)
    {
        auto target_triple = llvm::sys::getDefaultTargetTriple();

        std::string err;
        auto target = llvm::TargetRegistry::lookupTarget(target_triple, err);
        if(!target)
            throw CujException(err);

        return target->createTargetMachine(
            target_triple, "generic", {},
            {}, {}, {}, codegen_opt, true);
    }

    void do_llvm_optimize(
        llvm::Module *mod, llvm::TargetMachine *tm, const Options &opts)
    {
        llvm::PassManagerBuilder pass_mgr_builder;
        pass_mgr_builder.OptLevel = static_cast<int>(opts.opt_level);
        pass_mgr_builder.Inliner = llvm::createFunctionInliningPass(
            pass_mgr_builder.OptLevel, 0, false);
        pass_mgr_builder.SLPVectorize = true;
        pass_mgr_builder.LoopVectorize = true;
        if(opts.opt_level == OptimizationLevel::O2 ||
           opts.opt_level == OptimizationLevel::O3)
            pass_mgr_builder.MergeFunctions = true;
        else
            pass_mgr_builder.MergeFunctions = false;
        tm->adjustPassManager(pass_mgr_builder);

        llvm::legacy::FunctionPassManager fp_mgr(mod);
        pass_mgr_builder.populateFunctionPassManager(fp_mgr);
        fp_mgr.doInitialization();
        for(auto &f : mod->functions())
            fp_mgr.run(f);
        fp_mgr.doFinalization();

        llvm::legacy::PassManager passes;
        passes.add(
            createTargetTransformInfoWrapperPass(tm->getTargetIRAnalysis()));
        pass_mgr_builder.populateModulePassManager(passes);
        passes.run(*mod);
    }

    LLVMModuleData build_llvm_module(
        const dsl::Module &mod, const Options &opts)
    {
        std::once_flag init_mcjit;
        std::call_once(init_mcjit, [] 
        {
            llvm::InitializeNativeTarget();
            llvm::InitializeNativeTargetAsmPrinter();
            LLVMLinkInMCJIT();
        });

        const llvm::CodeGenOpt::Level codegen_opt =
            llvm_helper::get_codegen_opt_level(opts.opt_level);
        auto target_machine = get_native_target_machine(codegen_opt);
        auto data_layout = target_machine->createDataLayout();

        LLVMIRGenerator llvm_ir_gen;
        llvm_ir_gen.disable_basic_optimizations();
        llvm_ir_gen.set_target(LLVMIRGenerator::Target::Native);
        if(opts.fast_math)
            llvm_ir_gen.use_fast_math();
        if(opts.approx_math_func)
            llvm_ir_gen.use_approx_math_func();
        llvm_ir_gen.set_data_layout(&data_layout);
        llvm_ir_gen.generate(mod);

        do_llvm_optimize(llvm_ir_gen.get_llvm_module(), target_machine, opts);

        LLVMModuleData ret;
        std::tie(ret.llvm_context, ret.llvm_module) =
            llvm_ir_gen.get_data_ownership();
        ret.codegen_opt = codegen_opt;
        ret.machine = target_machine;

        return ret;
    }

    void add_native_intrinsic_functions(llvm::ExecutionEngine &ee)
    {
#define ADD_GLOBAL_FUNC(NAME, FUNC) \
        ee.addGlobalMapping(#NAME, reinterpret_cast<uint64_t>(FUNC))

        auto *f32_exp10    = +[](float x)            { return std::pow(10.0f, x); };
        auto *f32_rsqrt    = +[](float x)            { return 1 / std::sqrt(x); };
        auto *f32_isfinite = +[](float x) -> int32_t { return std::isfinite(x); };
        auto *f32_isinf    = +[](float x) -> int32_t { return std::isinf(x); };
        auto *f32_isnan    = +[](float x) -> int32_t { return std::isnan(x); };
        
        auto *f64_exp10    = +[](double x)            { return std::pow(10.0, x); };
        auto *f64_rsqrt    = +[](double x)            { return 1 / std::sqrt(x); };
        auto *f64_isfinite = +[](double x) -> int32_t { return std::isfinite(x); };
        auto *f64_isinf    = +[](double x) -> int32_t { return std::isinf(x); };
        auto *f64_isnan    = +[](double x) -> int32_t { return std::isnan(x); };

        ADD_GLOBAL_FUNC(__cuj_intrinsic_f32_mod,      &::fmodf);
        ADD_GLOBAL_FUNC(__cuj_intrinsic_f32_rem,      &::remainderf);
        ADD_GLOBAL_FUNC(__cuj_intrinsic_f32_exp10,    f32_exp10);
        ADD_GLOBAL_FUNC(__cuj_intrinsic_f32_rsqrt,    f32_rsqrt);
        ADD_GLOBAL_FUNC(__cuj_intrinsic_f32_tan,      &::tanf);
        ADD_GLOBAL_FUNC(__cuj_intrinsic_f32_asin,     &::asinf);
        ADD_GLOBAL_FUNC(__cuj_intrinsic_f32_acos,     &::acosf);
        ADD_GLOBAL_FUNC(__cuj_intrinsic_f32_atan,     &::atanf);
        ADD_GLOBAL_FUNC(__cuj_intrinsic_f32_atan2,    &::atan2f);
        ADD_GLOBAL_FUNC(__cuj_intrinsic_f32_isfinite, f32_isfinite);
        ADD_GLOBAL_FUNC(__cuj_intrinsic_f32_isinf,    f32_isinf);
        ADD_GLOBAL_FUNC(__cuj_intrinsic_f32_isnan,    f32_isnan);

        using DD  = double(*)(double);
        using DDD = double(*)(double, double);

        ADD_GLOBAL_FUNC(__cuj_intrinsic_f64_mod,      static_cast<DDD>(&::fmod));
        ADD_GLOBAL_FUNC(__cuj_intrinsic_f64_rem,      static_cast<DDD>(&::remainder));
        ADD_GLOBAL_FUNC(__cuj_intrinsic_f64_exp10,    f64_exp10);
        ADD_GLOBAL_FUNC(__cuj_intrinsic_f64_rsqrt,    f64_rsqrt);
        ADD_GLOBAL_FUNC(__cuj_intrinsic_f64_tan,      static_cast<DD>(&::tan));
        ADD_GLOBAL_FUNC(__cuj_intrinsic_f64_asin,     static_cast<DD>(&::asin));
        ADD_GLOBAL_FUNC(__cuj_intrinsic_f64_acos,     static_cast<DD>(&::acos));
        ADD_GLOBAL_FUNC(__cuj_intrinsic_f64_atan,     static_cast<DD>(&::atan));
        ADD_GLOBAL_FUNC(__cuj_intrinsic_f64_atan2,    static_cast<DDD>(&::atan2));
        ADD_GLOBAL_FUNC(__cuj_intrinsic_f64_isfinite, f64_isfinite);
        ADD_GLOBAL_FUNC(__cuj_intrinsic_f64_isinf,    f64_isinf);
        ADD_GLOBAL_FUNC(__cuj_intrinsic_f64_isnan,    f64_isnan);

#undef ADD_GLOBAL_FUNC
    }

} // namespace anonymous

struct MCJIT::MCJITData
{
    std::string                llvm_ir;
    Box<llvm::LLVMContext>     llvm_context;
    Box<llvm::ExecutionEngine> exec_engine;
};

MCJIT::MCJIT(MCJIT &&other) noexcept
{
    std::swap(opts_, other.opts_);
    std::swap(llvm_data_, other.llvm_data_);
}

MCJIT &MCJIT::operator=(MCJIT &&other) noexcept
{
    std::swap(opts_, other.opts_);
    std::swap(llvm_data_, other.llvm_data_);
    return *this;
}

MCJIT::~MCJIT()
{
    delete llvm_data_;
}

void MCJIT::set_options(const Options &opts)
{
    opts_ = opts;
}

void MCJIT::generate(const dsl::Module &mod)
{
    delete llvm_data_;
    llvm_data_ = new MCJITData;
    
    auto llvm_mod = build_llvm_module(mod, opts_);
    llvm_data_->llvm_context = std::move(llvm_mod.llvm_context);

    llvm::raw_string_ostream ss(llvm_data_->llvm_ir);
    ss << *llvm_mod.llvm_module;
    ss.flush();

    std::string err;
    llvm::EngineBuilder engine_builder(std::move(llvm_mod.llvm_module));
    engine_builder.setErrorStr(&err);
    engine_builder.setOptLevel(llvm_mod.codegen_opt);

    auto exec_engine = engine_builder.create(llvm_mod.machine);
    if(!exec_engine)
        throw CujException(err);
    llvm_data_->exec_engine.reset(exec_engine);

    add_native_intrinsic_functions(*llvm_data_->exec_engine);

    llvm_data_->exec_engine->finalizeObject();
}

const std::string &MCJIT::get_llvm_string() const
{
    return llvm_data_->llvm_ir;
}

void *MCJIT::get_function_impl(const std::string &symbol_name) const
{
    return reinterpret_cast<void *>(
        llvm_data_->exec_engine->getFunctionAddress(symbol_name));
}

void *MCJIT::get_global_variable_impl(const std::string &symbol_name) const
{
    return reinterpret_cast<void *>(
        llvm_data_->exec_engine->getGlobalValueAddress(symbol_name));
}

CUJ_NAMESPACE_END(cuj::gen)

#ifdef _MSC_VER
#pragma warning(pop)
#endif
