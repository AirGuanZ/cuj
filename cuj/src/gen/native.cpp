#include <cmath>
#include <iostream>

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable: 4141)
#pragma warning(disable: 4244)
#pragma warning(disable: 4624)
#pragma warning(disable: 4626)
#pragma warning(disable: 4996)
#endif

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

#ifdef _MSC_VER
#pragma warning(pop)
#endif

#include <cuj/gen/llvm.h>
#include <cuj/gen/native.h>

CUJ_NAMESPACE_BEGIN(cuj::gen)

namespace
{

    struct IntermediateModule
    {
        Box<llvm::LLVMContext>  llvm_context;
        Box<llvm::Module>       llvm_module;
        llvm::TargetMachine     *machine;
        llvm::CodeGenOpt::Level  codegen_opt;
    };

    IntermediateModule construct_llvm_module(
        const ir::Program &prog, const Options &opts)
    {
        llvm::InitializeNativeTarget();
        llvm::InitializeNativeTargetAsmPrinter();
        LLVMLinkInMCJIT();

        llvm::CodeGenOpt::Level codegen_opt;
        switch(opts.optimize_level)
        {
        case OptimizeLevel::O0: codegen_opt = llvm::CodeGenOpt::None;       break;
        case OptimizeLevel::O1: codegen_opt = llvm::CodeGenOpt::Less;       break;
        case OptimizeLevel::O2: codegen_opt = llvm::CodeGenOpt::Default;    break;
        case OptimizeLevel::O3: codegen_opt = llvm::CodeGenOpt::Aggressive; break;
        }

        auto target_triple = llvm::sys::getDefaultTargetTriple();

        std::string err;
        auto target = llvm::TargetRegistry::lookupTarget(target_triple, err);
        if(!target)
            throw CUJException(err);

        auto machine = target->createTargetMachine(
            target_triple, "generic", {},
            {}, {}, {}, codegen_opt, true);
        auto data_layout = machine->createDataLayout();
        
        LLVMIRGenerator llvm_gen;
        if(opts.fast_math)
            llvm_gen.use_fast_math();
        llvm_gen.disable_basic_optimizations();
        llvm_gen.set_target(LLVMIRGenerator::Target::Host);
        llvm_gen.generate(prog, &data_layout);

        llvm::PassManagerBuilder pass_mgr_builder;
        switch(opts.optimize_level)
        {
        case OptimizeLevel::O0: pass_mgr_builder.OptLevel = 0; break;
        case OptimizeLevel::O1: pass_mgr_builder.OptLevel = 1; break;
        case OptimizeLevel::O2: pass_mgr_builder.OptLevel = 2; break;
        case OptimizeLevel::O3: pass_mgr_builder.OptLevel = 3; break;
        }
        pass_mgr_builder.Inliner = llvm::createFunctionInliningPass(
            pass_mgr_builder.OptLevel, 0, false);

        pass_mgr_builder.SLPVectorize  = opts.auto_vectorization;
        pass_mgr_builder.LoopVectorize = opts.auto_vectorization;

        if(opts.optimize_level == OptimizeLevel::O2 ||
           opts.optimize_level == OptimizeLevel::O3)
            pass_mgr_builder.MergeFunctions = true;
        else
            pass_mgr_builder.MergeFunctions = false;

        machine->adjustPassManager(pass_mgr_builder);
        
        {
            llvm::legacy::FunctionPassManager fp_mgr(llvm_gen.get_module());
            pass_mgr_builder.populateFunctionPassManager(fp_mgr);
            fp_mgr.doInitialization();
            for(auto &f : llvm_gen.get_module()->functions())
                fp_mgr.run(f);
            fp_mgr.doFinalization();
        }

        {
            llvm::legacy::PassManager passes;
            passes.add(
                createTargetTransformInfoWrapperPass(machine->getTargetIRAnalysis()));
            pass_mgr_builder.populateModulePassManager(passes);
            passes.run(*llvm_gen.get_module());
        }

        IntermediateModule result;
        std::tie(result.llvm_context, result.llvm_module) =
            llvm_gen.get_data_ownership();
        result.codegen_opt = codegen_opt;
        result.machine = machine;

        return result;
    }

} // namespace anonymous

struct NativeJIT::Impl
{
    Box<llvm::LLVMContext>        llvm_context;
    Box<llvm::ExecutionEngine>    exec_engine;
    std::vector<RC<UntypedOwner>> func_contexts;
};

std::string NativeJIT::generate_llvm_ir(
    const ir::Program &prog, const Options &opts)
{
    auto im = construct_llvm_module(prog, opts);
    std::string result;
    llvm::raw_string_ostream ss(result);
    ss << *im.llvm_module;
    ss.flush();
    return result;
}

NativeJIT::NativeJIT(NativeJIT &&rhs) noexcept
    : NativeJIT()
{
    std::swap(impl_, rhs.impl_);
}

NativeJIT &NativeJIT::operator=(NativeJIT &&rhs) noexcept
{
    std::swap(impl_, rhs.impl_);
    return *this;
}

NativeJIT::~NativeJIT()
{
    delete impl_;
}

namespace
{

    int is_finite_f32(float x) { return std::isfinite(x); }
    int is_inf_f32   (float x) { return std::isinf(x); }
    int is_nan_f32   (float x) { return std::isnan(x); }
    
    int is_finite_f64(double x) { return std::isfinite(x); }
    int is_inf_f64   (double x) { return std::isinf(x); }
    int is_nan_f64   (double x) { return std::isnan(x); }

    float  rsqrt_f32(float x)  { return 1 / std::sqrt(x); }
    double rsqrt_f64(double x) { return 1 / std::sqrt(x); }

    float  exp10_f32(float x)  { return std::pow(10.0f, x); }
    double exp10_f64(double x) { return std::pow(10.0,  x); }

    int32_t min_i32(int32_t a, int32_t b) { return std::min(a, b); }
    int64_t min_i64(int64_t a, int64_t b) { return std::min(a, b); }
    int32_t max_i32(int32_t a, int32_t b) { return std::max(a, b); }
    int64_t max_i64(int64_t a, int64_t b) { return std::max(a, b); }

    float  min_f32(float a,  float b)  { return std::min(a, b); }
    double min_f64(double a, double b) { return std::min(a, b); }
    float  max_f32(float a,  float b)  { return std::max(a, b); }
    double max_f64(double a, double b) { return std::max(a, b); }

    void print(const char *msg)
    {
        std::cout << msg;
    }

    [[noreturn]] void assert_fail(
        const char *msg, const char *file, uint32_t line, const char *func)
    {
        (void)func;
        std::cerr << "Assertion failed: " << msg
                  << ", file " << file
                  << ", line " << line;
        std::abort();
    }

} // namespace anonymous

void NativeJIT::generate(const ir::Program &prog, const Options &opts)
{
    if(impl_)
        delete impl_;
    impl_ = new Impl;

    auto im = construct_llvm_module(prog, opts);
    impl_->llvm_context = std::move(im.llvm_context);

    std::string err;
    llvm::EngineBuilder engine_builder(std::move(im.llvm_module));
    engine_builder.setErrorStr(&err);
    engine_builder.setOptLevel(im.codegen_opt);

    auto exec_engine = engine_builder.create(im.machine);
    if(!exec_engine)
        throw CUJException(err);
    impl_->exec_engine.reset(exec_engine);

#define ADD_HOST_FUNC(NAME, FUNC)                                               \
    do {                                                                        \
        impl_->exec_engine->addGlobalMapping(                                   \
            NAME, reinterpret_cast<uint64_t>(FUNC));                            \
    } while(false)

    for(auto &f : prog.funcs)
    {
        if(auto func = f.as_if<RC<ir::ImportedHostFunction>>())
        {
            auto &hf = **func;
            auto host_func_symbol_name = hf.context_data ?
                ("_cuj_host_contexted_func_" + hf.symbol_name) : hf.symbol_name;
            impl_->exec_engine->addGlobalMapping(
                host_func_symbol_name, hf.address);
        }
    }

    ADD_HOST_FUNC("host.math.abs.f32",       &::fabsf);
    ADD_HOST_FUNC("host.math.mod.f32",       &::fmodf);
    ADD_HOST_FUNC("host.math.remainder.f32", &::remainderf);
    ADD_HOST_FUNC("host.math.exp.f32",       &::expf);
    ADD_HOST_FUNC("host.math.exp2.f32",      &::exp2f);
    ADD_HOST_FUNC("host.math.exp10.f32",     &exp10_f32);
    ADD_HOST_FUNC("host.math.log.f32",       &::logf);
    ADD_HOST_FUNC("host.math.log2.f32",      &::log2f);
    ADD_HOST_FUNC("host.math.log10.f32",     &::log10f);
    ADD_HOST_FUNC("host.math.pow.f32",       &::powf);
    ADD_HOST_FUNC("host.math.sqrt.f32",      &::sqrtf);
    ADD_HOST_FUNC("host.math.rsqrt.f32",     &rsqrt_f32);
    ADD_HOST_FUNC("host.math.sin.f32",       &::sinf);
    ADD_HOST_FUNC("host.math.cos.f32",       &::cosf);
    ADD_HOST_FUNC("host.math.tan.f32",       &::tanf);
    ADD_HOST_FUNC("host.math.asin.f32",      &::asinf);
    ADD_HOST_FUNC("host.math.acos.f32",      &::acosf);
    ADD_HOST_FUNC("host.math.atan.f32",      &::atanf);
    ADD_HOST_FUNC("host.math.atan2.f32",     &::atan2f);
    ADD_HOST_FUNC("host.math.ceil.f32",      &::ceilf);
    ADD_HOST_FUNC("host.math.floor.f32",     &::floorf);
    ADD_HOST_FUNC("host.math.trunc.f32",     &::truncf);
    ADD_HOST_FUNC("host.math.round.f32",     &::roundf);
    ADD_HOST_FUNC("host.math.isfinite.f32",  &is_finite_f32);
    ADD_HOST_FUNC("host.math.isinf.f32",     &is_inf_f32);
    ADD_HOST_FUNC("host.math.isnan.f32",     &is_nan_f32);
    
    using DD  = double(*)(double);
    using DD2 = double(*)(double, double);
    
    ADD_HOST_FUNC("host.math.abs.f64",       static_cast<DD> (&::fabs));
    ADD_HOST_FUNC("host.math.mod.f64",       static_cast<DD2>(&::fmod));
    ADD_HOST_FUNC("host.math.remainder.f64", static_cast<DD2>(&::remainder));
    ADD_HOST_FUNC("host.math.exp.f64",       static_cast<DD> (&::exp));
    ADD_HOST_FUNC("host.math.exp2.f64",      static_cast<DD> (&::exp2));
    ADD_HOST_FUNC("host.math.exp10.f64",     &exp10_f64);
    ADD_HOST_FUNC("host.math.log.f64",       static_cast<DD> (&::log));
    ADD_HOST_FUNC("host.math.log2.f64",      static_cast<DD> (&::log2));
    ADD_HOST_FUNC("host.math.log10.f64",     static_cast<DD> (&::log10));
    ADD_HOST_FUNC("host.math.pow.f64",       static_cast<DD2>(&::pow));
    ADD_HOST_FUNC("host.math.sqrt.f64",      static_cast<DD> (&::sqrt));
    ADD_HOST_FUNC("host.math.rsqrt.f64",     &rsqrt_f64);
    ADD_HOST_FUNC("host.math.sin.f64",       static_cast<DD> (&::sin));
    ADD_HOST_FUNC("host.math.cos.f64",       static_cast<DD> (&::cos));
    ADD_HOST_FUNC("host.math.tan.f64",       static_cast<DD> (&::tan));
    ADD_HOST_FUNC("host.math.asin.f64",      static_cast<DD> (&::asin));
    ADD_HOST_FUNC("host.math.acos.f64",      static_cast<DD> (&::acos));
    ADD_HOST_FUNC("host.math.atan.f64",      static_cast<DD> (&::atan));
    ADD_HOST_FUNC("host.math.atan2.f64",     static_cast<DD2>(&::atan2));
    ADD_HOST_FUNC("host.math.ceil.f64",      static_cast<DD> (&::ceil));
    ADD_HOST_FUNC("host.math.floor.f64",     static_cast<DD> (&::floor));
    ADD_HOST_FUNC("host.math.trunc.f64",     static_cast<DD> (&::trunc));
    ADD_HOST_FUNC("host.math.round.f64",     static_cast<DD> (&::round));
    ADD_HOST_FUNC("host.math.isfinite.f64",  &is_finite_f64);
    ADD_HOST_FUNC("host.math.isinf.f64",     &is_inf_f64);
    ADD_HOST_FUNC("host.math.isnan.f64",     &is_nan_f64);

    ADD_HOST_FUNC("host.math.min.i32", &min_i32);
    ADD_HOST_FUNC("host.math.min.i64", &min_i64);
    ADD_HOST_FUNC("host.math.min.f32", &min_f32);
    ADD_HOST_FUNC("host.math.min.f64", &min_f64);

    ADD_HOST_FUNC("host.math.max.i32", &max_i32);
    ADD_HOST_FUNC("host.math.max.i64", &max_i64);
    ADD_HOST_FUNC("host.math.max.f32", &max_f32);
    ADD_HOST_FUNC("host.math.max.f64", &max_f64);

    ADD_HOST_FUNC("host.system.print",  &print);
    ADD_HOST_FUNC("host.system.malloc", &::malloc);
    ADD_HOST_FUNC("host.system.free",   &::free);

    ADD_HOST_FUNC("host.system.assertfail", &assert_fail);

#undef ADD_HOST_FUNC

    impl_->exec_engine->finalizeObject();

    for(auto &f : prog.funcs)
    {
        if(auto func = f.as_if<RC<ir::ImportedHostFunction>>())
        {
            if(auto &c = func->get()->context_data)
                impl_->func_contexts.push_back(c);
        }
    }
}

void *NativeJIT::get_symbol_impl(const std::string &name) const
{
    return reinterpret_cast<void *>(
        impl_->exec_engine->getFunctionAddress(name));
}

CUJ_NAMESPACE_END(cuj::gen)
