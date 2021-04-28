#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable: 4141)
#pragma warning(disable: 4624)
#endif

#include <llvm/ExecutionEngine/ExecutionEngine.h>
#include <llvm/Support/TargetSelect.h>

#ifdef _MSC_VER
#pragma warning(pop)
#endif

#include <cuj/gen/llvm.h>
#include <cuj/gen/native.h>

CUJ_NAMESPACE_BEGIN(cuj::gen)

struct NativeJIT::Impl
{
    LLVMIRGenerator llvm_gen;

    std::unique_ptr<llvm::ExecutionEngine> exec_engine;
};

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
    if(impl_)
        delete impl_;
}

namespace
{

    int is_finite_f32(float x) { return isfinite(x); }
    int is_inf_f32   (float x) { return isinf(x); }
    int is_nan_f32   (float x) { return isnan(x); }
    
    int is_finite_f64(double x) { return isfinite(x); }
    int is_inf_f64   (double x) { return isinf(x); }
    int is_nan_f64   (double x) { return isnan(x); }

} // namespace anonymous

void NativeJIT::generate(const ir::Program &prog)
{
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
    LLVMLinkInMCJIT();

    CUJ_ASSERT(!impl_);
    impl_ = new Impl;
    impl_->llvm_gen.set_target(LLVMIRGenerator::Target::Host);
    impl_->llvm_gen.generate(prog);

    std::string err_str;
    llvm::EngineBuilder engine_builder(impl_->llvm_gen.get_module_ownership());
    engine_builder.setErrorStr(&err_str);

    auto exec_engine = engine_builder.create();
    if(!exec_engine)
        throw CUJException(err_str);
    impl_->exec_engine.reset(exec_engine);

#define ADD_HOST_FUNC(NAME, FUNC)                                               \
    do {                                                                        \
        impl_->exec_engine->addGlobalMapping(                                   \
            NAME, reinterpret_cast<uint64_t>(FUNC));                            \
    } while(false)

    ADD_HOST_FUNC("host.math.abs.f32",       &::fabsf);
    ADD_HOST_FUNC("host.math.mod.f32",       &::fmodf);
    ADD_HOST_FUNC("host.math.remainder.f32", &::remainderf);
    ADD_HOST_FUNC("host.math.exp.f32",       &::expf);
    ADD_HOST_FUNC("host.math.exp2.f32",      &::exp2f);
    ADD_HOST_FUNC("host.math.log.f32",       &::logf);
    ADD_HOST_FUNC("host.math.log2.f32",      &::log2f);
    ADD_HOST_FUNC("host.math.log10.f32",     &::log10f);
    ADD_HOST_FUNC("host.math.pow.f32",       &::powf);
    ADD_HOST_FUNC("host.math.sqrt.f32",      &::sqrtf);
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
    ADD_HOST_FUNC("host.math.log.f64",       static_cast<DD> (&::log));
    ADD_HOST_FUNC("host.math.log2.f64",      static_cast<DD> (&::log2));
    ADD_HOST_FUNC("host.math.log10.f64",     static_cast<DD> (&::log10));
    ADD_HOST_FUNC("host.math.pow.f64",       static_cast<DD2>(&::pow));
    ADD_HOST_FUNC("host.math.sqrt.f64",      static_cast<DD> (&::sqrt));
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

#undef ADD_HOST_FUNC
}

void *NativeJIT::get_symbol_impl(const std::string &name) const
{
    return reinterpret_cast<void *>(
        impl_->exec_engine->getFunctionAddress(name));
}

CUJ_NAMESPACE_END(cuj::gen)
