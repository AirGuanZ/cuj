#if CUJ_ENABLE_LLVM

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
}

void *NativeJIT::get_symbol_impl(const std::string &name) const
{
    return reinterpret_cast<void *>(
        impl_->exec_engine->getFunctionAddress(name));
}

CUJ_NAMESPACE_END(cuj::gen)

#endif // #if CUJ_ENABLE_LLVM
