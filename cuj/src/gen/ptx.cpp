#if CUJ_ENABLE_CUDA && CUJ_ENABLE_LLVM

#include <llvm/IR/LegacyPassManager.h>
#include <llvm/Support/TargetRegistry.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Target/TargetMachine.h>
#include <llvm/Target/TargetOptions.h>

#include <cuj/gen/ptx.h>

CUJ_NAMESPACE_BEGIN(cuj::gen)

namespace
{

    const char *get_target_triple(PTXGenerator::Target target)
    {
        switch(target)
        {
        case PTXGenerator::Target::PTX32:
            return "nvptx-nvidia-cuda";
        case PTXGenerator::Target::PTX64:
            return "nvptx64-nvidia-cuda";
        }
        unreachable();
    }

} // namespace detail

void PTXGenerator::set_target(Target target)
{
    target_ = target;
}

void PTXGenerator::generate(const ir::Program &prog)
{
    LLVMInitializeNVPTXTargetInfo();
    LLVMInitializeNVPTXTarget();
    LLVMInitializeNVPTXTargetMC();
    LLVMInitializeNVPTXAsmPrinter();

    std::string err;
    const char *target_triple = get_target_triple(target_);
    auto target = llvm::TargetRegistry::lookupTarget(target_triple, err);
    if(!target)
        throw std::runtime_error(err);

    auto machine = target->createTargetMachine(target_triple, "", {}, {}, {});

    auto data_layout = machine->createDataLayout();

    LLVMIRGenerator ir_gen;
    ir_gen.set_target(LLVMIRGenerator::Target::PTX);
    ir_gen.set_machine(&data_layout, target_triple);
    ir_gen.generate(prog);

    auto llvm_module = ir_gen.get_module();

    llvm::SmallString<8> output_buf;
    llvm::raw_svector_ostream output_stream(output_buf);

    llvm::legacy::PassManager passes;
    if(machine->addPassesToEmitFile(
        passes, output_stream, nullptr, llvm::CGFT_AssemblyFile))
        throw std::runtime_error("ptx file emission is not supported");

    passes.run(*llvm_module);

    result_.resize(output_buf.size());
    std::memcpy(result_.data(), output_buf.data(), output_buf.size());
}

const std::string &PTXGenerator::get_result() const
{
    return result_;
}

CUJ_NAMESPACE_END(cuj::gen)

#endif // #if CUJ_ENABLE_CUDA && CUJ_ENABLE_LLVM
