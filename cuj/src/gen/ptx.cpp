#if CUJ_ENABLE_CUDA

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable: 4141)
#pragma warning(disable: 4244)
#pragma warning(disable: 4624)
#pragma warning(disable: 4626)
#pragma warning(disable: 4996)
#endif

#include <llvm/Target/TargetLoweringObjectFile.h>

#include <llvm/Analysis/TargetTransformInfo.h>
#include <llvm/Analysis/TargetLibraryInfo.h>
#include <llvm/CodeGen/MachineModuleInfo.h>
#include <llvm/CodeGen/TargetPassConfig.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/TargetRegistry.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Target/TargetMachine.h>
#include <llvm/Target/TargetOptions.h>
#include <llvm/Transforms/IPO/PassManagerBuilder.h>
#include <llvm/Transforms/IPO.h>

#ifdef _MSC_VER
#pragma warning(pop)
#endif

#include <cuj/gen/ptx.h>

CUJ_NAMESPACE_BEGIN(cuj::gen)

namespace
{

    struct IntermediateModule
    {
        std::unique_ptr<llvm::Module> llvm_module;
        llvm::TargetMachine          *machine;
    };

    IntermediateModule construct_intermediate_module(
        const ir::Program &prog, OptLevel opt)
    {
        LLVMInitializeNVPTXTargetInfo();
        LLVMInitializeNVPTXTarget();
        LLVMInitializeNVPTXTargetMC();
        LLVMInitializeNVPTXAsmPrinter();

        std::string err;
        const char *target_triple = "nvptx64-nvidia-cuda";
        auto target = llvm::TargetRegistry::lookupTarget(target_triple, err);
        if(!target)
            throw CUJException(err);

        auto machine = target->createTargetMachine(
            target_triple, "sm_20", "+ptx63", {}, {},
            {}, llvm::CodeGenOpt::Aggressive);
        auto data_layout = machine->createDataLayout();

        LLVMIRGenerator ir_gen;
        ir_gen.set_target(LLVMIRGenerator::Target::PTX);
        ir_gen.generate(prog, &data_layout);

        auto llvm_module = ir_gen.get_module();
        llvm_module->setTargetTriple(target_triple);
        llvm_module->setDataLayout(data_layout);

        llvm::PassManagerBuilder pass_mgr_builder;
        switch(opt)
        {
        case OptLevel::O0: pass_mgr_builder.OptLevel = 0; break;
        case OptLevel::O1: pass_mgr_builder.OptLevel = 1; break;
        case OptLevel::O2: pass_mgr_builder.OptLevel = 2; break;
        case OptLevel::O3: pass_mgr_builder.OptLevel = 3; break;
        }
        pass_mgr_builder.Inliner = llvm::createFunctionInliningPass(
            pass_mgr_builder.OptLevel, 0, false);
        pass_mgr_builder.SLPVectorize = true;
        pass_mgr_builder.MergeFunctions = true;

        {
            llvm::legacy::FunctionPassManager fp_mgr(llvm_module);
            pass_mgr_builder.populateFunctionPassManager(fp_mgr);
            for(auto &f : llvm_module->functions())
                fp_mgr.run(f);
        }

        machine->adjustPassManager(pass_mgr_builder);

        llvm::legacy::PassManager passes;
        passes.add(
            createTargetTransformInfoWrapperPass(machine->getTargetIRAnalysis()));
        pass_mgr_builder.populateModulePassManager(passes);

        passes.run(*llvm_module);

        IntermediateModule result;
        result.llvm_module = ir_gen.get_module_ownership();
        result.machine     = machine;

        return result;
    }

} // namespace anonymouos

std::string PTXGenerator::generate_llvm_ir(
    const ir::Program &prog, OptLevel opt)
{
    auto im = construct_intermediate_module(prog, opt);

    std::string result;
    llvm::raw_string_ostream ss(result);
    ss << *im.llvm_module;
    ss.flush();
    return result;
}

void PTXGenerator::generate(const ir::Program &prog, OptLevel opt)
{
    auto im = construct_intermediate_module(prog, opt);
    
    llvm::legacy::PassManager passes;
    llvm::SmallString<8> output_buf;
    llvm::raw_svector_ostream output_stream(output_buf);
    if(im.machine->addPassesToEmitFile(
        passes, output_stream, nullptr, llvm::CGFT_AssemblyFile))
        throw CUJException("ptx file emission is not supported");

    passes.run(*im.llvm_module);

    result_.resize(output_buf.size());
    std::memcpy(result_.data(), output_buf.data(), output_buf.size());
}

const std::string &PTXGenerator::get_result() const
{
    return result_;
}

CUJ_NAMESPACE_END(cuj::gen)

#endif // #if CUJ_ENABLE_CUDA
