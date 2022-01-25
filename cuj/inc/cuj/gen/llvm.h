#pragma once

#include <cuj/dsl/module.h>

namespace llvm
{

    class LLVMContext;
    class DataLayout;
    class Function;
    class FunctionType;
    class Module;
    class Type;
    class Value;

}

CUJ_NAMESPACE_BEGIN(cuj::gen)

class LLVMIRGenerator : public Uncopyable
{
public:

    enum class Target
    {
        Native,
        PTX
    };

    struct LLVMData;

    ~LLVMIRGenerator();

    void set_target(Target target);

    void use_fast_math();

    void use_approx_math_func();

    void disable_basic_optimizations();

    void set_data_layout(llvm::DataLayout *data_layout);

    void generate(const dsl::Module &mod);

    llvm::Module *get_llvm_module() const;

    std::pair<Box<llvm::LLVMContext>, Box<llvm::Module>> get_data_ownership();

    std::string get_llvm_string() const;

private:

    std::map<std::type_index, const core::Type *>
        build_type_to_index(const core::Prog &prog);

    llvm::Type *build_llvm_type(const core::Type *type);

    void build_llvm_struct_body(const core::Type *type);

    llvm::Type *get_llvm_type(const core::Type *type) const;

    void generate_global_variables();

    llvm::FunctionType *get_function_type(const core::Func &func);

    void declare_function(const core::Func *func);

    void define_function(const core::Func *func);

    void clear_temp_function_data();

    void generate_local_allocs(const core::Func *func);

    void generate_default_ret(const core::Func *func);

    void generate(const core::Stat &stat);

    void generate(const core::Store &store);

    void generate(const core::Copy &copy);

    void generate(const core::Block &block);

    void generate(const core::Return &ret);

    void generate(const core::If &if_s);

    void generate(const core::Loop &loop);

    void generate(const core::Break &break_s);

    void generate(const core::Continue &continue_s);

    void generate(const core::Switch &switch_s);

    void generate(const core::CallFuncStat &call);

    void generate(const core::MakeScope &make_scope);

    void generate(const core::ExitScope &exit_scope);

    void generate(const core::InlineAsm &inline_asm);

    llvm::Value *generate(const core::Expr &expr);

    llvm::Value *generate(const core::FuncArgAddr &expr);

    llvm::Value *generate(const core::LocalAllocAddr &expr);

    llvm::Value *generate(const core::Load &expr);

    llvm::Value *generate(const core::Immediate &expr);

    llvm::Value *generate(const core::NullPtr &expr);

    llvm::Value *generate(const core::ArithmeticCast &expr);

    llvm::Value *generate(const core::BitwiseCast &expr);

    llvm::Value *generate(const core::PointerOffset &expr);

    llvm::Value *generate(const core::ClassPointerToMemberPointer &expr);

    llvm::Value *generate(const core::DerefClassPointer &expr);

    llvm::Value *generate(const core::DerefArrayPointer &expr);

    llvm::Value *generate(const core::SaveClassIntoLocalAlloc &expr);

    llvm::Value *generate(const core::SaveArrayIntoLocalAlloc &expr);

    llvm::Value *generate(const core::ArrayAddrToFirstElemAddr &expr);

    llvm::Value *generate(const core::Binary &expr);

    llvm::Value *generate(const core::Unary &expr);

    llvm::Value *generate(const core::CallFunc &expr);

    llvm::Value *generate(const core::GlobalVarAddr &expr);

    llvm::Value *generate(const core::GlobalConstAddr &expr);

    llvm::Value *process_intrinsic_call(
        const core::CallFunc &call, const std::vector<llvm::Value *> &args);

    Target            target_              = Target::Native;
    bool              fast_math_           = false;
    bool              approx_math_func_    = false;
    bool              basic_optimizations_ = true;
    llvm::DataLayout *data_layout_         = nullptr;

    LLVMData *llvm_ = nullptr;
};

CUJ_NAMESPACE_END(cuj::gen)
