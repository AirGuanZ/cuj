#pragma once

#include <cuj/dsl/module.h>
#include <cuj/utils/printer.h>

CUJ_NAMESPACE_BEGIN(cuj::gen)

class CPPCodeGenerator : public Uncopyable
{
public:

    enum class Target
    {
        Native,
        PTX
    };

    void set_target(Target target);

    void set_assert(bool enabled);

    const std::string &get_cpp_string() const;

    void generate(const dsl::Module &mod);

private:

    struct TypeDefineState
    {
        bool declared = false;
        bool complete = false;
    };

    static std::map<const core::Type *, std::type_index> build_type_to_index(const core::Prog &prog);

    static std::map<std::type_index, std::string> build_index_to_name(
        const std::map<const core::Type *, std::type_index> &type_to_index,
        const std::map<std::type_index, const core::Type *> &index_to_type);

    void define_types(const core::Prog &prog);

    void declare_type(std::map<std::string, TypeDefineState> &states, const core::Type *type);

    void define_type(std::map<std::string, TypeDefineState> &states, const core::Type *type);

    void generate_global_variables(const core::Prog &prog);

    void generate_global_consts(const core::Prog &prog);

    void declare_function(const core::Func &func, bool var_name);

    void define_function(const core::Func &func);

    void generate_local_allocas(const core::Func &func);

    void generate_local_temp_allocas(const core::Func &func);

    void generate(const core::Stat &s);

    void generate(const core::Store &s);

    void generate(const core::Copy &s);

    void generate(const core::Block &s);

    void generate(const core::Return &s);

    void generate(const core::If &s);

    void generate(const core::Loop &s);

    void generate(const core::Break &s);

    void generate(const core::Continue &s);

    void generate(const core::Switch &s);

    void generate(const core::CallFuncStat &s);

    void generate(const core::MakeScope &s);

    void generate(const core::ExitScope &s);

    void generate(const core::InlineAsm &s);

    std::string generate(const core::Expr &e) const;

    std::string generate(const core::FuncArgAddr &e) const;

    std::string generate(const core::LocalAllocAddr &e) const;

    std::string generate(const core::Load &e) const;

    std::string generate(const core::Immediate &e) const;

    std::string generate(const core::NullPtr &e) const;

    std::string generate(const core::ArithmeticCast &e) const;

    std::string generate(const core::BitwiseCast &e) const;

    std::string generate(const core::PointerOffset &e) const;

    std::string generate(const core::ClassPointerToMemberPointer &e) const;

    std::string generate(const core::DerefClassPointer &e) const;

    std::string generate(const core::DerefArrayPointer &e) const;

    std::string generate(const core::SaveClassIntoLocalAlloc &e) const;

    std::string generate(const core::SaveArrayIntoLocalAlloc &e) const;

    std::string generate(const core::ArrayAddrToFirstElemAddr &e) const;

    std::string generate(const core::Binary &e) const;

    std::string generate(const core::Unary &e) const;

    std::string generate(const core::CallFunc &e) const;

    std::string generate(const core::GlobalVarAddr &e) const;

    std::string generate(const core::GlobalConstAddr &e) const;

    std::string generate_intrinsic_call(const core::CallFunc &e) const;

    Target target_ = Target::Native;

#if defined(DEBUG) || defined(_DEBUG)
    bool enable_assert_ = true;
#else
    bool enable_assert_ = false;
#endif

    TextBuilder builder_;
    std::string result_;

    std::map<const core::Type *, std::string> type_names_;

    std::map<std::vector<unsigned char>, size_t> global_const_indices_;
    
    size_t next_label_index_ = 0;
    std::stack<std::string> break_dest_label_names_;
    std::stack<std::string> exit_scope_label_names_;

    mutable size_t local_temp_index_ = 0;

    const core::Prog *prog_ = nullptr;
};

CUJ_NAMESPACE_END(cuj::gen)
