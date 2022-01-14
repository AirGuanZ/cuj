#pragma once

#include <cuj/gen/option.h>

CUJ_NAMESPACE_BEGIN(cuj::dsl)

class Module;

CUJ_NAMESPACE_END(cuj::dsl)

CUJ_NAMESPACE_BEGIN(cuj::gen)

class MCJIT : public Uncopyable
{
public:

    MCJIT() = default;

    MCJIT(MCJIT &&other) noexcept;

    MCJIT &operator=(MCJIT &&other) noexcept;

    ~MCJIT();

    void set_options(const Options &opts);

    void generate(const dsl::Module &mod);

    const std::string &get_llvm_string() const;

    template<typename T>
        requires std::is_function_v<T>
    T *get_function(const std::string &symbol_name) const;

    template<typename T, typename Ret, typename...Args>
        requires std::is_function_v<T>
    T *get_function(const dsl::Function<Ret(Args...)> &func) const;

    template<typename Ret, typename...Args>
        requires (!std::is_function_v<Ret>)
    auto get_function(const dsl::Function<Ret(Args...)> &func) const;

    template<typename T>
    T *get_global_variable(const std::string &symbol_name) const;

    template<typename T>
    auto get_global_variable(const dsl::GlobalVariable<T> &var) const;

    template<typename T, typename U>
    auto get_global_variable(const dsl::GlobalVariable<U> &var) const;

private:

    struct MCJITData;

    void *get_function_impl(const std::string &symbol_name) const;

    void *get_global_variable_impl(const std::string &symbol_name) const;

    Options    opts_;
    MCJITData *llvm_data_ = nullptr;
};

CUJ_NAMESPACE_END(cuj::gen)

#include <cuj/gen/impl/mcjit.inl>
