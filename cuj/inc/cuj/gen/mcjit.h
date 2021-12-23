#pragma once

#include <cuj/dsl/module.h>
#include <cuj/gen/option.h>

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
    T *get_function(const std::string &symbol_name) const;

private:

    struct MCJITData;

    void *get_function_impl(const std::string &symbol_name) const;

    Options    opts_;
    MCJITData *llvm_data_ = nullptr;
};

template<typename T>
T *MCJIT::get_function(const std::string &symbol_name) const
{
    return reinterpret_cast<T *>(get_function_impl(symbol_name));
}

CUJ_NAMESPACE_END(cuj::gen)
