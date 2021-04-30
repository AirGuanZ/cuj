#pragma once

#include <cuj/ast/func.h>
#include <cuj/ir/prog.h>
#include <cuj/util/uncopyable.h>

CUJ_NAMESPACE_BEGIN(cuj::gen)

class NativeJIT : public Uncopyable
{
public:

    enum class OptLevel
    {
        O0,
        O1,
        O2,
        O3,
        Default = O2,
    };

    NativeJIT() = default;

    NativeJIT(NativeJIT &&rhs) noexcept;

    NativeJIT &operator=(NativeJIT &&rhs) noexcept;

    ~NativeJIT();

    void generate(const ir::Program &prog, OptLevel opt = OptLevel::Default);

    template<typename FuncType>
    FuncType *get_symbol(const std::string &name) const;

    template<typename FuncType>
    typename ast::Function<FuncType>::CFunctionPointer
        get_symbol(const ast::Function<FuncType> &func) const;

private:

    void *get_symbol_impl(const std::string &name) const;

    struct Impl;

    Impl *impl_ = nullptr;
};

template<typename FuncType>
FuncType *NativeJIT::get_symbol(const std::string &name) const
{
    return reinterpret_cast<FuncType *>(get_symbol_impl(name));
}

template<typename FuncType>
typename ast::Function<FuncType>::CFunctionPointer
    NativeJIT::get_symbol(const ast::Function<FuncType> &func) const
{
    return reinterpret_cast<typename ast::Function<FuncType>::CFunctionPointer>(
        this->get_symbol_impl(func.get_name()));
}

CUJ_NAMESPACE_END(cuj::gen)
