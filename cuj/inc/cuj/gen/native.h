#pragma once

#include <cuj/ast/func.h>
#include <cuj/ir/prog.h>
#include <cuj/gen/llvm.h>
#include <cuj/util/uncopyable.h>

CUJ_NAMESPACE_BEGIN(cuj::gen)

class NativeJIT : public Uncopyable
{
public:

    struct Options
    {
        OptLevel opt_level  = OptLevel::Default;
        bool     fast_math  = false;
        bool     enable_slp = true;
    };

    static std::string generate_llvm_ir(
        const ir::Program &prog, OptLevel opt = OptLevel::Default);

    static std::string generate_llvm_ir(
        const ir::Program &prog, const Options &opts);

    NativeJIT() = default;

    NativeJIT(NativeJIT &&rhs) noexcept;

    NativeJIT &operator=(NativeJIT &&rhs) noexcept;

    ~NativeJIT();

    void generate(const ir::Program &prog, OptLevel opt = OptLevel::Default);

    void generate(const ir::Program &prog, const Options &opts);

    template<typename FuncType>
    FuncType *get_function_by_name(const std::string &name) const;

    template<typename ForcedCFunction, typename FuncType>
    typename ast::Function<ForcedCFunction, FuncType>::CFunctionPointer
        get_function(const ast::Function<ForcedCFunction, FuncType> &func) const;

private:

    void *get_symbol_impl(const std::string &name) const;

    struct Impl;

    Impl *impl_ = nullptr;
};

template<typename FuncType>
FuncType *NativeJIT::get_function_by_name(const std::string &name) const
{
    return reinterpret_cast<FuncType *>(get_symbol_impl(name));
}

template<typename ForcedCFunction, typename FuncType>
typename ast::Function<ForcedCFunction, FuncType>::CFunctionPointer
    NativeJIT::get_function(const ast::Function<ForcedCFunction, FuncType> &func) const
{
    return reinterpret_cast<typename ast::Function<ForcedCFunction, FuncType>::CFunctionPointer>(
        this->get_symbol_impl(func.get_name()));
}

CUJ_NAMESPACE_END(cuj::gen)
