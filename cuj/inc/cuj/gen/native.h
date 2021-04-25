#pragma once

#if CUJ_ENABLE_LLVM

#include <cuj/ir/prog.h>
#include <cuj/util/uncopyable.h>

CUJ_NAMESPACE_BEGIN(cuj::gen)

class NativeJIT : public Uncopyable
{
public:

    NativeJIT() = default;

    NativeJIT(NativeJIT &&rhs) noexcept;

    NativeJIT &operator=(NativeJIT &&rhs) noexcept;

    ~NativeJIT();

    void generate(const ir::Program &prog);

    template<typename FuncType>
    FuncType *get_symbol(const std::string &name) const;

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

CUJ_NAMESPACE_END(cuj::gen)

#endif // #if CUJ_ENABLE_LLVM
