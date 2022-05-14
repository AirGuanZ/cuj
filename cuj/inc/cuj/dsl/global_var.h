#pragma once

#include <cuj/core/expr.h>
#include <cuj/dsl/pointer.h>

CUJ_NAMESPACE_BEGIN(cuj::dsl)

using MemoryType = core::GlobalVar::MemoryType;

template<typename T>
class GlobalVariable
{
    RC<core::GlobalVar> var_;

public:

    explicit GlobalVariable(RC<core::GlobalVar> var = {});

    ptr<T> get_address() const;

    ref<T> get_reference() const;

    const std::string &get_symbol_name() const;
};

CUJ_NAMESPACE_END(cuj::dsl)
