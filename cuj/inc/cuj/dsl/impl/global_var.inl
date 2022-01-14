#pragma once

#include <string>

#include <cuj/dsl/function.h>
#include <cuj/dsl/global_var.h>

#include "cuj/dsl/module.h"

CUJ_NAMESPACE_BEGIN(cuj::dsl)
    template<typename T>
GlobalVariable<T>::GlobalVariable(RC<core::GlobalVar> var)
    : var_(var)
{
    
}

template<typename T>
ptr<T> GlobalVariable<T>::get_address() const
{
    return ptr<T>::_from_expr(core::GlobalVarAddr{
        .var = var_
    });
}

template<typename T>
ref<T> GlobalVariable<T>::get_reference() const
{
    return get_address().deref();
}

template<typename T>
const std::string &GlobalVariable<T>::get_symbol_name() const
{
    return var_->symbol_name;
}

CUJ_NAMESPACE_END(cuj::dsl)
