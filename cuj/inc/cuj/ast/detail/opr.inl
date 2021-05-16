#pragma once

#include <cuj/ast/stat_builder.h>

CUJ_NAMESPACE_BEGIN(cuj::ast)

template<typename T, typename C>
ArithmeticValue<T> select(
    const ArithmeticValue<C> &cond,
    const ArithmeticValue<T> &true_val,
    const ArithmeticValue<T> &false_val)
{
    ArithmeticValue<T> ret;
    IfBuilder() + cond + [&] { ret = true_val; } - [&] { ret = false_val; };
    return ret;
}

CUJ_NAMESPACE_END(cuj::ast)
