#pragma once

#include <cuj/common.h>

CUJ_NAMESPACE_BEGIN(cuj)

namespace typelist_detail
{

    template<int I, typename T, typename...Ts>
    struct TypeListGetImpl
    {
        using Type = TypeListGetImpl<I - 1, Ts...>;
    };

    template<typename T, typename...Ts>
    struct TypeListGetImpl<0, T, Ts...>
    {
        using Type = T;
    };

} // namespace typelist_detail

template<typename...Ts>
struct TypeList
{
    static constexpr int N = sizeof...(Ts);

    template<int I>
    using Get = typelist_detail::TypeListGetImpl<I, Ts...>;
};

CUJ_NAMESPACE_END(cuj)
