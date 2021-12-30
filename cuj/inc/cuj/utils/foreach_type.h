#pragma once

#include <cuj/common.h>

CUJ_NAMESPACE_BEGIN(cuj)

namespace detail
{

    template<typename ArgTuple, typename F, int...Is>
    void foreach_type_indexed_aux(const F &f, std::integer_sequence<int, Is...>)
    {
        ((f.template operator()<std::tuple_element_t<Is, ArgTuple>, Is>()), ...);
    }

} // namespace detail

template<typename...Args, typename F>
void foreach_type(const F &f)
{
    ((f.template operator()<Args>()), ...);
}

template<typename...Args, typename F>
void foreach_type_indexed(const F &f)
{
    detail::foreach_type_indexed_aux<std::tuple<Args...>>(
        f, std::make_integer_sequence<int, sizeof...(Args)>());
}

CUJ_NAMESPACE_END(cuj)
