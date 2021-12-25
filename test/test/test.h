#pragma once

#include <catch2/catch.hpp>

#include <cuj.h>

using namespace cuj;

namespace test_detail
{

    template<typename Tuple, int...Is>
    auto tuple_prefix(const Tuple &tuple, std::integer_sequence<int, Is...>)
    {
        return std::tuple{ std::get<Is>(tuple)... };
    }

} // namespace test_detail

template<typename F, typename Action>
void with_mcjit(F &&f, Action &&action)
{
    ScopedModule mod;
    auto func = function(std::forward<F>(f));
    MCJIT mcjit;
    mcjit.generate(mod);
    auto c_func = mcjit.get_function(func);
    REQUIRE(c_func);
    if(c_func)
        std::forward<Action>(action)(c_func);
}

template<typename F, typename...Args>
void mcjit_require(F &&f, Args...args)
{
    with_mcjit(std::forward<F>(f), [&](auto c_func)
    {
        auto func_args = test_detail::tuple_prefix(
            std::tuple{ args... },
            std::make_integer_sequence<int, sizeof...(args) - 1>());
        auto result = std::get<sizeof...(args) - 1>(std::tuple{ args... });
        if constexpr(std::is_floating_point_v<decltype(result)>)
        {
            REQUIRE(
                std::apply(c_func, func_args) == Approx(result));
        }
        else
        {
            REQUIRE(
                std::apply(c_func, func_args) == result);
        }
    });
}
