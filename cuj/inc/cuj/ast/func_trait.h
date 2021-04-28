#pragma once

#include <functional>

#include <cuj/ast/func.h>

CUJ_NAMESPACE_BEGIN(cuj::ast)

namespace func_trait_detail
{

    template<typename UsrRet, typename T>
    struct FunctionTrait
    {
        using ArgTypes = typename FunctionTrait<UsrRet, rm_cvref_t<decltype(
            std::function{ std::declval<rm_cvref_t<T>>() })>>::ArgTypes;

        using RetType = typename FunctionTrait<UsrRet, rm_cvref_t<decltype(
            std::function{ std::declval<rm_cvref_t<T>>() })>>::RetType;

        using FuncType = typename FunctionTrait<UsrRet, rm_cvref_t<decltype(
            std::function{ std::declval<rm_cvref_t<T>>() })>>::FuncType;
    };

    template<typename UsrRet, typename...Args>
    struct FunctionTrait<UsrRet, std::function<void(Args...)>>
    {
        using ArgTypes = std::tuple<
            typename detail::DeValueType<rm_cvref_t<Args>>::Type...>;

        using RetType = UsrRet;

        using FuncType = RetType(
            typename detail::DeValueType<rm_cvref_t<Args>>::Type...);
    };

} // namespace func_trait_detail

template<typename UsrRet, typename T>
using FunctionArgs = typename func_trait_detail::FunctionTrait<UsrRet, T>::ArgTypes;

template<typename UsrRet, typename T>
using FunctionRet = typename func_trait_detail::FunctionTrait<UsrRet, T>::RetType;

template<typename UsrRet, typename T>
using FunctionType = typename func_trait_detail::FunctionTrait<UsrRet, T>::FuncType;

CUJ_NAMESPACE_END(cuj::ast)
