#pragma once

#include <functional>

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
        using ArgTypes = std::tuple<deval_t<rm_cvref_t<Args>>...>;

        using RetType = UsrRet;

        using FuncType = RetType(deval_t<rm_cvref_t<Args>>...);
    };
    
    template<typename T>
    auto cpp_arg_to_cuj_arg_type_aux()
    {
        using T1 = rm_cvref_t<T>;
        static_assert(std::is_arithmetic_v<T1> || std::is_pointer_v<T1>);
        if constexpr(std::is_arithmetic_v<T1>)
            return reinterpret_cast<T *>(0);
        else
            return reinterpret_cast<void **>(0);
    }

    template<typename T>
    struct CPPArgToCUJArgTypeAux
    {
        using Type =
            std::remove_reference_t<decltype(*cpp_arg_to_cuj_arg_type_aux<T>())>;
    };

    template<>
    struct CPPArgToCUJArgTypeAux<void>
    {
        using Type = void;
    };

    template<typename T>
    using to_cuj_arg_t = typename CPPArgToCUJArgTypeAux<T>::Type;

    template<typename Ret, typename...Args>
    using to_cuj_func_t = to_cuj_t<to_cuj_arg_t<Ret>>(to_cuj_arg_t<Args>...);

} // namespace func_trait_detail

template<typename UsrRet, typename T>
using func_args_t = typename func_trait_detail::FunctionTrait<UsrRet, T>::ArgTypes;

template<typename UsrRet, typename T>
using func_ret_t = typename func_trait_detail::FunctionTrait<UsrRet, T>::RetType;

template<typename UsrRet, typename T>
using func_t = detail::deval_func_t<
    typename func_trait_detail::FunctionTrait<UsrRet, T>::FuncType>;

CUJ_NAMESPACE_END(cuj::ast)
