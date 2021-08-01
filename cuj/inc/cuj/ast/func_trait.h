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
        using ArgTypes = std::tuple<
            typename detail::DeValueType<rm_cvref_t<Args>>::Type...>;

        using RetType = UsrRet;

        using FuncType =
            RetType(typename detail::DeValueType<rm_cvref_t<Args>>::Type...);
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
    using CPPArgToCUJArgType = typename CPPArgToCUJArgTypeAux<T>::Type;

    template<typename Ret, typename...Args>
    using CPPFuncToCUJFuncType =
        RawToCUJType<CPPArgToCUJArgType<Ret>>(CPPArgToCUJArgType<Args>...);

} // namespace func_trait_detail

template<typename UsrRet, typename T>
using FunctionArgs = typename func_trait_detail::FunctionTrait<UsrRet, T>::ArgTypes;

template<typename UsrRet, typename T>
using FunctionRet = typename func_trait_detail::FunctionTrait<UsrRet, T>::RetType;

template<typename UsrRet, typename T>
using FunctionType = typename func_trait_detail::FunctionTrait<UsrRet, T>::FuncType;

CUJ_NAMESPACE_END(cuj::ast)
