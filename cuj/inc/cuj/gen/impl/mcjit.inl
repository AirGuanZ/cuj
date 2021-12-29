#pragma once

#include <cassert>

CUJ_NAMESPACE_BEGIN(cuj::gen)

namespace mcjit_detail
{

    template<typename CArg, typename Arg>
    constexpr bool is_arg_compatible()
    {
        if constexpr(dsl::is_cuj_ref_v<Arg>)
        {
            if constexpr(std::is_pointer_v<CArg>)
            {
                using DCArg = std::remove_const_t<std::remove_pointer_t<CArg>>;
                using DArg = dsl::remove_reference_t<Arg>;
                return std::is_same_v<dsl::cxx<DCArg>, DArg> ||
                       std::is_same_v<DCArg, char> ||
                       std::is_same_v<DCArg, unsigned char> ||
                       std::is_same_v<DCArg, signed char> ||
                       std::is_same_v<DCArg, void>;
            }
            else
                return false;
        }
        else if constexpr(std::is_same_v<Arg, dsl::CujVoid>)
            return false;
        else if constexpr(dsl::is_cuj_arithmetic_v<Arg> ||
                          dsl::is_cuj_class_v<Arg> ||
                          dsl::is_cuj_array_v<Arg>)
            return std::is_same_v<dsl::cxx<CArg>, Arg>;
        else
        {
            static_assert(dsl::is_cuj_pointer_v<Arg>);
            if constexpr(std::is_pointer_v<CArg>)
            {
                using CPointed = std::remove_const_t<std::remove_pointer_t<CArg>>;
                using Pointed = typename Arg::PointedType;
                return std::is_same_v<dsl::cxx<CPointed>, Pointed> ||
                       std::is_same_v<CPointed, char> ||
                       std::is_same_v<CPointed, unsigned char> ||
                       std::is_same_v<CPointed, signed char> ||
                       std::is_same_v<CPointed, void>;
            }
            else
                return false;
        }
    }

    template<typename CRet, typename Ret>
    constexpr bool is_ret_compatible()
    {
        if constexpr(std::is_same_v<Ret, dsl::CujVoid>)
            return std::is_same_v<CRet, void>;
        else
            return is_arg_compatible<CRet, Ret>();
    }

    template<typename CArgTuple, typename ArgTuple, int...Is>
    constexpr bool is_args_compatible(std::integer_sequence<int, Is...>)
    {
        return ((is_arg_compatible<
            std::tuple_element_t<Is, CArgTuple>,
            std::tuple_element_t<Is, ArgTuple>>()) && ...);
    }

    template<typename CArgTuple, typename ArgTuple>
    constexpr bool is_args_compatible()
    {
        if constexpr(std::tuple_size_v<CArgTuple> != std::tuple_size_v<ArgTuple>)
            return false;
        {
            return is_args_compatible<CArgTuple, ArgTuple>(
                std::make_integer_sequence<int, std::tuple_size_v<CArgTuple>>());
        }
    }

    template<typename Arg>
    struct ArgToCArg { };

    template<typename T>
    struct ArgToCArg<dsl::Arithmetic<T>>
    {
        using Type = T;
    };

    template<typename T>
    struct ArgToCArg<dsl::Pointer<T>>
    {
        using Type = std::add_pointer_t<typename ArgToCArg<T>::Type>;
    };

    template<typename T, size_t N>
    struct ArgToCArg<dsl::Array<T, N>>
    {
        using Type = dsl::cuj_to_cxx_t<dsl::Array<T, N>>;
    };

    template<typename T> requires dsl::is_cuj_class_v<T>
    struct ArgToCArg<T>
    {
        using Type = dsl::cuj_to_cxx_t<T>;
    };

    template<typename T>
    struct ArgToCArg<dsl::ref<T>>
    {
        using Type = std::add_pointer_t<typename ArgToCArg<T>::Type>;
    };

    template<>
    struct ArgToCArg<dsl::CujVoid>
    {
        using Type = void;
    };

    template<typename F>
    struct FunctionTypeToCFunctionType { };

    template<typename Ret, typename...Args>
    struct FunctionTypeToCFunctionType<Ret(Args...)>
    {
        using Type =
            typename ArgToCArg<Ret>::Type(typename ArgToCArg<Args>::Type...);
    };

    template<typename T, typename Ret, typename...Args>
    struct CFunctionSignatureTrait
    {
        
    };

    template<typename CRet, typename...CArgs, typename Ret, typename...Args>
    struct CFunctionSignatureTrait<CRet(CArgs...), Ret, Args...>
    {
        static constexpr bool ret_compatible = is_ret_compatible<CRet, Ret>();
        static constexpr bool arg_compatible =
            is_args_compatible<std::tuple<CArgs...>, std::tuple<Args...>>();
        static constexpr bool compatible = ret_compatible && arg_compatible;
    };

} // namespace mcjit_detail

template<typename T>
    requires std::is_function_v<T>
T *MCJIT::get_function(const std::string &symbol_name) const
{
    return reinterpret_cast<T *>(get_function_impl(symbol_name));
}

template<typename T, typename Ret, typename...Args>
    requires std::is_function_v<T>
T *MCJIT::get_function(const dsl::Function<Ret(Args...)> &func) const
{
    static_assert(
        mcjit_detail::CFunctionSignatureTrait<T, Ret, Args...>::compatible,
        "function signature doesn't match");
    const auto &name = func._get_context()->get_core_func()->name;
    assert(!name.empty());
    return this->get_function<T>(name);
}

template<typename Ret, typename ... Args>
    requires !std::is_function_v<Ret>
auto MCJIT::get_function(const dsl::Function<Ret(Args...)> &func) const
{
    using CFunctionType =
        typename mcjit_detail::FunctionTypeToCFunctionType<Ret(Args...)>::Type;
    return this->get_function<CFunctionType>(func);
}

CUJ_NAMESPACE_END(cuj::gen)
