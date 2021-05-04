#pragma once

#include <cuj/ast/expr.h>

CUJ_NAMESPACE_BEGIN(cuj::ast)

namespace detail
{
    
    template<typename T>
    struct FuncRetType
    {
        using Type = Value<T>;
    };

    template<>
    struct FuncRetType<void>
    {
        using Type = void;
    };

    template<typename T>
    struct MakeArgValue
    {
        static_assert(std::is_arithmetic_v<T>);

        static auto process(T val)
        {
            return create_literial(val);
        }
    };

    template<typename T>
    struct MakeArgValue<ClassValue<T>>
    {
        static auto process(const ClassValue<T> &val)
        {
            return val;
        }
    };

    template<typename T, size_t N>
    struct MakeArgValue<Array<T, N>>
    {
        static auto process(const Array<T, N> &val)
        {
            return val;
        }
    };

    template<typename T>
    struct MakeArgValue<ArithmeticValue<T>>
    {
        static auto process(const ArithmeticValue<T> &val)
        {
            return val;
        }
    };

    template<typename T>
    struct MakeArgValue<Pointer<T>>
    {
        static auto process(const Pointer<T> &val)
        {
            return val;
        }
    };

} // namespace detail

class Context;
class FunctionContext;

template<typename Ret, typename...Args>
class FunctionImpl
{
public:

    explicit FunctionImpl(int func_index);

    template<typename...CallArgs>
    typename detail::FuncRetType<Ret>::Type
        operator()(const CallArgs &...args) const;

    const std::string &get_name() const;

private:
    
    bool check_return_type() const;
    
    bool check_param_types() const;

    template<size_t...Is>
    bool check_param_types_aux(std::index_sequence<Is...>) const;

    int index_;
};

template<typename T>
class Function;

namespace detail
{

    template<typename T>
    auto CFunctionArgTypeAux()
    {
        if constexpr(is_cuj_class<T> || is_array<T>)
            return std::declval<void*>();
        else
            return std::declval<T>();
    }

    template<typename T>
    using CFunctionArgType = rm_cvref_t<decltype(CFunctionArgTypeAux<T>())>;

} // namespace detail

template<typename Ret, typename...Args>
class Function<Ret(Args...)> : FunctionImpl<
    typename detail::DeValueType<RawToCUJType<Ret>>::Type, RawToCUJType<Args>...>
{
public:

    using ReturnType = typename detail::DeValueType<RawToCUJType<Ret>>::Type;

    using CFunctionType = std::conditional_t<
        is_cuj_class<ReturnType> || is_array<ReturnType>,
        void(void *, detail::CFunctionArgType<RawToCUJType<Args>>...),
        RawToCUJType<ReturnType>(detail::CFunctionArgType<RawToCUJType<Args>>...)>;

    using CFunctionPointer = CFunctionType*;

    template<typename...CallArgs>
    typename detail::FuncRetType<ReturnType>::Type
        operator()(const CallArgs &...args) const;

    using FunctionImpl<ReturnType, RawToCUJType<Args>...>::get_name;

private:

    friend class Context;

    static void get_arg_types(std::vector<const ir::Type *> &output);

    using FunctionImpl<ReturnType, RawToCUJType<Args>...>::FunctionImpl;
};

CUJ_NAMESPACE_END(cuj::ast)
