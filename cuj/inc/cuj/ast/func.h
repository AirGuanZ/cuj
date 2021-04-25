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

    template<typename T>
    struct DeArithmeticValueType
    {
        using Type = T;
    };

    template<typename T>
    struct DeArithmeticValueType<ArithmeticValue<T>>
    {
        using Type = T;
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

private:
    
    bool check_return_type() const;
    
    bool check_param_types() const;

    template<size_t...Is>
    bool check_param_types_aux(std::index_sequence<Is...>) const;

    int index_;
};

template<typename T>
class Function;

template<typename Ret, typename...Args>
class Function<Ret(Args...)> :
    FunctionImpl<RawToCUJType<Ret>, RawToCUJType<Args>...>
{
public:

    using ReturnType = RawToCUJType<Ret>;

    template<typename...CallArgs>
    typename detail::FuncRetType<RawToCUJType<Ret>>::Type
        operator()(const CallArgs &...args) const;

private:

    friend class Context;

    static void get_arg_types(std::vector<const ir::Type *> &output);

    using FunctionImpl<RawToCUJType<Ret>, RawToCUJType<Args>...>::FunctionImpl;
};

CUJ_NAMESPACE_END(cuj::ast)
