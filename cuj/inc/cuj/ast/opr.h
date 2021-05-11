#pragma once

#include <cuj/ast/value.h>

CUJ_NAMESPACE_BEGIN(cuj::ast)

// binary arithmetic

#define CUJ_OVERLOAD_BINARY_ARITHMETIC_OP(OP, SYM)                              \
template<typename L, typename R>                                                \
auto operator SYM(const ArithmeticValue<L> &lhs, const ArithmeticValue<R> &rhs) \
{                                                                               \
    using T = decltype(std::declval<L>() SYM std::declval<R>());                \
    auto impl = create_binary_operator<T, L, R>(                                \
        ir::BinaryOp::Type::OP, lhs.get_impl(), rhs.get_impl());                \
    return Value<T>(std::move(impl));                                           \
}                                                                               \
template<typename L, typename R,                                                \
         typename = std::enable_if_t<std::is_arithmetic_v<R>>>                  \
auto operator SYM(const ArithmeticValue<L> &lhs, R rhs)                         \
{                                                                               \
    return lhs SYM create_literial(rhs);                                        \
}                                                                               \
template<typename L, typename R,                                                \
         typename = std::enable_if_t<std::is_arithmetic_v<L>>>                  \
auto operator SYM(L lhs, const ArithmeticValue<R> &rhs)                         \
{                                                                               \
    return create_literial(lhs) SYM rhs;                                        \
}

CUJ_OVERLOAD_BINARY_ARITHMETIC_OP(Add, +)
CUJ_OVERLOAD_BINARY_ARITHMETIC_OP(Sub, -)
CUJ_OVERLOAD_BINARY_ARITHMETIC_OP(Mul, *)
CUJ_OVERLOAD_BINARY_ARITHMETIC_OP(Div, /)

CUJ_OVERLOAD_BINARY_ARITHMETIC_OP(And, &&)
CUJ_OVERLOAD_BINARY_ARITHMETIC_OP(Or,  ||)
CUJ_OVERLOAD_BINARY_ARITHMETIC_OP(XOr, ^)

CUJ_OVERLOAD_BINARY_ARITHMETIC_OP(Equal,        ==)
CUJ_OVERLOAD_BINARY_ARITHMETIC_OP(NotEqual,     !=)
CUJ_OVERLOAD_BINARY_ARITHMETIC_OP(Less,         <)
CUJ_OVERLOAD_BINARY_ARITHMETIC_OP(LessEqual,    <=)
CUJ_OVERLOAD_BINARY_ARITHMETIC_OP(Greater,      >)
CUJ_OVERLOAD_BINARY_ARITHMETIC_OP(GreaterEqual, >=)

#undef CUJ_OVERLOAD_BINARY_ARITHMETIC_OP

// binary pointer

#define CUJ_OVERLOAD_BINARY_POINTER_OP(OP, SYM)                                     \
    template<typename L, typename R>                                                \
    Value<bool> operator SYM(const PointerImpl<L> &lhs, const PointerImpl<R> &rhs)  \
    {                                                                               \
        auto impl = create_binary_operator<bool, PointerImpl<L>, PointerImpl<R>>(   \
            ir::BinaryOp::Type::OP, lhs.get_impl(), rhs.get_impl());                \
        return Value<bool>(std::move(impl));                                        \
    }                                                                               \
    template<typename T>                                                            \
    Value<bool> operator SYM(const PointerImpl<T> &ptr, const std::nullptr_t &)     \
    {                                                                               \
        return ptr SYM PointerImpl<T>(nullptr);                                     \
    }                                                                               \
    template<typename T>                                                            \
    Value<bool> operator SYM(const std::nullptr_t &, const PointerImpl<T> &ptr)     \
    {                                                                               \
        return PointerImpl<T>(nullptr) SYM ptr;                                     \
    }

CUJ_OVERLOAD_BINARY_POINTER_OP(Equal,        ==)
CUJ_OVERLOAD_BINARY_POINTER_OP(NotEqual,     !=)
CUJ_OVERLOAD_BINARY_POINTER_OP(Less,         <)
CUJ_OVERLOAD_BINARY_POINTER_OP(LessEqual,    <=)
CUJ_OVERLOAD_BINARY_POINTER_OP(Greater,      >)
CUJ_OVERLOAD_BINARY_POINTER_OP(GreaterEqual, >=)

#undef CUJ_OVERLOAD_BINARY_POINTER_OP

#define CUJ_OVERLOAD_POINTER_ARITH_BOOL_OP(SYM)                                 \
    template<typename L, typename R>                                            \
    ArithmeticValue<bool> operator SYM(                                         \
        const PointerImpl<L> &lhs, const ArithmeticValue<R> &rhs)               \
    {                                                                           \
        return (lhs != nullptr) SYM rhs;                                        \
    }                                                                           \
    template<typename L, typename R>                                            \
    ArithmeticValue<bool> operator SYM(                                         \
        const ArithmeticValue<L> &lhs, const PointerImpl<R> &rhs)               \
    {                                                                           \
        return lhs SYM (rhs != nullptr);                                        \
    }                                                                           \
    template<typename L, typename R,                                            \
             typename = std::enable_if_t<std::is_arithmetic_v<R>>>              \
    ArithmeticValue<bool> operator SYM(const PointerImpl<L> &lhs, R rhs)        \
    {                                                                           \
        return lhs SYM create_literial(rhs);                                    \
    }                                                                           \
    template<typename L, typename R,                                            \
             typename = std::enable_if_t<std::is_arithmetic_v<L>>>              \
    ArithmeticValue<bool> operator SYM(L lhs, const PointerImpl<R> &rhs)        \
    {                                                                           \
        return create_literial(lhs) SYM rhs;                                    \
    }

CUJ_OVERLOAD_POINTER_ARITH_BOOL_OP(&&)
CUJ_OVERLOAD_POINTER_ARITH_BOOL_OP(||)

#undef CUJ_OVERLOAD_POINTER_ARITH_BOOL_OP

// pointer +-

template<typename L, typename R>
PointerImpl<L> operator+(const PointerImpl<L> &lhs, const ArithmeticValue<R> &rhs)
{
    return lhs.offset(rhs);
}

template<typename L, typename R>
PointerImpl<R> operator+(const ArithmeticValue<L> &lhs, const PointerImpl<R> &rhs)
{
    return rhs + lhs;
}

template<typename L, typename R,
         typename = std::enable_if_t<std::is_arithmetic_v<R>>>
PointerImpl<L> operator+(const PointerImpl<L> &lhs, R rhs)
{
    return lhs.offset(create_literial(rhs));
}

template<typename L, typename R,
         typename = std::enable_if_t<std::is_arithmetic_v<L>>>
PointerImpl<R> operator+(L lhs, const PointerImpl<R> &rhs)
{
    return rhs + lhs;
}

template<typename L, typename R>
PointerImpl<L> operator-(const PointerImpl<L> &lhs, const ArithmeticValue<R> &rhs)
{
    return lhs + (create_literial(R(0)) - rhs);
}

template<typename L, typename R,
         typename = std::enable_if_t<std::is_arithmetic_v<R>>>
PointerImpl<L> operator-(const PointerImpl<L> &lhs, R rhs)
{
    return lhs - create_literial(rhs);
}

template<typename T>
ArithmeticValue<int64_t> operator-(const PointerImpl<T> &lhs, const PointerImpl<T> &rhs)
{
    auto impl = newRC<InternalPointerDiff<T>>();
    impl->lhs = lhs.get_impl();
    impl->rhs = rhs.get_impl();
    return ArithmeticValue<int64_t>(std::move(impl));
}

// unary arithmetic

#define CUJ_OVERLOAD_UNARY_ARITHMETIC_OP(OP, SYM)                               \
template<typename I>                                                            \
auto operator SYM(const ArithmeticValue<I> &input)                              \
{                                                                               \
    using T = decltype(SYM std::declval<I>());                                  \
    auto impl = create_unary_operator<T, I>(                                    \
        ir::UnaryOp::Type::OP, input.get_impl());                               \
    return ArithmeticValue<T>(std::move(impl));                                 \
}

CUJ_OVERLOAD_UNARY_ARITHMETIC_OP(Neg, -)
CUJ_OVERLOAD_UNARY_ARITHMETIC_OP(Not, !)

#undef CUJ_OVERLOAD_UNARY_ARITHMETIC_OP

// unary pointer

template<typename T>
Value<bool> operator!(const PointerImpl<T> &ptr)
{
    return ptr == nullptr;
}

// cast

template<typename To, typename From>
ArithmeticValue<To> cast(const ArithmeticValue<From> &from)
{
    using TTo = typename detail::DeValueType<To>::Type;
    auto impl = newRC<InternalCastArithmeticValue<From, TTo>>();
    impl->from = from.get_impl();
    return ArithmeticValue<TTo>(std::move(impl));
}

template<typename To, typename From>
Pointer<To> ptr_cast(const PointerImpl<From> &from)
{
    using TTo = typename detail::DeValueType<To>::Type;
    auto impl = newRC<InternalCastPointerValue<From, TTo>>();
    impl->from = from.get_impl();
    return Pointer<TTo>(std::move(impl));
}

CUJ_NAMESPACE_END(cuj::ast)
