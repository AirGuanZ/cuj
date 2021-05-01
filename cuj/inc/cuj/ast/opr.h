#pragma once

#include <cuj/ast/expr.h>

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

#define CUJ_OVERLOAD_BINARY_POINTER_OP(OP, SYM)                                 \
    template<typename L, typename R>                                            \
    Value<bool> operator SYM(const Pointer<L> &lhs, const Pointer<R> &rhs)      \
    {                                                                           \
        auto impl = create_binary_operator<bool, Pointer<L>, Pointer<R>>(       \
            ir::BinaryOp::Type::OP, lhs.get_impl(), rhs.get_impl());            \
        return Value<bool>(std::move(impl));                                    \
    }                                                                           \
    template<typename T>                                                        \
    Value<bool> operator SYM(const Pointer<T> &ptr, const std::nullptr_t &)     \
    {                                                                           \
        return ptr SYM Pointer<T>(nullptr);                                     \
    }                                                                           \
    template<typename T>                                                        \
    Value<bool> operator SYM(const std::nullptr_t &, const Pointer<T> &ptr)     \
    {                                                                           \
        return Pointer<T>(nullptr) SYM ptr;                                     \
    }

CUJ_OVERLOAD_BINARY_POINTER_OP(Equal,        ==)
CUJ_OVERLOAD_BINARY_POINTER_OP(NotEqual,     !=)
CUJ_OVERLOAD_BINARY_POINTER_OP(Less,         <)
CUJ_OVERLOAD_BINARY_POINTER_OP(LessEqual,    <=)
CUJ_OVERLOAD_BINARY_POINTER_OP(Greater,      >)
CUJ_OVERLOAD_BINARY_POINTER_OP(GreaterEqual, >=)

#undef CUJ_OVERLOAD_BINARY_POINTER_OP

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
Value<bool> operator!(const Pointer<T> &ptr)
{
    return ptr == nullptr;
}

// cast

template<typename To, typename From>
ArithmeticValue<To> cast(const ArithmeticValue<From> &from)
{
    auto impl = newRC<InternalCastArithmeticValue<From, To>>();
    impl->from = from.get_impl();
    return ArithmeticValue<To>(std::move(impl));
}

CUJ_NAMESPACE_END(cuj::ast)
