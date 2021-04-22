#pragma once

#include <cuj/ast/expr.h>

CUJ_NAMESPACE_BEGIN(cuj::ast)

// binary

#define CUJ_OVERLOAD_BINARY_OP(OP, SYM)                                         \
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

CUJ_OVERLOAD_BINARY_OP(Add, +)
CUJ_OVERLOAD_BINARY_OP(Sub, -)
CUJ_OVERLOAD_BINARY_OP(Mul, *)
CUJ_OVERLOAD_BINARY_OP(Div, /)

CUJ_OVERLOAD_BINARY_OP(And, &&)
CUJ_OVERLOAD_BINARY_OP(Or,  ||)
CUJ_OVERLOAD_BINARY_OP(XOr, ^)

CUJ_OVERLOAD_BINARY_OP(Equal, ==)
CUJ_OVERLOAD_BINARY_OP(NotEqual, !=)
CUJ_OVERLOAD_BINARY_OP(Less, <)
CUJ_OVERLOAD_BINARY_OP(LessEqual, <=)
CUJ_OVERLOAD_BINARY_OP(Greater, >)
CUJ_OVERLOAD_BINARY_OP(GreaterEqual, >=)

#undef CUJ_OVERLOAD_BINARY_OP

// unary

#define CUJ_OVERLOAD_UNARY_OP(OP, SYM)                                          \
template<typename I>                                                            \
auto operator SYM(const ArithmeticValue<I> &input)                              \
{                                                                               \
    using T = decltype(SYM std::declval<I>());                                  \
    auto impl = create_unary_operator<T, I>(                                    \
        ir::UnaryOp::Type::OP, input.get_impl());                               \
    return ArithmeticValue<T>(std::move(impl));                                 \
}

CUJ_OVERLOAD_UNARY_OP(Neg, -)
CUJ_OVERLOAD_UNARY_OP(Not, !)

// cast

template<typename To, typename From>
ArithmeticValue<To> cast(const ArithmeticValue<From> &from)
{
    auto impl = newRC<InternalCastArithmeticValue<From, To>>();
    impl->from = from.get_impl();
    return ArithmeticValue<To>(std::move(impl));
}

CUJ_NAMESPACE_END(cuj::ast)
