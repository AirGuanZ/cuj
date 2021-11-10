#pragma once

#include <cstring>

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
CUJ_OVERLOAD_BINARY_ARITHMETIC_OP(Mod, %)

CUJ_OVERLOAD_BINARY_ARITHMETIC_OP(And, &&)
CUJ_OVERLOAD_BINARY_ARITHMETIC_OP(Or,  ||)

CUJ_OVERLOAD_BINARY_ARITHMETIC_OP(BitwiseAnd, &)
CUJ_OVERLOAD_BINARY_ARITHMETIC_OP(BitwiseOr,  |)
CUJ_OVERLOAD_BINARY_ARITHMETIC_OP(BitwiseXOr, ^)

CUJ_OVERLOAD_BINARY_ARITHMETIC_OP(LeftShift,  <<)
CUJ_OVERLOAD_BINARY_ARITHMETIC_OP(RightShift, >>)

CUJ_OVERLOAD_BINARY_ARITHMETIC_OP(Equal,        ==)
CUJ_OVERLOAD_BINARY_ARITHMETIC_OP(NotEqual,     !=)
CUJ_OVERLOAD_BINARY_ARITHMETIC_OP(Less,         <)
CUJ_OVERLOAD_BINARY_ARITHMETIC_OP(LessEqual,    <=)
CUJ_OVERLOAD_BINARY_ARITHMETIC_OP(Greater,      >)
CUJ_OVERLOAD_BINARY_ARITHMETIC_OP(GreaterEqual, >=)

#undef CUJ_OVERLOAD_BINARY_ARITHMETIC_OP

// binary pointer

#define CUJ_OVERLOAD_BINARY_POINTER_OP(OP, SYM)                                       \
    template<typename L, typename R>                                                  \
    Value<bool> operator SYM(const PointerImpl<L> &lhs, const PointerImpl<R> &rhs)    \
    {                                                                                 \
        auto impl = create_binary_operator<bool, PointerImpl<L>, PointerImpl<R>>(     \
            ir::BinaryOp::Type::OP, lhs.get_impl(), rhs.get_impl());                  \
        return Value<bool>(std::move(impl));                                          \
    }                                                                                 \
    template<typename T>                                                              \
    Value<bool> operator SYM(const PointerImpl<T> &ptr, const std::nullptr_t &)       \
    {                                                                                 \
        return ptr SYM PointerImpl<T>(nullptr);                                       \
    }                                                                                 \
    template<typename T>                                                              \
    Value<bool> operator SYM(const std::nullptr_t &, const PointerImpl<T> &ptr)       \
    {                                                                                 \
        return PointerImpl<T>(nullptr) SYM ptr;                                       \
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
    Value<bool> operator SYM(                                                   \
        const PointerImpl<L> &lhs, const ArithmeticValue<R> &rhs)               \
    {                                                                           \
        return (lhs != nullptr) SYM rhs;                                        \
    }                                                                           \
    template<typename L, typename R>                                            \
    Value<bool> operator SYM(                                                   \
        const ArithmeticValue<L> &lhs, const PointerImpl<R> &rhs)               \
    {                                                                           \
        return lhs SYM (rhs != nullptr);                                        \
    }                                                                           \
    template<typename L, typename R,                                            \
             typename = std::enable_if_t<std::is_arithmetic_v<R>>>              \
    Value<bool> operator SYM(const PointerImpl<L> &lhs, R rhs)                  \
    {                                                                           \
        return lhs SYM create_literial(rhs);                                    \
    }                                                                           \
    template<typename L, typename R,                                            \
             typename = std::enable_if_t<std::is_arithmetic_v<L>>>              \
    Value<bool> operator SYM(L lhs, const PointerImpl<R> &rhs)                  \
    {                                                                           \
        return create_literial(lhs) SYM rhs;                                    \
    }

CUJ_OVERLOAD_POINTER_ARITH_BOOL_OP(&&)
CUJ_OVERLOAD_POINTER_ARITH_BOOL_OP(||)

#undef CUJ_OVERLOAD_POINTER_ARITH_BOOL_OP

// pointer +-

template<typename L, typename R>
auto operator+(const PointerImpl<L> &lhs, const ArithmeticValue<R> &rhs)
{
    return lhs.offset(rhs);
}

template<typename L, typename R>
auto operator+(const ArithmeticValue<L> &lhs, const PointerImpl<R> &rhs)
{
    return rhs + lhs;
}

template<typename L, typename R,
         typename = std::enable_if_t<std::is_arithmetic_v<R>>>
auto operator+(const PointerImpl<L> &lhs, R rhs)
{
    return lhs.offset(create_literial(rhs));
}

template<typename L, typename R,
         typename = std::enable_if_t<std::is_arithmetic_v<L>>>
auto operator+(L lhs, const PointerImpl<R> &rhs)
{
    return rhs + lhs;
}

template<typename L, typename R>
auto operator-(const PointerImpl<L> &lhs, const ArithmeticValue<R> &rhs)
{
    return lhs + (create_literial(R(0)) - rhs);
}

template<typename L, typename R,
         typename = std::enable_if_t<std::is_arithmetic_v<R>>>
auto operator-(const PointerImpl<L> &lhs, R rhs)
{
    return lhs - create_literial(rhs);
}

template<typename T>
auto operator-(const PointerImpl<T> &lhs, const PointerImpl<T> &rhs)
{
    auto impl = newRC<InternalPointerDiff<T>>();
    impl->lhs = lhs.get_impl();
    impl->rhs = rhs.get_impl();
    return Value<int64_t>(std::move(impl));
}

// unary arithmetic

#define CUJ_OVERLOAD_UNARY_ARITHMETIC_OP(OP, SYM)                               \
template<typename I>                                                            \
auto operator SYM(const ArithmeticValue<I> &input)                              \
{                                                                               \
    using T = decltype(SYM std::declval<I>());                                  \
    auto impl = create_unary_operator<T, I>(                                    \
        ir::UnaryOp::Type::OP, input.get_impl());                               \
    return Value<T>(std::move(impl));                                           \
}

CUJ_OVERLOAD_UNARY_ARITHMETIC_OP(Neg,        -)
CUJ_OVERLOAD_UNARY_ARITHMETIC_OP(Not,        !)
CUJ_OVERLOAD_UNARY_ARITHMETIC_OP(BitwiseNot, ~)

#undef CUJ_OVERLOAD_UNARY_ARITHMETIC_OP

// unary pointer

template<typename T>
auto operator!(const PointerImpl<T> &ptr)
{
    return ptr == nullptr;
}

// cast

template<typename To, typename From>
auto cast(const ArithmeticValue<From> &from)
{
    using TTo = deval_t<to_cuj_t<To>>;
    auto impl = newRC<InternalCastArithmeticValue<From, TTo>>();
    impl->from = from.get_impl();
    return Value<TTo>(std::move(impl));
}

template<typename To, typename From>
auto ptr_cast(const PointerImpl<From> &from)
{
    using TTo = deval_t<to_cuj_t<To>>;
    auto impl = newRC<InternalCastPointerValue<From, TTo>>();
    impl->from = from.get_impl();
    return PointerImpl<TTo>(std::move(impl));
}

template<typename T>
auto uint_to_ptr(const ArithmeticValue<size_t> &uint)
{
    using TTo = deval_t<to_cuj_t<T>>;
    auto impl = newRC<InternalUIntToPointer<TTo>>();
    impl->value = uint.get_impl();
    return PointerImpl<TTo>(std::move(impl));
}

template<typename T>
ArithmeticValue<size_t> ptr_to_uint(const PointerImpl<T> &ptr)
{
    auto impl = newRC<InternalPointerToUInt<T>>();
    impl->pointer = ptr.get_impl();
    return ArithmeticValue<size_t>(std::move(impl));
}

template<typename T>
auto ptr_literial(T *raw)
{
    return uint_to_ptr<T>(reinterpret_cast<size_t>(raw));
}

template<typename T>
auto ptr_literial(const T *raw)
{
    return uint_to_ptr<T>(reinterpret_cast<size_t>(raw));
}

// const data

template<typename T>
auto const_data(const void *data, size_t bytes)
{
    using TT = deval_t<to_cuj_t<T>>;
    auto impl = newRC<InternalConstData<TT>>();
    impl->bytes.resize(bytes);
    std::memcpy(impl->bytes.data(), data, bytes);
    return PointerImpl<TT>(impl);
}

template<typename T, typename U, size_t N>
auto const_data(const U(&data)[N])
{
    return ast::const_data<T>(&data[0], sizeof(U) * N);
}

template<typename T>
auto const_data(const std::vector<T> &data)
{
    return const_data<T>(data.data(), data.size() * sizeof(T));
}

// string literial

inline auto string_literial(std::string_view s)
{
    std::vector<char> data(s.size() + 1);
    std::memcpy(data.data(), s.data(), s.size());
    return const_data<char>(data.data(), data.size());
}

inline auto operator ""_cuj(const char *s, size_t size)
{
    return string_literial(std::string_view(s, size));
}

// select

template<typename T, typename C>
ArithmeticValue<T> select(
    const ArithmeticValue<C> &cond,
    const ArithmeticValue<T> &true_val,
    const ArithmeticValue<T> &false_val);

CUJ_NAMESPACE_END(cuj::ast)
