#pragma once

#include <cuj/core/expr.h>
#include <cuj/dsl/variable_forward.h>

CUJ_NAMESPACE_BEGIN(cuj::dsl)

template<typename T> requires std::is_arithmetic_v<T>
class num
{
    size_t alloc_index_;

public:

    using RawType = T;

    num();

    num(T immediate_value);

    template<typename U> requires !std::is_same_v<T, U>
    explicit num(const num<U> &other);
    
    template<typename U> requires !std::is_same_v<T, U>
    explicit num(const ref<num<U>> &other);

    num(const num &other);

    num(const ref<num<T>> &ref);

    num(num &&other) noexcept;

    num &operator=(const num &other);

    template<typename U> requires is_cuj_arithmetic_v<U>
    U as() const;

    num operator-() const;

    num operator+(const num &rhs) const;
    num operator-(const num &rhs) const;
    num operator*(const num &rhs) const;
    num operator/(const num &rhs) const;
    num operator%(const num &rhs) const;

    num<bool> operator==(const num &rhs) const;
    num<bool> operator!=(const num &rhs) const;
    num<bool> operator< (const num &rhs) const;
    num<bool> operator<=(const num &rhs) const;
    num<bool> operator> (const num &rhs) const;
    num<bool> operator>=(const num &rhs) const;

    num<bool> operator==(T rhs) const;
    num<bool> operator!=(T rhs) const;
    num<bool> operator< (T rhs) const;
    num<bool> operator<=(T rhs) const;
    num<bool> operator> (T rhs) const;
    num<bool> operator>=(T rhs) const;

    num operator>>(const num &rhs) const;
    num operator<<(const num &rhs) const;

    num operator&(const num &rhs) const;
    num operator|(const num &rhs) const;
    num operator^(const num &rhs) const;

    num operator~() const;

    ptr<num> address() const;

    static num _from_expr(core::Expr expr);

    core::Load _load() const;

    core::LocalAllocAddr _addr() const;
};

template<typename T>
num<T> operator+(T lhs, const num<T> &rhs);
template<typename T>
num<T> operator-(T lhs, const num<T> &rhs);
template<typename T>
num<T> operator*(T lhs, const num<T> &rhs);
template<typename T>
num<T> operator/(T lhs, const num<T> &rhs);
template<typename T>
num<T> operator%(T lhs, const num<T> &rhs);

template<typename T> requires std::is_arithmetic_v<T>
num<bool> operator==(T lhs, const num<T> &rhs);
template<typename T> requires std::is_arithmetic_v<T>
num<bool> operator!=(T lhs, const num<T> &rhs);
template<typename T> requires std::is_arithmetic_v<T>
num<bool> operator<(T lhs, const num<T> &rhs);
template<typename T> requires std::is_arithmetic_v<T>
num<bool> operator<=(T lhs, const num<T> &rhs);
template<typename T> requires std::is_arithmetic_v<T>
num<bool> operator>(T lhs, const num<T> &rhs);
template<typename T> requires std::is_arithmetic_v<T>
num<bool> operator>=(T lhs, const num<T> &rhs);

inline num<bool> operator!(const num<bool> &val);

template<typename T> requires std::is_integral_v<T>
num<T> operator<<(T lhs, const num<T> &rhs);
template<typename T> requires std::is_integral_v<T> && !std::is_signed_v<T>
num<T> operator>>(T lhs, const num<T> &rhs);

template<typename T> requires std::is_integral_v<T>
num<T> operator&(T lhs, const num<T> &rhs);
template<typename T> requires std::is_integral_v<T>
num<T> operator|(T lhs, const num<T> &rhs);
template<typename T> requires std::is_integral_v<T>
num<T> operator^(T lhs, const num<T> &rhs);

CUJ_NAMESPACE_END(cuj::dsl)
