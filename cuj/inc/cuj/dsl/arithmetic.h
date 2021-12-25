#pragma once

#include <cuj/core/expr.h>
#include <cuj/dsl/variable_forward.h>

CUJ_NAMESPACE_BEGIN(cuj::dsl)

template<typename T> requires std::is_arithmetic_v<T>
class Arithmetic
{
    size_t alloc_index_;

public:

    using RawType = T;

    Arithmetic();

    Arithmetic(T immediate_value);

    template<typename U> requires !std::is_same_v<T, U>
    explicit Arithmetic(const Arithmetic<U> &other);
    
    template<typename U> requires !std::is_same_v<T, U>
    explicit Arithmetic(const ref<Arithmetic<U>> &other);

    Arithmetic(const Arithmetic &other);

    Arithmetic(const ref<Arithmetic<T>> &ref);

    Arithmetic(Arithmetic &&other) noexcept;

    Arithmetic &operator=(const Arithmetic &other);

    template<typename U> requires is_cuj_arithmetic_v<U>
    U as() const;

    Arithmetic operator-() const;

    Arithmetic operator+(const Arithmetic &rhs) const;
    Arithmetic operator-(const Arithmetic &rhs) const;
    Arithmetic operator*(const Arithmetic &rhs) const;
    Arithmetic operator/(const Arithmetic &rhs) const;
    Arithmetic operator%(const Arithmetic &rhs) const;

    Arithmetic<bool> operator==(const Arithmetic &rhs) const;
    Arithmetic<bool> operator!=(const Arithmetic &rhs) const;
    Arithmetic<bool> operator< (const Arithmetic &rhs) const;
    Arithmetic<bool> operator<=(const Arithmetic &rhs) const;
    Arithmetic<bool> operator> (const Arithmetic &rhs) const;
    Arithmetic<bool> operator>=(const Arithmetic &rhs) const;

    Arithmetic<bool> operator==(T rhs) const;
    Arithmetic<bool> operator!=(T rhs) const;
    Arithmetic<bool> operator< (T rhs) const;
    Arithmetic<bool> operator<=(T rhs) const;
    Arithmetic<bool> operator> (T rhs) const;
    Arithmetic<bool> operator>=(T rhs) const;

    Arithmetic operator>>(const Arithmetic &rhs) const;
    Arithmetic operator<<(const Arithmetic &rhs) const;

    Arithmetic operator&(const Arithmetic &rhs) const;
    Arithmetic operator|(const Arithmetic &rhs) const;
    Arithmetic operator^(const Arithmetic &rhs) const;

    Arithmetic operator~() const;

    Pointer<Arithmetic> address() const;

    static Arithmetic _from_expr(core::Expr expr);

    core::Load _load() const;

    core::LocalAllocAddr _addr() const;
};

template<typename T>
Arithmetic<T> operator+(T lhs, const Arithmetic<T> &rhs);
template<typename T>
Arithmetic<T> operator-(T lhs, const Arithmetic<T> &rhs);
template<typename T>
Arithmetic<T> operator*(T lhs, const Arithmetic<T> &rhs);
template<typename T>
Arithmetic<T> operator/(T lhs, const Arithmetic<T> &rhs);
template<typename T>
Arithmetic<T> operator%(T lhs, const Arithmetic<T> &rhs);

template<typename T> requires std::is_arithmetic_v<T>
Arithmetic<bool> operator==(T lhs, const Arithmetic<T> &rhs);
template<typename T> requires std::is_arithmetic_v<T>
Arithmetic<bool> operator!=(T lhs, const Arithmetic<T> &rhs);
template<typename T> requires std::is_arithmetic_v<T>
Arithmetic<bool> operator<(T lhs, const Arithmetic<T> &rhs);
template<typename T> requires std::is_arithmetic_v<T>
Arithmetic<bool> operator<=(T lhs, const Arithmetic<T> &rhs);
template<typename T> requires std::is_arithmetic_v<T>
Arithmetic<bool> operator>(T lhs, const Arithmetic<T> &rhs);
template<typename T> requires std::is_arithmetic_v<T>
Arithmetic<bool> operator>=(T lhs, const Arithmetic<T> &rhs);

inline Arithmetic<bool> operator!(const Arithmetic<bool> &rhs);

template<typename T> requires std::is_integral_v<T>
Arithmetic<T> operator<<(T lhs, const Arithmetic<T> &rhs);
template<typename T> requires std::is_integral_v<T> && !std::is_signed_v<T>
Arithmetic<T> operator>>(T lhs, const Arithmetic<T> &rhs);

template<typename T> requires std::is_integral_v<T>
Arithmetic<T> operator&(T lhs, const Arithmetic<T> &rhs);
template<typename T> requires std::is_integral_v<T>
Arithmetic<T> operator|(T lhs, const Arithmetic<T> &rhs);
template<typename T> requires std::is_integral_v<T>
Arithmetic<T> operator^(T lhs, const Arithmetic<T> &rhs);

CUJ_NAMESPACE_END(cuj::dsl)
