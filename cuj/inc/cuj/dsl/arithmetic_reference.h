#pragma once

#include <cuj/dsl/pointer.h>
#include <cuj/dsl/variable_forward.h>

CUJ_NAMESPACE_BEGIN(cuj::dsl)

template<typename T> requires std::is_arithmetic_v<T>
class ref<num<T>>
{
    ptr<num<T>> addr_;

    ref() = default;

public:

    using RawType = T;

    ref(const num<T> &var);

    ref(const ref &ref);

    ref(ref &&other) noexcept;

    ref &operator=(const ref &other);

    ref &operator=(const num<T> &other);

    template<typename U> requires is_cuj_arithmetic_v<U>
    U as() const;

    num<T> operator-() const;

    num<T> operator+(const num<T> &rhs) const;
    num<T> operator-(const num<T> &rhs) const;
    num<T> operator*(const num<T> &rhs) const;
    num<T> operator/(const num<T> &rhs) const;
    num<T> operator%(const num<T> &rhs) const;

    num<bool> operator==(const num<T> &rhs) const;
    num<bool> operator!=(const num<T> &rhs) const;
    num<bool> operator< (const num<T> &rhs) const;
    num<bool> operator<=(const num<T> &rhs) const;
    num<bool> operator> (const num<T> &rhs) const;
    num<bool> operator>=(const num<T> &rhs) const;

    num<T> operator>>(const num<T> &rhs) const;
    num<T> operator<<(const num<T> &rhs) const;

    num<T> operator&(const num<T> &rhs) const;
    num<T> operator|(const num<T> &rhs) const;
    num<T> operator^(const num<T> &rhs) const;

    num<T> operator~() const;

    ptr<num<T>> address() const;
    
    core::Load _load() const;

    static ref _from_ptr(const ptr<num<T>> &ptr);
};

template<typename T>
num<T> operator+(T lhs, const ref<num<T>> &rhs);
template<typename T>
num<T> operator-(T lhs, const ref<num<T>> &rhs);
template<typename T>
num<T> operator*(T lhs, const ref<num<T>> &rhs);
template<typename T>
num<T> operator/(T lhs, const ref<num<T>> &rhs);
template<typename T>
num<T> operator%(T lhs, const ref<num<T>> &rhs);

template<typename T> requires std::is_arithmetic_v<T>
num<bool> operator==(T lhs, const ref<num<T>> &rhs);
template<typename T> requires std::is_arithmetic_v<T>
num<bool> operator!=(T lhs, const ref<num<T>> &rhs);
template<typename T> requires std::is_arithmetic_v<T>
num<bool> operator<(T lhs, const ref<num<T>> &rhs);
template<typename T> requires std::is_arithmetic_v<T>
num<bool> operator<=(T lhs, const ref<num<T>> &rhs);
template<typename T> requires std::is_arithmetic_v<T>
num<bool> operator>(T lhs, const ref<num<T>> &rhs);
template<typename T> requires std::is_arithmetic_v<T>
num<bool> operator>=(T lhs, const ref<num<T>> &rhs);

inline num<bool> operator!(const ref<num<bool>> &val);

template<typename T> requires std::is_integral_v<T>
num<T> operator<<(T lhs, const ref<num<T>> &rhs);
template<typename T> requires std::is_integral_v<T> && !std::is_signed_v<T>
num<T> operator>>(T lhs, const ref<num<T>> &rhs);

template<typename T> requires std::is_integral_v<T>
num<T> operator&(T lhs, const ref<num<T>> &rhs);
template<typename T> requires std::is_integral_v<T>
num<T> operator|(T lhs, const ref<num<T>> &rhs);
template<typename T> requires std::is_integral_v<T>
num<T> operator^(T lhs, const ref<num<T>> &rhs);

CUJ_NAMESPACE_END(cuj::dsl)
