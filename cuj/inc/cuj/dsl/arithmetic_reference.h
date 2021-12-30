#pragma once

#include <cuj/dsl/pointer.h>
#include <cuj/dsl/variable_forward.h>

CUJ_NAMESPACE_BEGIN(cuj::dsl)

template<typename T> requires std::is_arithmetic_v<T>
class ref<Arithmetic<T>>
{
    Pointer<Arithmetic<T>> addr_;

    ref() = default;

public:

    using RawType = T;

    ref(const Arithmetic<T> &var);

    ref(const ref &ref);

    ref(ref &&other) noexcept;

    ref &operator=(const ref &other);

    ref &operator=(const Arithmetic<T> &other);

    template<typename U> requires is_cuj_arithmetic_v<U>
    U as() const;

    Arithmetic<T> operator-() const;

    Arithmetic<T> operator+(const Arithmetic<T> &rhs) const;
    Arithmetic<T> operator-(const Arithmetic<T> &rhs) const;
    Arithmetic<T> operator*(const Arithmetic<T> &rhs) const;
    Arithmetic<T> operator/(const Arithmetic<T> &rhs) const;
    Arithmetic<T> operator%(const Arithmetic<T> &rhs) const;

    Arithmetic<bool> operator==(const Arithmetic<T> &rhs) const;
    Arithmetic<bool> operator!=(const Arithmetic<T> &rhs) const;
    Arithmetic<bool> operator< (const Arithmetic<T> &rhs) const;
    Arithmetic<bool> operator<=(const Arithmetic<T> &rhs) const;
    Arithmetic<bool> operator> (const Arithmetic<T> &rhs) const;
    Arithmetic<bool> operator>=(const Arithmetic<T> &rhs) const;

    Arithmetic<T> operator>>(const Arithmetic<T> &rhs) const;
    Arithmetic<T> operator<<(const Arithmetic<T> &rhs) const;

    Arithmetic<T> operator&(const Arithmetic<T> &rhs) const;
    Arithmetic<T> operator|(const Arithmetic<T> &rhs) const;
    Arithmetic<T> operator^(const Arithmetic<T> &rhs) const;

    Arithmetic<T> operator~() const;

    Pointer<Arithmetic<T>> address() const;
    
    core::Load _load() const;

    static ref _from_ptr(const Pointer<Arithmetic<T>> &ptr);
};

template<typename T>
Arithmetic<T> operator+(T lhs, const ref<Arithmetic<T>> &rhs);
template<typename T>
Arithmetic<T> operator-(T lhs, const ref<Arithmetic<T>> &rhs);
template<typename T>
Arithmetic<T> operator*(T lhs, const ref<Arithmetic<T>> &rhs);
template<typename T>
Arithmetic<T> operator/(T lhs, const ref<Arithmetic<T>> &rhs);
template<typename T>
Arithmetic<T> operator%(T lhs, const ref<Arithmetic<T>> &rhs);

template<typename T> requires std::is_arithmetic_v<T>
Arithmetic<bool> operator==(T lhs, const ref<Arithmetic<T>> &rhs);
template<typename T> requires std::is_arithmetic_v<T>
Arithmetic<bool> operator!=(T lhs, const ref<Arithmetic<T>> &rhs);
template<typename T> requires std::is_arithmetic_v<T>
Arithmetic<bool> operator<(T lhs, const ref<Arithmetic<T>> &rhs);
template<typename T> requires std::is_arithmetic_v<T>
Arithmetic<bool> operator<=(T lhs, const ref<Arithmetic<T>> &rhs);
template<typename T> requires std::is_arithmetic_v<T>
Arithmetic<bool> operator>(T lhs, const ref<Arithmetic<T>> &rhs);
template<typename T> requires std::is_arithmetic_v<T>
Arithmetic<bool> operator>=(T lhs, const ref<Arithmetic<T>> &rhs);

inline Arithmetic<bool> operator!(const ref<Arithmetic<bool>> &val);

template<typename T> requires std::is_integral_v<T>
Arithmetic<T> operator<<(T lhs, const ref<Arithmetic<T>> &rhs);
template<typename T> requires std::is_integral_v<T> && (!std::is_signed_v<T>)
Arithmetic<T> operator>>(T lhs, const ref<Arithmetic<T>> &rhs);

template<typename T> requires std::is_integral_v<T>
Arithmetic<T> operator&(T lhs, const ref<Arithmetic<T>> &rhs);
template<typename T> requires std::is_integral_v<T>
Arithmetic<T> operator|(T lhs, const ref<Arithmetic<T>> &rhs);
template<typename T> requires std::is_integral_v<T>
Arithmetic<T> operator^(T lhs, const ref<Arithmetic<T>> &rhs);

CUJ_NAMESPACE_END(cuj::dsl)
