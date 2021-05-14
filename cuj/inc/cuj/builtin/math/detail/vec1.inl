#pragma once

#include <cuj/builtin/math/basic.h>
#include <cuj/builtin/math/vec1.h>

CUJ_NAMESPACE_BEGIN(cuj::builtin::math)

template<typename T>
Vec1Impl<T>::Vec1Impl(ClassAddress addr)
    : Vec1Impl(std::move(addr), 0)
{
    
}

template<typename T>
Vec1Impl<T>::Vec1Impl(ClassAddress addr, const Value<T> &v)
    : ClassBase<Vec1Impl>(std::move(addr))
{
    x = v;
}

template<typename T>
Vec1Impl<T>::Vec1Impl(ClassAddress addr, const ClassValue<Vec1Impl> &other)
    : Vec1Impl(std::move(addr), other->x)
{
    
}

template<typename T>
Value<T> Vec1Impl<T>::length_square() const
{
    return x * x;
}

template<typename T>
Value<T> Vec1Impl<T>::length() const
{
    static_assert(std::is_floating_point_v<T>);
    return math::sqrt(length_square());
}

template<typename T>
Value<T> Vec1Impl<T>::min_elem() const
{
    return x;
}

template<typename T>
Value<T> Vec1Impl<T>::max_elem() const
{
    return x;
}

template<typename T>
ClassValue<Vec1Impl<T>> Vec1Impl<T>::normalize() const
{
    static_assert(std::is_floating_point_v<T>);
    Value<T> inv_len = T(1) / length();
    return make_vec1<T>(x * inv_len);
}

template<typename T>
Value<T> Vec1Impl<T>::elem(const ArithmeticValue<size_t> &i) const
{
    return elem_addr(i).deref();
}

template<typename T>
Pointer<T> Vec1Impl<T>::elem_addr(const ArithmeticValue<size_t> &i) const
{
    return x.address() + i;
}

template<typename T>
Vec1<T> make_vec1()
{
    return make_vec1<T>(0);
}

template<typename T>
Vec1<T> make_vec1(const Value<T> &v)
{
    return Vec1<T>(v);
}

inline Vec1f make_vec1f()
{
    return make_vec1<float>();
}

inline Vec1f make_vec1f(const f32 &v)
{
    return make_vec1<float>(v);
}

inline Vec1d make_vec1d()
{
    return make_vec1<double>();
}

inline Vec1d make_vec1d(const f64 &v)
{
    return make_vec1<double>(v);
}

inline Vec1i make_vec1i()
{
    return make_vec1<int>();
}

inline Vec1i make_vec1i(const i32 &v)
{
    return make_vec1<int>(v);
}

template<typename T>
Vec1<T> operator-(const Vec1<T> &v)
{
    return make_vec1<T>(-v->x);
}

template<typename T>
Vec1<T> operator+(const Vec1<T> &lhs, const Vec1<T> &rhs)
{
    return make_vec1<T>(lhs->x + rhs->x);
}

template<typename T>
Vec1<T> operator-(const Vec1<T> &lhs, const Vec1<T> &rhs)
{
    return make_vec1<T>(lhs->x - rhs->x);
}

template<typename T>
Vec1<T> operator*(const Vec1<T> &lhs, const Vec1<T> &rhs)
{
    return make_vec1<T>(lhs->x * rhs->x);
}

template<typename T>
Vec1<T> operator/(const Vec1<T> &lhs, const Vec1<T> &rhs)
{
    return make_vec1<T>(lhs->x / rhs->x);
}

template<typename T>
Vec1<T> operator+(const Vec1<T> &lhs, const Value<T> &rhs)
{
    return make_vec1<T>(lhs->x + rhs);
}

template<typename T>
Vec1<T> operator-(const Vec1<T> &lhs, const Value<T> &rhs)
{
    return make_vec1<T>(lhs->x - rhs);
}

template<typename T>
Vec1<T> operator*(const Vec1<T> &lhs, const Value<T> &rhs)
{
    return make_vec1<T>(lhs->x * rhs);
}

template<typename T>
Vec1<T> operator/(const Vec1<T> &lhs, const Value<T> &rhs)
{
    return make_vec1<T>(lhs->x / rhs);
}

template<typename T>
Vec1<T> operator+(const Value<T> &lhs, const Vec1<T> &rhs)
{
    return make_vec1<T>(lhs + rhs->x);
}

template<typename T>
Vec1<T> operator-(const Value<T> &lhs, const Vec1<T> &rhs)
{
    return make_vec1<T>(lhs - rhs->x);
}

template<typename T>
Vec1<T> operator*(const Value<T> &lhs, const Vec1<T> &rhs)
{
    return make_vec1<T>(lhs * rhs->x);
}

template<typename T>
Vec1<T> operator/(const Value<T> &lhs, const Vec1<T> &rhs)
{
    return make_vec1<T>(lhs / rhs->x);
}

template<typename T>
Value<T> dot(const Vec1<T> &a, const Vec1<T> &b)
{
    return a->x * b->x;
}

template<typename T>
Value<T> cos(const Vec1<T> &a, const Vec1<T> &b)
{
    return dot(a, b) / (a->length() * b->length());
}

CUJ_NAMESPACE_END(cuj::builtin::math)
