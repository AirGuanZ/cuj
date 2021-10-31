#pragma once

#include <cuj/ast/ast.h>
#include <cuj/builtin/math/basic.h>

CUJ_NAMESPACE_BEGIN(cuj::builtin::math)

template<typename T>
struct Vec1Host
{
    T x;
};

#define CUJ_DEFINE_VEC1(BASE, NAME, COMP)                                       \
CUJ_MAKE_PROXY_BEGIN(BASE, NAME, x)                                             \
    CUJ_PROXY_CONSTRUCTOR(NAME)                                                 \
    {                                                                           \
        x = 0;                                                                  \
    }                                                                           \
    explicit CUJ_PROXY_CONSTRUCTOR(NAME, COMP _x)                               \
    {                                                                           \
        x = _x;                                                                 \
    }                                                                           \
    auto operator[](const Value<size_t> &idx) const                             \
    {                                                                           \
        return Value<COMP>((x.address() + idx).deref().get_impl());             \
    }                                                                           \
    auto length_square() const { return x * x; }                                \
    auto length() const { return sqrt(length_square()); }                       \
    auto min_elem() const { return x; }                                         \
    auto max_elem() const { return x; }                                         \
    auto normalize() const                                                      \
    {                                                                           \
        return NAME(x / length());                                              \
    }                                                                           \
CUJ_MAKE_PROXY_END

CUJ_DEFINE_VEC1(Vec1Host<int>,    Vec1i, i32)
CUJ_DEFINE_VEC1(Vec1Host<float>,  Vec1f, f32)
CUJ_DEFINE_VEC1(Vec1Host<double>, Vec1d, f64)

#undef CUJ_DEFINE_VEC1

template<typename T> struct Vec1Aux { };
template<> struct Vec1Aux<int>    { using Type = Vec1i; };
template<> struct Vec1Aux<float>  { using Type = Vec1f; };
template<> struct Vec1Aux<double> { using Type = Vec1d; };

template<typename T>
using Vec1 = typename Vec1Aux<T>::Type;

template<typename T>
Vec1<T> make_vec1(const Value<T> &v)
{
    return Vec1<T>(v);
}

template<typename T>
Vec1<T> make_vec1()
{
    return make_vec1<T>(0);
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

#define CUJ_DEFINE_VEC1_OPERATORS(T)                                                \
inline Vec1<T> operator-(const Vec1<T> &v)                                             \
{                                                                               \
    return make_vec1<T>(-v.x);                                                  \
}                                                                               \
inline Vec1<T> operator+(const Vec1<T> &lhs, const Vec1<T> &rhs)                       \
{                                                                               \
    return make_vec1<T>(lhs.x + rhs.x);                                         \
}                                                                               \
inline Vec1<T> operator-(const Vec1<T> &lhs, const Vec1<T> &rhs)                       \
{                                                                               \
    return make_vec1<T>(lhs.x - rhs.x);                                         \
}                                                                               \
inline Vec1<T> operator*(const Vec1<T> &lhs, const Vec1<T> &rhs)                       \
{                                                                               \
    return make_vec1<T>(lhs.x * rhs.x);                                         \
}                                                                               \
inline Vec1<T> operator/(const Vec1<T> &lhs, const Vec1<T> &rhs)                       \
{                                                                               \
    return make_vec1<T>(lhs.x / rhs.x);                                         \
}                                                                               \
inline Vec1<T> operator+(const Vec1<T> &lhs, const Value<T> &rhs)                      \
{                                                                               \
    return make_vec1<T>(lhs.x + rhs);                                           \
}                                                                               \
inline Vec1<T> operator-(const Vec1<T> &lhs, const Value<T> &rhs)                      \
{                                                                               \
    return make_vec1<T>(lhs.x - rhs);                                           \
}                                                                               \
inline Vec1<T> operator*(const Vec1<T> &lhs, const Value<T> &rhs)                      \
{                                                                               \
    return make_vec1<T>(lhs.x * rhs);                                           \
}                                                                               \
inline Vec1<T> operator/(const Vec1<T> &lhs, const Value<T> &rhs)                      \
{                                                                               \
    return make_vec1<T>(lhs.x / rhs);                                           \
}                                                                               \
inline Vec1<T> operator+(const Value<T> &lhs, const Vec1<T> &rhs)                      \
{                                                                               \
    return make_vec1<T>(lhs + rhs.x);                                           \
}                                                                               \
inline Vec1<T> operator-(const Value<T> &lhs, const Vec1<T> &rhs)                      \
{                                                                               \
    return make_vec1<T>(lhs - rhs.x);                                           \
}                                                                               \
inline Vec1<T> operator*(const Value<T> &lhs, const Vec1<T> &rhs)                      \
{                                                                               \
    return make_vec1<T>(lhs * rhs.x);                                           \
}                                                                               \
inline Vec1<T> operator/(const Value<T> &lhs, const Vec1<T> &rhs)                      \
{                                                                               \
    return make_vec1<T>(lhs / rhs.x);                                           \
}                                                                               \
inline Value<T> dot(const Vec1<T> &a, const Vec1<T> &b)                                \
{                                                                               \
    return a.x * b.x;                                                           \
}                                                                               \
inline Value<T> cos(const Vec1<T> &a, const Vec1<T> &b)                                \
{                                                                               \
    return dot(a, b) / (a.length() * b.length());                               \
}

CUJ_DEFINE_VEC1_OPERATORS(int)
CUJ_DEFINE_VEC1_OPERATORS(float)
CUJ_DEFINE_VEC1_OPERATORS(double)

#undef CUJ_DEFINE_VEC1_OPERATORS

CUJ_NAMESPACE_END(cuj::builtin::math)
