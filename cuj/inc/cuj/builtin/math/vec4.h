#pragma once

#include <cuj/ast/ast.h>

CUJ_NAMESPACE_BEGIN(cuj::builtin::math)

template<typename T>
struct Vec4Host
{
    T x, y, z, w;
};

#define CUJ_DEFINE_VEC4(BASE, NAME, COMP)                                       \
CUJ_MAKE_PROXY_BEGIN(BASE, NAME, x, y, z, w)                                    \
    CUJ_PROXY_CONSTRUCTOR(NAME)                                                 \
    {                                                                           \
        x = 0; y = 0; z = 0; w = 0;                                             \
    }                                                                           \
    explicit CUJ_PROXY_CONSTRUCTOR(NAME, COMP v)                                \
    {                                                                           \
        x = v; y = v; z = v; w = v;                                             \
    }                                                                           \
    CUJ_PROXY_CONSTRUCTOR(NAME, COMP _x, COMP _y, COMP _z, COMP _w)             \
    {                                                                           \
        x = _x; y = _y; z = _z; w = _w;                                         \
    }                                                                           \
    auto operator[](const Value<size_t> &idx) const                             \
    {                                                                           \
        return Value<COMP>((x.address() + idx).deref().get_impl());             \
    }                                                                           \
    auto length_square() const { return x * x + y * y + z * z + w * w; }        \
    auto length() const { return sqrt(length_square()); }                       \
    auto min_elem() const { return min(min(x, y), min(z, w)); }                 \
    auto max_elem() const { return max(max(x, y), max(z, w)); }                 \
    auto normalize() const                                                      \
    {                                                                           \
        auto f = 1 / length();                                                  \
        return NAME(f * x, f * y, f * z, f * w);                                \
    }                                                                           \
CUJ_MAKE_PROXY_END

CUJ_DEFINE_VEC4(Vec4Host<int>,    Vec4i, i32)
CUJ_DEFINE_VEC4(Vec4Host<float>,  Vec4f, f32)
CUJ_DEFINE_VEC4(Vec4Host<double>, Vec4d, f64)

#undef CUJ_DEFINE_VEC4

template<typename T> struct Vec4Aux { };
template<> struct Vec4Aux<int>    { using Type = Vec4i; };
template<> struct Vec4Aux<float>  { using Type = Vec4f; };
template<> struct Vec4Aux<double> { using Type = Vec4d; };

template<typename T>
using Vec4 = typename Vec4Aux<T>::Type;

template<typename T>
Vec4<T> make_vec4()
{
    return make_vec4<T>(0, 0, 0, 0);
}

template<typename T>
Vec4<T> make_vec4(const Value<T> &v)
{
    return make_vec4<T>(v, v, v, v);
}

template<typename T>
Vec4<T> make_vec4(
    const Value<T> &x, const Value<T> &y, const Value<T> &z, const Value<T> &w)
{
    return Vec4<T>(x, y, z, w);
}

inline Vec4f make_vec4f()
{
    return make_vec4<float>();
}

inline Vec4f make_vec4f(const f32 &v)
{
    return make_vec4<float>(v);
}

inline Vec4f make_vec4f(
    const f32 &x, const f32 &y, const f32 &z, const f32 &w)
{
    return make_vec4<float>(x, y, z, w);
}

inline Vec4d make_vec4d()
{
    return make_vec4<double>();
}

inline Vec4d make_vec4d(const f64 &v)
{
    return make_vec4<double>(v);
}

inline Vec4d make_vec4d(
    const f64 &x, const f64 &y, const f64 &z, const f64 &w)
{
    return make_vec4<double>(x, y, z, w);
}

inline Vec4i make_vec4i()
{
    return make_vec4<int>();
}

inline Vec4i make_vec4i(const i32 &v)
{
    return make_vec4<int>(v);
}

inline Vec4i make_vec4i(
    const i32 &x, const i32 &y, const i32 &z, const i32 &w)
{
    return make_vec4<int>(x, y, z, w);
}

#define CUJ_DEFINE_VEC4_OPERATORS(T)                                            \
inline Vec4<T> operator-(const Vec4<T> &v)                                             \
{                                                                               \
    return make_vec4<T>(-v.x, -v.y, -v.z, -v.w);                                \
}                                                                               \
inline Vec4<T> operator+(const Vec4<T> &lhs, const Vec4<T> &rhs)                       \
{                                                                               \
    return make_vec4<T>(                                                        \
        lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z, lhs.w + rhs.w);            \
}                                                                               \
inline Vec4<T> operator-(const Vec4<T> &lhs, const Vec4<T> &rhs)                       \
{                                                                               \
    return make_vec4<T>(                                                        \
        lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z, lhs.w - rhs.w);            \
}                                                                               \
inline Vec4<T> operator*(const Vec4<T> &lhs, const Vec4<T> &rhs)                       \
{                                                                               \
    return make_vec4<T>(                                                        \
        lhs.x * rhs.x, lhs.y * rhs.y, lhs.z * rhs.z, lhs.w * rhs.w);            \
}                                                                               \
inline Vec4<T> operator/(const Vec4<T> &lhs, const Vec4<T> &rhs)                       \
{                                                                               \
    return make_vec4<T>(                                                        \
        lhs.x / rhs.x, lhs.y / rhs.y, lhs.z / rhs.z, lhs.w / rhs.w);            \
}                                                                               \
inline Vec4<T> operator+(const Vec4<T> &lhs, const Value<T> &rhs)                      \
{                                                                               \
    return make_vec4<T>(                                                        \
        lhs.x + rhs, lhs.y + rhs, lhs.z + rhs, lhs.w + rhs);                    \
}                                                                               \
inline Vec4<T> operator-(const Vec4<T> &lhs, const Value<T> &rhs)                      \
{                                                                               \
    return make_vec4<T>(                                                        \
        lhs.x - rhs, lhs.y - rhs, lhs.z - rhs, lhs.w - rhs);                    \
}                                                                               \
inline Vec4<T> operator*(const Vec4<T> &lhs, const Value<T> &rhs)                      \
{                                                                               \
    return make_vec4<T>(                                                        \
        lhs.x * rhs, lhs.y * rhs, lhs.z * rhs, lhs.w * rhs);                    \
}                                                                               \
inline Vec4<T> operator/(const Vec4<T> &lhs, const Value<T> &rhs)                      \
{                                                                               \
    return make_vec4<T>(                                                        \
        lhs.x / rhs, lhs.y / rhs, lhs.z / rhs, lhs.w / rhs);                    \
}                                                                               \
inline Vec4<T> operator+(const Value<T> &lhs, const Vec4<T> &rhs)                      \
{                                                                               \
    return make_vec4<T>(                                                        \
        lhs + rhs.x, lhs + rhs.y, lhs + rhs.z, lhs + rhs.w);                    \
}                                                                               \
inline Vec4<T> operator-(const Value<T> &lhs, const Vec4<T> &rhs)                      \
{                                                                               \
    return make_vec4<T>(                                                        \
        lhs - rhs.x, lhs - rhs.y, lhs - rhs.z, lhs - rhs.w);                    \
}                                                                               \
inline Vec4<T> operator*(const Value<T> &lhs, const Vec4<T> &rhs)                      \
{                                                                               \
    return make_vec4<T>(                                                        \
        lhs * rhs.x, lhs * rhs.y, lhs * rhs.z, lhs * rhs.w);                    \
}                                                                               \
inline Vec4<T> operator/(const Value<T> &lhs, const Vec4<T> &rhs)                      \
{                                                                               \
    return make_vec4<T>(                                                        \
        lhs / rhs.x, lhs / rhs.y, lhs / rhs.z, lhs / rhs.w);                    \
}                                                                               \
inline Value<T> dot(const Vec4<T> &a, const Vec4<T> &b)                                \
{                                                                               \
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;                       \
}                                                                               \
inline Value<T> cos(const Vec4<T> &a, const Vec4<T> &b)                                \
{                                                                               \
    return dot(a, b) / (a.length() * b.length());                               \
}

CUJ_DEFINE_VEC4_OPERATORS(int)
CUJ_DEFINE_VEC4_OPERATORS(float)
CUJ_DEFINE_VEC4_OPERATORS(double)

#undef CUJ_DEFINE_VEC4_OPERATORS

CUJ_NAMESPACE_END(cuj::builtin::math)
