#pragma once

#include <cuj/ast/ast.h>

CUJ_NAMESPACE_BEGIN(cuj::builtin::math)

template<typename T>
struct Vec2Host
{
    T x, y;
};

#define CUJ_DEFINE_VEC2(NAME, BASE, COMP)                                       \
CUJ_MAKE_PROXY_CLASS(NAME, BASE, x, y)                                          \
{                                                                               \
    using CUJBase::CUJBase;                                                     \
    NAME() { x = 0; y = 0; }                                                    \
    NAME(COMP _x, COMP _y) { x = _x; y = _y; }                                  \
    auto operator[](const Value<size_t> &idx) const                             \
    { return Value<COMP>((x.address() + idx).deref().get_impl()); }             \
    auto length_square() const { return x * x + y * y; }                        \
    auto length() const { return sqrt(length_square()); }                       \
    auto min_elem() const { return min(x, y); }                                 \
    auto max_elem() const { return max(x, y); }                                 \
    auto normalize() const                                                      \
    {                                                                           \
        COMP f = 1 / length();                                                  \
        return NAME(f * x, f * y);                                              \
    }                                                                           \
};

CUJ_DEFINE_VEC2(Vec2i, Vec2Host<int>,    i32)
CUJ_DEFINE_VEC2(Vec2f, Vec2Host<float>,  f32)
CUJ_DEFINE_VEC2(Vec2d, Vec2Host<double>, f64)

#undef CUJ_DEFINE_VEC2

template<typename T> struct Vec2Aux { };
template<> struct Vec2Aux<int>    { using Type = Vec2i; };
template<> struct Vec2Aux<float>  { using Type = Vec2f; };
template<> struct Vec2Aux<double> { using Type = Vec2d; };

template<typename T>
using Vec2 = typename Vec2Aux<T>::Type;

template<typename T>
Vec2<T> make_vec2(const Value<T> &x, const Value<T> &y)
{
    return Vec2<T>(x, y);
}

template<typename T>
Vec2<T> make_vec2()
{
    return make_vec2<T>(0, 0);
}

template<typename T>
Vec2<T> make_vec2(const Value<T> &v)
{
    return make_vec2<T>(v, v);
}

inline Vec2f make_vec2f()
{
    return make_vec2<float>();
}

inline Vec2f make_vec2f(const f32 &v)
{
    return make_vec2<float>(v, v);
}

inline Vec2f make_vec2f(const f32 &x, const f32 &y)
{
    return make_vec2<float>(x, y);
}

inline Vec2d make_vec2d()
{
    return make_vec2<double>();
}

inline Vec2d make_vec2d(const f64 &v)
{
    return make_vec2<double>(v);
}

inline Vec2d make_vec2d(const f64 &x, const f64 &y)
{
    return make_vec2<double>(x, y);
}

inline Vec2i make_vec2i()
{
    return make_vec2<int>();
}

inline Vec2i make_vec2i(const i32 &v)
{
    return make_vec2<int>(v);
}

inline Vec2i make_vec2i(const i32 &x, const i32 &y)
{
    return make_vec2<int>(x, y);
}

#define CUJ_DEFINE_VEC2_OPERATORS(T)                                                \
inline Vec2<T> operator-(const Vec2<T> &v)                                             \
{                                                                               \
    return make_vec2<T>(-v.x, -v.y);                                            \
}                                                                               \
inline Vec2<T> operator+(const Vec2<T> &lhs, const Vec2<T> &rhs)                       \
{                                                                               \
    return make_vec2<T>(lhs.x + rhs.x, lhs.y + rhs.y);                          \
}                                                                               \
inline Vec2<T> operator-(const Vec2<T> &lhs, const Vec2<T> &rhs)                       \
{                                                                               \
    return make_vec2<T>(lhs.x - rhs.x, lhs.y - rhs.y);                          \
}                                                                               \
inline Vec2<T> operator*(const Vec2<T> &lhs, const Vec2<T> &rhs)                       \
{                                                                               \
    return make_vec2<T>(lhs.x * rhs.x, lhs.y * rhs.y);                          \
}                                                                               \
inline Vec2<T> operator/(const Vec2<T> &lhs, const Vec2<T> &rhs)                       \
{                                                                               \
    return make_vec2<T>(lhs.x / rhs.x, lhs.y / rhs.y);                          \
}                                                                               \
inline Vec2<T> operator+(const Vec2<T> &lhs, const Value<T> &rhs)                      \
{                                                                               \
    return make_vec2<T>(lhs.x + rhs, lhs.y + rhs);                              \
}                                                                               \
inline Vec2<T> operator-(const Vec2<T> &lhs, const Value<T> &rhs)                      \
{                                                                               \
    return make_vec2<T>(lhs.x - rhs, lhs.y - rhs);                              \
}                                                                               \
inline Vec2<T> operator*(const Vec2<T> &lhs, const Value<T> &rhs)                      \
{                                                                               \
    return make_vec2<T>(lhs.x * rhs, lhs.y * rhs);                              \
}                                                                               \
inline Vec2<T> operator/(const Vec2<T> &lhs, const Value<T> &rhs)                      \
{                                                                               \
    return make_vec2<T>(lhs.x / rhs, lhs.y / rhs);                              \
}                                                                               \
inline Vec2<T> operator+(const Value<T> &lhs, const Vec2<T> &rhs)                      \
{                                                                               \
    return make_vec2<T>(lhs + rhs.x, lhs + rhs.y);                              \
}                                                                               \
inline Vec2<T> operator-(const Value<T> &lhs, const Vec2<T> &rhs)                      \
{                                                                               \
    return make_vec2<T>(lhs - rhs.x, lhs - rhs.y);                              \
}                                                                               \
inline Vec2<T> operator*(const Value<T> &lhs, const Vec2<T> &rhs)                      \
{                                                                               \
    return make_vec2<T>(lhs * rhs.x, lhs * rhs.y);                              \
}                                                                               \
inline Vec2<T> operator/(const Value<T> &lhs, const Vec2<T> &rhs)                      \
{                                                                               \
    return make_vec2<T>(lhs / rhs.x, lhs / rhs.y);                              \
}                                                                               \
inline Value<T> dot(const Vec2<T> &a, const Vec2<T> &b)                                \
{                                                                               \
    return a.x * b.x + a.y * b.y;                                               \
}                                                                               \
inline Value<T> cos(const Vec2<T> &a, const Vec2<T> &b)                                \
{                                                                               \
    return dot(a, b) / (a.length() * b.length());                               \
}

CUJ_DEFINE_VEC2_OPERATORS(int)
CUJ_DEFINE_VEC2_OPERATORS(float)
CUJ_DEFINE_VEC2_OPERATORS(double)

#undef CUJ_DEFINE_VEC2_OPERATORS

CUJ_NAMESPACE_END(cuj::builtin::math)
