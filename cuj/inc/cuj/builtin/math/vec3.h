#pragma once

#include <cuj/ast/ast.h>

CUJ_NAMESPACE_BEGIN(cuj::builtin::math)

template<typename T>
struct Vec3Host
{
    T x, y, z;
};

#define CUJ_DEFINE_VEC3(NAME, BASE, COMP)                                       \
CUJ_MAKE_PROXY_CLASS(NAME, BASE, x, y, z)                                       \
{                                                                               \
    using CUJBase::CUJBase;                                                     \
    NAME() { x = 0; y = 0; z = 0; }                                             \
    NAME(COMP _x, COMP _y, COMP _z) { x = _x; y = _y; z = _z; }                 \
    auto operator[](const Value<size_t> &idx) const                             \
    { return Value<COMP>((x.address() + idx).deref().get_impl()); }             \
    auto length_square() const { return x * x + y * y + z * z; }                \
    auto length() const { return sqrt(length_square()); }                       \
    auto min_elem() const { return min(x, min(y, z)); }                         \
    auto max_elem() const { return max(x, max(y, z)); }                         \
    auto normalize() const                                                      \
    {                                                                           \
        COMP f = 1 / length();                                                  \
        return NAME(f * x, f * y, f * z);                                       \
    }                                                                           \
};

CUJ_DEFINE_VEC3(Vec3i, Vec3Host<int>,    i32)
CUJ_DEFINE_VEC3(Vec3f, Vec3Host<float>,  f32)
CUJ_DEFINE_VEC3(Vec3d, Vec3Host<double>, f64)

#undef CUJ_DEFINE_VEC3

template<typename T> struct Vec3Aux { };
template<> struct Vec3Aux<int>    { using Type = Vec3i; };
template<> struct Vec3Aux<float>  { using Type = Vec3f; };
template<> struct Vec3Aux<double> { using Type = Vec3d; };

template<typename T>
using Vec3 = typename Vec3Aux<T>::Type;

template<typename T>
Vec3<T> make_vec3(const Value<T> &x, const Value<T> &y, const Value<T> &z)
{
    return Vec3<T>(x, y, z);
}

template<typename T>
Vec3<T> make_vec3()
{
    return make_vec3<T>(0, 0, 0);
}

template<typename T>
Vec3<T> make_vec3(const Value<T> &v)
{
    return make_vec3<T>(v, v, v);
}

inline Vec3f make_vec3f()
{
    return make_vec3<float>();
}

inline Vec3f make_vec3f(const f32 &v)
{
    return make_vec3<float>(v, v, v);
}

inline Vec3f make_vec3f(const f32 &x, const f32 &y, const f32 &z)
{
    return make_vec3<float>(x, y, z);
}

inline Vec3d make_vec3d()
{
    return make_vec3<double>();
}

inline Vec3d make_vec3d(const f64 &v)
{
    return make_vec3<double>(v);
}

inline Vec3d make_vec3d(const f64 &x, const f64 &y, const f64 &z)
{
    return make_vec3<double>(x, y, z);
}

inline Vec3i make_vec3i()
{
    return make_vec3<int>();
}

inline Vec3i make_vec3i(const i32 &v)
{
    return make_vec3<int>(v);
}

inline Vec3i make_vec3i(const i32 &x, const i32 &y, const i32 &z)
{
    return make_vec3<int>(x, y, z);
}

#define CUJ_DEFINE_VEC3_OPERATORS(T)                                            \
inline Vec3<T> operator-(const Vec3<T> &v)                                             \
{                                                                               \
    return make_vec3<T>(-v.x, -v.y, -v.z);                                      \
}                                                                               \
inline Vec3<T> operator+(const Vec3<T> &lhs, const Vec3<T> &rhs)                       \
{                                                                               \
    return make_vec3<T>(lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z);           \
}                                                                               \
inline Vec3<T> operator-(const Vec3<T> &lhs, const Vec3<T> &rhs)                       \
{                                                                               \
    return make_vec3<T>(lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z);           \
}                                                                               \
inline Vec3<T> operator*(const Vec3<T> &lhs, const Vec3<T> &rhs)                       \
{                                                                               \
    return make_vec3<T>(lhs.x * rhs.x, lhs.y * rhs.y, lhs.z * rhs.z);           \
}                                                                               \
inline Vec3<T> operator/(const Vec3<T> &lhs, const Vec3<T> &rhs)                       \
{                                                                               \
    return make_vec3<T>(lhs.x / rhs.x, lhs.y / rhs.y, lhs.z / rhs.z);           \
}                                                                               \
inline Vec3<T> operator+(const Vec3<T> &lhs, const Value<T> &rhs)                      \
{                                                                               \
    return make_vec3<T>(lhs.x + rhs, lhs.y + rhs, lhs.z + rhs);                 \
}                                                                               \
inline Vec3<T> operator-(const Vec3<T> &lhs, const Value<T> &rhs)                      \
{                                                                               \
    return make_vec3<T>(lhs.x - rhs, lhs.y - rhs, lhs.z - rhs);                 \
}                                                                               \
inline Vec3<T> operator*(const Vec3<T> &lhs, const Value<T> &rhs)                      \
{                                                                               \
    return make_vec3<T>(lhs.x * rhs, lhs.y * rhs, lhs.z * rhs);                 \
}                                                                               \
inline Vec3<T> operator/(const Vec3<T> &lhs, const Value<T> &rhs)                      \
{                                                                               \
    return make_vec3<T>(lhs.x / rhs, lhs.y / rhs, lhs.z / rhs);                 \
}                                                                               \
inline Vec3<T> operator+(const Value<T> &lhs, const Vec3<T> &rhs)                      \
{                                                                               \
    return make_vec3<T>(lhs + rhs.x, lhs + rhs.y, lhs + rhs.z);                 \
}                                                                               \
inline Vec3<T> operator-(const Value<T> &lhs, const Vec3<T> &rhs)                      \
{                                                                               \
    return make_vec3<T>(lhs - rhs.x, lhs - rhs.y, lhs - rhs.z);                 \
}                                                                               \
inline Vec3<T> operator*(const Value<T> &lhs, const Vec3<T> &rhs)                      \
{                                                                               \
    return make_vec3<T>(lhs * rhs.x, lhs * rhs.y, lhs * rhs.z);                 \
}                                                                               \
inline Vec3<T> operator/(const Value<T> &lhs, const Vec3<T> &rhs)                      \
{                                                                               \
    return make_vec3<T>(lhs / rhs.x, lhs / rhs.y, lhs / rhs.z);                 \
}                                                                               \
inline Value<T> dot(const Vec3<T> &a, const Vec3<T> &b)                                \
{                                                                               \
    return a.x * b.x + a.y * b.y + a.z * b.z;                                   \
}                                                                               \
inline Value<T> cos(const Vec3<T> &a, const Vec3<T> &b)                                \
{                                                                               \
    return dot(a, b) / (a.length() * b.length());                               \
}                                                                               \
inline Vec3<T> cross(const Vec3<T> &a, const Vec3<T> &b)                               \
{                                                                               \
    return make_vec3<T>(                                                        \
        a.y * b.z - a.z * b.y,                                                  \
        a.z * b.x - a.x * b.z,                                                  \
        a.x * b.y - a.y * b.x);                                                 \
}

CUJ_DEFINE_VEC3_OPERATORS(int)
CUJ_DEFINE_VEC3_OPERATORS(float)
CUJ_DEFINE_VEC3_OPERATORS(double)

#undef CUJ_DEFINE_VEC3_OPERATORS

CUJ_NAMESPACE_END(cuj::builtin::math)
