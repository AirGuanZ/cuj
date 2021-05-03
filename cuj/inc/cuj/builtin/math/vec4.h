#pragma once

#include <cuj/ast/ast.h>

CUJ_NAMESPACE_BEGIN(cuj::builtin::math)

template<typename T>
class Vec4Impl : public ClassBase<Vec4Impl<T>>
{
public:

    CUJ_DEFINE_CLASS(Vec4Impl)

    $mem(T, x);
    $mem(T, y);
    $mem(T, z);
    $mem(T, w);

    using ClassBase<Vec4Impl>::ClassBase;

    explicit Vec4Impl(ClassAddress addr);

    Vec4Impl(ClassAddress addr, const Value<T> &v);

    Vec4Impl(
        ClassAddress    addr,
        const Value<T> &_x,
        const Value<T> &_y,
        const Value<T> &_z,
        const Value<T> &_w);

    Vec4Impl(ClassAddress addr, const ClassValue<Vec4Impl<T>> &other);

    Value<T> length_square() const;

    Value<T> length() const;

    Value<T> min_elem() const;

    Value<T> max_elem() const;

    ClassValue<Vec4Impl<T>> normalize() const;
};

template<typename T>
using Vec4 = ClassValue<Vec4Impl<T>>;

using Vec4f = Vec4<float>;
using Vec4d = Vec4<double>;
using Vec4i = Vec4<int>;

template<typename T>
Vec4<T> make_vec4();
template<typename T>
Vec4<T> make_vec4(const Value<T> &v);
template<typename T>
Vec4<T> make_vec4(
    const Value<T> &x, const Value<T> &y, const Value<T> &z, const Value<T> &w);

inline Vec4f make_vec4f();
inline Vec4f make_vec4f(const f32 &v);
inline Vec4f make_vec4f(
    const f32 &x, const f32 &y, const f32 &z, const f32 &w);

inline Vec4d make_vec4d();
inline Vec4d make_vec4d(const f64 &v);
inline Vec4d make_vec4d(
    const f64 &x, const f64 &y, const f64 &z, const f64 &w);

inline Vec4i make_vec4i();
inline Vec4i make_vec4i(const i32 &v);
inline Vec4i make_vec4i(
    const i32 &x, const i32 &y, const i32 &z, const i32 &w);

template<typename T>
Vec4<T> operator+(const Vec4<T> &lhs, const Vec4<T> &rhs);
template<typename T>
Vec4<T> operator-(const Vec4<T> &lhs, const Vec4<T> &rhs);
template<typename T>
Vec4<T> operator*(const Vec4<T> &lhs, const Vec4<T> &rhs);
template<typename T>
Vec4<T> operator/(const Vec4<T> &lhs, const Vec4<T> &rhs);

template<typename T>
Vec4<T> operator+(const Vec4<T> &lhs, const Value<T> &rhs);
template<typename T>
Vec4<T> operator-(const Vec4<T> &lhs, const Value<T> &rhs);
template<typename T>
Vec4<T> operator*(const Vec4<T> &lhs, const Value<T> &rhs);
template<typename T>
Vec4<T> operator/(const Vec4<T> &lhs, const Value<T> &rhs);

template<typename T>
Vec4<T> operator+(const Value<T> &lhs, const Vec4<T> &rhs);
template<typename T>
Vec4<T> operator-(const Value<T> &lhs, const Vec4<T> &rhs);
template<typename T>
Vec4<T> operator*(const Value<T> &lhs, const Vec4<T> &rhs);
template<typename T>
Vec4<T> operator/(const Value<T> &lhs, const Vec4<T> &rhs);

template<typename T>
Value<T> dot(const Vec4<T> &a, const Vec4<T> &b);
template<typename T>
Value<T> cos(const Vec4<T> &a, const Vec4<T> &b);

CUJ_NAMESPACE_END(cuj::builtin::math)
