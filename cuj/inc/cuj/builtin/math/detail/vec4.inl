#pragma once

#include <cuj/builtin/math/basic.h>
#include <cuj/builtin/math/vec4.h>

CUJ_NAMESPACE_BEGIN(cuj::builtin::math)

template<typename T>
Vec4Impl<T>::Vec4Impl(ClassAddress addr)
    : Vec4Impl(std::move(addr), 0, 0, 0, 0)
{
    
}

template<typename T>
Vec4Impl<T>::Vec4Impl(ClassAddress addr, const Value<T> &v)
    : Vec4Impl(std::move(addr), v, v, v, v)
{
    
}

template<typename T>
Vec4Impl<T>::Vec4Impl(
    ClassAddress    addr,
    const Value<T> &_x,
    const Value<T> &_y,
    const Value<T> &_z,
    const Value<T> &_w)
    : ClassBase<Vec4Impl>(std::move(addr))
{
    x = _x;
    y = _y;
    z = _z;
    w = _w;
}

template<typename T>
Vec4Impl<T>::Vec4Impl(ClassAddress addr, const ClassValue<Vec4Impl> &other)
    : Vec4Impl(std::move(addr), other->x, other->y, other->z, other->w)
{
    
}

template<typename T>
Value<T> Vec4Impl<T>::length_square() const
{
    return x * x + y * y + z * z + w * w;
}

template<typename T>
Value<T> Vec4Impl<T>::length() const
{
    static_assert(std::is_floating_point_v<T>);
    return math::sqrt(length_square());
}

template<typename T>
Value<T> Vec4Impl<T>::min_elem() const
{
    return math::min(math::min(x, y), math::min(z, w));
}

template<typename T>
Value<T> Vec4Impl<T>::max_elem() const
{
    return math::max(math::max(x, y), math::max(z, w));
}

template<typename T>
ClassValue<Vec4Impl<T>> Vec4Impl<T>::normalize() const
{
    static_assert(std::is_floating_point_v<T>);
    Value<T> inv_len = T(1) / length();
    return make_vec4<T>(x * inv_len, y * inv_len, z * inv_len, w * inv_len);
}

template<typename T>
Value<T> Vec4Impl<T>::operator[](const ArithmeticValue<size_t> &i) const
{
    return Value<T>((x.address() + i).deref().get_impl());
}

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

template<typename T>
Vec4<T> operator-(const Vec4<T> &v)
{
    return make_vec4<T>(-v->x, -v->y, -v->z, -v->w);
}

template<typename T>
Vec4<T> operator+(const Vec4<T> &lhs, const Vec4<T> &rhs)
{
    return make_vec4<T>(
        lhs->x + rhs->x, lhs->y + rhs->y, lhs->z + rhs->z, lhs->w + rhs->w);
}

template<typename T>
Vec4<T> operator-(const Vec4<T> &lhs, const Vec4<T> &rhs)
{
    return make_vec4<T>(
        lhs->x - rhs->x, lhs->y - rhs->y, lhs->z - rhs->z, lhs->w - rhs->w);
}

template<typename T>
Vec4<T> operator*(const Vec4<T> &lhs, const Vec4<T> &rhs)
{
    return make_vec4<T>(
        lhs->x * rhs->x, lhs->y * rhs->y, lhs->z * rhs->z, lhs->w * rhs->w);
}

template<typename T>
Vec4<T> operator/(const Vec4<T> &lhs, const Vec4<T> &rhs)
{
    return make_vec4<T>(
        lhs->x / rhs->x, lhs->y / rhs->y, lhs->z / rhs->z, lhs->w / rhs->w);
}

template<typename T>
Vec4<T> operator+(const Vec4<T> &lhs, const Value<T> &rhs)
{
    return make_vec4<T>(
        lhs->x + rhs, lhs->y + rhs, lhs->z + rhs, lhs->w + rhs);
}

template<typename T>
Vec4<T> operator-(const Vec4<T> &lhs, const Value<T> &rhs)
{
    return make_vec4<T>(
        lhs->x - rhs, lhs->y - rhs, lhs->z - rhs, lhs->w - rhs);
}

template<typename T>
Vec4<T> operator*(const Vec4<T> &lhs, const Value<T> &rhs)
{
    return make_vec4<T>(
        lhs->x * rhs, lhs->y * rhs, lhs->z * rhs, lhs->w * rhs);
}

template<typename T>
Vec4<T> operator/(const Vec4<T> &lhs, const Value<T> &rhs)
{
    return make_vec4<T>(
        lhs->x / rhs, lhs->y / rhs, lhs->z / rhs, lhs->w / rhs);
}

template<typename T>
Vec4<T> operator+(const Value<T> &lhs, const Vec4<T> &rhs)
{
    return make_vec4<T>(
        lhs + rhs->x, lhs + rhs->y, lhs + rhs->z, lhs + rhs->w);
}

template<typename T>
Vec4<T> operator-(const Value<T> &lhs, const Vec4<T> &rhs)
{
    return make_vec4<T>(
        lhs - rhs->x, lhs - rhs->y, lhs - rhs->z, lhs - rhs->w);
}

template<typename T>
Vec4<T> operator*(const Value<T> &lhs, const Vec4<T> &rhs)
{
    return make_vec4<T>(
        lhs * rhs->x, lhs * rhs->y, lhs * rhs->z, lhs * rhs->w);
}

template<typename T>
Vec4<T> operator/(const Value<T> &lhs, const Vec4<T> &rhs)
{
    return make_vec4<T>(
        lhs / rhs->x, lhs / rhs->y, lhs / rhs->z, lhs / rhs->w);
}

template<typename T>
Value<T> dot(const Vec4<T> &a, const Vec4<T> &b)
{
    return a->x * b->x + a->y * b->y + a->z * b->z + a->w * b->w;
}

template<typename T>
Value<T> cos(const Vec4<T> &a, const Vec4<T> &b)
{
    return dot(a, b) / (a->length() * b->length());
}

CUJ_NAMESPACE_END(cuj::builtin::math)
