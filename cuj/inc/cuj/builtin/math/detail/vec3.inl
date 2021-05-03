#pragma once

#include <cuj/builtin/math/basic.h>
#include <cuj/builtin/math/vec3.h>

CUJ_NAMESPACE_BEGIN(cuj::builtin::math)

template<typename T>
Vec3Impl<T>::Vec3Impl(ClassAddress addr)
    : Vec3Impl(std::move(addr), 0, 0, 0)
{
    
}

template<typename T>
Vec3Impl<T>::Vec3Impl(ClassAddress addr, const Value<T> &v)
    : Vec3Impl(std::move(addr), v, v, v)
{
    
}

template<typename T>
Vec3Impl<T>::Vec3Impl(
    ClassAddress    addr,
    const Value<T> &_x,
    const Value<T> &_y,
    const Value<T> &_z)
    : ClassBase<Vec3Impl>(std::move(addr))
{
    x = _x;
    y = _y;
    z = _z;
}

template<typename T>
Vec3Impl<T>::Vec3Impl(ClassAddress addr, const ClassValue<Vec3Impl> &other)
    : Vec3Impl(std::move(addr), other->x, other->y, other->z)
{
    
}

template<typename T>
Value<T> Vec3Impl<T>::length_square() const
{
    return x * x + y * y + z * z;
}

template<typename T>
Value<T> Vec3Impl<T>::length() const
{
    static_assert(std::is_floating_point_v<T>);
    return math::sqrt(length_square());
}

template<typename T>
Value<T> Vec3Impl<T>::min_elem() const
{
    return math::min(x, math::min(y, z));
}

template<typename T>
Value<T> Vec3Impl<T>::max_elem() const
{
    return math::max(x, math::max(y, z));
}

template<typename T>
ClassValue<Vec3Impl<T>> Vec3Impl<T>::normalize() const
{
    static_assert(std::is_floating_point_v<T>);
    Value<T> inv_len = T(1) / length();
    return make_vec3<T>(x * inv_len, y * inv_len, z * inv_len);
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

template<typename T>
Vec3<T> make_vec3(const Value<T> &x, const Value<T> &y, const Value<T> &z)
{
    return Vec3<T>(x, y, z);
}

inline Vec3f make_vec3f()
{
    return make_vec3<float>();
}

inline Vec3f make_vec3f(const $float &v)
{
    return make_vec3<float>(v, v, v);
}

inline Vec3f make_vec3f(const $float &x, const $float &y, const $float &z)
{
    return make_vec3<float>(x, y, z);
}

inline Vec3d make_vec3d()
{
    return make_vec3<double>();
}

inline Vec3d make_vec3d(const $double &v)
{
    return make_vec3<double>(v);
}

inline Vec3d make_vec3d(const $double &x, const $double &y, const $double &z)
{
    return make_vec3<double>(x, y, z);
}

inline Vec3i make_vec3i()
{
    return make_vec3<int>();
}

inline Vec3i make_vec3i(const $int &v)
{
    return make_vec3<int>(v);
}

inline Vec3i make_vec3i(const $int &x, const $int &y, const $int &z)
{
    return make_vec3<int>(x, y, z);
}

template<typename T>
Vec3<T> operator+(const Vec3<T> &lhs, const Vec3<T> &rhs)
{
    return make_vec3<T>(lhs->x + rhs->x, lhs->y + rhs->y, lhs->z + rhs->z);
}

template<typename T>
Vec3<T> operator-(const Vec3<T> &lhs, const Vec3<T> &rhs)
{
    return make_vec3<T>(lhs->x - rhs->x, lhs->y - rhs->y, lhs->z - rhs->z);
}

template<typename T>
Vec3<T> operator*(const Vec3<T> &lhs, const Vec3<T> &rhs)
{
    return make_vec3<T>(lhs->x * rhs->x, lhs->y * rhs->y, lhs->z * rhs->z);
}

template<typename T>
Vec3<T> operator/(const Vec3<T> &lhs, const Vec3<T> &rhs)
{
    return make_vec3<T>(lhs->x / rhs->x, lhs->y / rhs->y, lhs->z / rhs->z);
}

template<typename T>
Vec3<T> operator+(const Vec3<T> &lhs, const Value<T> &rhs)
{
    return make_vec3<T>(lhs->x + rhs, lhs->y + rhs, lhs->z + rhs);
}

template<typename T>
Vec3<T> operator-(const Vec3<T> &lhs, const Value<T> &rhs)
{
    return make_vec3<T>(lhs->x - rhs, lhs->y - rhs, lhs->z - rhs);
}

template<typename T>
Vec3<T> operator*(const Vec3<T> &lhs, const Value<T> &rhs)
{
    return make_vec3<T>(lhs->x * rhs, lhs->y * rhs, lhs->z * rhs);
}

template<typename T>
Vec3<T> operator/(const Vec3<T> &lhs, const Value<T> &rhs)
{
    return make_vec3<T>(lhs->x / rhs, lhs->y / rhs, lhs->z / rhs);
}

template<typename T>
Vec3<T> operator+(const Value<T> &lhs, const Vec3<T> &rhs)
{
    return make_vec3<T>(lhs + rhs->x, lhs + rhs->y, lhs + rhs->z);
}

template<typename T>
Vec3<T> operator-(const Value<T> &lhs, const Vec3<T> &rhs)
{
    return make_vec3<T>(lhs - rhs->x, lhs - rhs->y, lhs - rhs->z);
}

template<typename T>
Vec3<T> operator*(const Value<T> &lhs, const Vec3<T> &rhs)
{
    return make_vec3<T>(lhs * rhs->x, lhs * rhs->y, lhs * rhs->z);
}

template<typename T>
Vec3<T> operator/(const Value<T> &lhs, const Vec3<T> &rhs)
{
    return make_vec3<T>(lhs / rhs->x, lhs / rhs->y, lhs / rhs->z);
}

template<typename T>
Value<T> dot(const Vec3<T> &a, const Vec3<T> &b)
{
    return a->x * b->x + a->y * b->y + a->z * b->z;
}

template<typename T>
Value<T> cos(const Vec3<T> &a, const Vec3<T> &b)
{
    return dot(a, b) / (a->length() * b->length());
}

template<typename T>
Vec3<T> cross(const Vec3<T> &a, const Vec3<T> &b)
{
    return make_vec3<T>(
        a->y * b->z - a->z * b->y,
        a->z * b->x - a->x * b->z,
        a->x * b->y - a->y * b->x);
}

CUJ_NAMESPACE_END(cuj::builtin::math)
