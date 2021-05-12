#pragma once

#include <cuj/builtin/math/basic.h>
#include <cuj/builtin/math/vec2.h>

CUJ_NAMESPACE_BEGIN(cuj::builtin::math)

template<typename T>
Vec2Impl<T>::Vec2Impl(ClassAddress addr)
    : Vec2Impl(std::move(addr), 0, 0)
{
    
}

template<typename T>
Vec2Impl<T>::Vec2Impl(ClassAddress addr, const Value<T> &v)
    : Vec2Impl(std::move(addr), v, v)
{
    
}

template<typename T>
Vec2Impl<T>::Vec2Impl(ClassAddress addr, const Value<T> &_x, const Value<T> &_y)
    : ClassBase<Vec2Impl>(std::move(addr))
{
    x = _x;
    y = _y;
}

template<typename T>
Vec2Impl<T>::Vec2Impl(ClassAddress addr, const ClassValue<Vec2Impl> &other)
    : Vec2Impl(std::move(addr), other->x, other->y)
{
    
}

template<typename T>
Value<T> Vec2Impl<T>::length_square() const
{
    return x * x + y * y;
}

template<typename T>
Value<T> Vec2Impl<T>::length() const
{
    static_assert(std::is_floating_point_v<T>);
    return math::sqrt(length_square());
}

template<typename T>
Value<T> Vec2Impl<T>::min_elem() const
{
    return math::min(x, y);
}

template<typename T>
Value<T> Vec2Impl<T>::max_elem() const
{
    return math::max(x, y);
}

template<typename T>
ClassValue<Vec2Impl<T>> Vec2Impl<T>::normalize() const
{
    static_assert(std::is_floating_point_v<T>);
    Value<T> inv_len = T(1) / length();
    return make_vec2<T>(x * inv_len, y * inv_len);
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

template<typename T>
Vec2<T> make_vec2(const Value<T> &x, const Value<T> &y)
{
    return Vec2<T>(x, y);
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

template<typename T>
Vec2<T> operator-(const Vec2<T> &v)
{
    return make_vec2<T>(-v->x, -v->y);
}

template<typename T>
Vec2<T> operator+(const Vec2<T> &lhs, const Vec2<T> &rhs)
{
    return make_vec2<T>(lhs->x + rhs->x, lhs->y + rhs->y);
}

template<typename T>
Vec2<T> operator-(const Vec2<T> &lhs, const Vec2<T> &rhs)
{
    return make_vec2<T>(lhs->x - rhs->x, lhs->y - rhs->y);
}

template<typename T>
Vec2<T> operator*(const Vec2<T> &lhs, const Vec2<T> &rhs)
{
    return make_vec2<T>(lhs->x * rhs->x, lhs->y * rhs->y);
}

template<typename T>
Vec2<T> operator/(const Vec2<T> &lhs, const Vec2<T> &rhs)
{
    return make_vec2<T>(lhs->x / rhs->x, lhs->y / rhs->y);
}

template<typename T>
Vec2<T> operator+(const Vec2<T> &lhs, const Value<T> &rhs)
{
    return make_vec2<T>(lhs->x + rhs, lhs->y + rhs);
}

template<typename T>
Vec2<T> operator-(const Vec2<T> &lhs, const Value<T> &rhs)
{
    return make_vec2<T>(lhs->x - rhs, lhs->y - rhs);
}

template<typename T>
Vec2<T> operator*(const Vec2<T> &lhs, const Value<T> &rhs)
{
    return make_vec2<T>(lhs->x * rhs, lhs->y * rhs);
}

template<typename T>
Vec2<T> operator/(const Vec2<T> &lhs, const Value<T> &rhs)
{
    return make_vec2<T>(lhs->x / rhs, lhs->y / rhs);
}

template<typename T>
Vec2<T> operator+(const Value<T> &lhs, const Vec2<T> &rhs)
{
    return make_vec2<T>(lhs + rhs->x, lhs + rhs->y);
}

template<typename T>
Vec2<T> operator-(const Value<T> &lhs, const Vec2<T> &rhs)
{
    return make_vec2<T>(lhs - rhs->x, lhs - rhs->y);
}

template<typename T>
Vec2<T> operator*(const Value<T> &lhs, const Vec2<T> &rhs)
{
    return make_vec2<T>(lhs * rhs->x, lhs * rhs->y);
}

template<typename T>
Vec2<T> operator/(const Value<T> &lhs, const Vec2<T> &rhs)
{
    return make_vec2<T>(lhs / rhs->x, lhs / rhs->y);
}

template<typename T>
Value<T> dot(const Vec2<T> &a, const Vec2<T> &b)
{
    return a->x * b->x + a->y * b->y;
}

template<typename T>
Value<T> cos(const Vec2<T> &a, const Vec2<T> &b)
{
    return dot(a, b) / (a->length() * b->length());
}

CUJ_NAMESPACE_END(cuj::builtin::math)
