#pragma once

#include <cuj/ast/ast.h>

CUJ_NAMESPACE_BEGIN(cuj::builtin::math)

template<typename T>
class Vec2Impl : public ClassBase<Vec2Impl<T>>
{
public:

    CUJ_DEFINE_CLASS(Vec2Impl)

    $mem(T, x);
    $mem(T, y);

    using ClassBase<Vec2Impl>::ClassBase;

    explicit Vec2Impl(ClassAddress addr);

    Vec2Impl(ClassAddress addr, const Value<T> &v);

    Vec2Impl(ClassAddress addr, const Value<T> &_x, const Value<T> &_y);

    Vec2Impl(ClassAddress addr, const ClassValue<Vec2Impl<T>> &other);

    Value<T> length_square() const;

    Value<T> length() const;

    Value<T> min_elem() const;

    Value<T> max_elem() const;

    ClassValue<Vec2Impl<T>> normalize() const;

    Value<T> operator[](const ArithmeticValue<size_t> &i) const;
};

template<typename T>
using Vec2 = ClassValue<Vec2Impl<T>>;

using Vec2f = Vec2<float>;
using Vec2d = Vec2<double>;
using Vec2i = Vec2<int>;

template<typename T>
Vec2<T> make_vec2();
template<typename T>
Vec2<T> make_vec2(const Value<T> &v);
template<typename T>
Vec2<T> make_vec2(const Value<T> &x, const Value<T> &y);

inline Vec2f make_vec2f();
inline Vec2f make_vec2f(const f32 &v);
inline Vec2f make_vec2f(const f32 &x, const f32 &y);

inline Vec2d make_vec2d();
inline Vec2d make_vec2d(const f64 &v);
inline Vec2d make_vec2d(const f64 &x, const f64 &y);

inline Vec2i make_vec2i();
inline Vec2i make_vec2i(const i32 &v);
inline Vec2i make_vec2i(const i32 &x, const i32 &y);

template<typename T>
Vec2<T> operator-(const Vec2<T> &v);

template<typename T>
Vec2<T> operator+(const Vec2<T> &lhs, const Vec2<T> &rhs);
template<typename T>
Vec2<T> operator-(const Vec2<T> &lhs, const Vec2<T> &rhs);
template<typename T>
Vec2<T> operator*(const Vec2<T> &lhs, const Vec2<T> &rhs);
template<typename T>
Vec2<T> operator/(const Vec2<T> &lhs, const Vec2<T> &rhs);

template<typename T>
Vec2<T> operator+(const Vec2<T> &lhs, const Value<T> &rhs);
template<typename T>
Vec2<T> operator-(const Vec2<T> &lhs, const Value<T> &rhs);
template<typename T>
Vec2<T> operator*(const Vec2<T> &lhs, const Value<T> &rhs);
template<typename T>
Vec2<T> operator/(const Vec2<T> &lhs, const Value<T> &rhs);

template<typename T>
Vec2<T> operator+(const Value<T> &lhs, const Vec2<T> &rhs);
template<typename T>
Vec2<T> operator-(const Value<T> &lhs, const Vec2<T> &rhs);
template<typename T>
Vec2<T> operator*(const Value<T> &lhs, const Vec2<T> &rhs);
template<typename T>
Vec2<T> operator/(const Value<T> &lhs, const Vec2<T> &rhs);

template<typename T>
Value<T> dot(const Vec2<T> &a, const Vec2<T> &b);
template<typename T>
Value<T> cos(const Vec2<T> &a, const Vec2<T> &b);

CUJ_NAMESPACE_END(cuj::builtin::math)
