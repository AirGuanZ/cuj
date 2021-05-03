#pragma once

#include <cuj/ast/ast.h>

CUJ_NAMESPACE_BEGIN(cuj::builtin::math)

template<typename T>
class Vec3Impl : public ClassBase<Vec3Impl<T>>
{
public:

    CUJ_DEFINE_CLASS(Vec3Impl)

    $mem(T, x);
    $mem(T, y);
    $mem(T, z);

    using ClassBase<Vec3Impl>::ClassBase;

    explicit Vec3Impl(ClassAddress addr);

    Vec3Impl(ClassAddress addr, const Value<T> &v);

    Vec3Impl(
        ClassAddress    addr,
        const Value<T> &_x,
        const Value<T> &_y,
        const Value<T> &_z);

    Vec3Impl(ClassAddress addr, const ClassValue<Vec3Impl<T>> &other);

    Value<T> length_square() const;

    Value<T> length() const;

    Value<T> min_elem() const;

    Value<T> max_elem() const;

    ClassValue<Vec3Impl<T>> normalize() const;
};

template<typename T>
using Vec3 = ClassValue<Vec3Impl<T>>;

using Vec3f = Vec3<float>;
using Vec3d = Vec3<double>;
using Vec3i = Vec3<int>;

template<typename T>
Vec3<T> make_vec3();
template<typename T>
Vec3<T> make_vec3(const Value<T> &v);
template<typename T>
Vec3<T> make_vec3(const Value<T> &x, const Value<T> &y, const Value<T> &z);

inline Vec3f make_vec3f();
inline Vec3f make_vec3f(const $float &v);
inline Vec3f make_vec3f(const $float &x, const $float &y, const $float &z);

inline Vec3d make_vec3d();
inline Vec3d make_vec3d(const $double &v);
inline Vec3d make_vec3d(const $double &x, const $double &y, const $double &z);

inline Vec3i make_vec3i();
inline Vec3i make_vec3i(const $int &v);
inline Vec3i make_vec3i(const $int &x, const $int &y, const $int &z);

template<typename T>
Vec3<T> operator+(const Vec3<T> &lhs, const Vec3<T> &rhs);
template<typename T>
Vec3<T> operator-(const Vec3<T> &lhs, const Vec3<T> &rhs);
template<typename T>
Vec3<T> operator*(const Vec3<T> &lhs, const Vec3<T> &rhs);
template<typename T>
Vec3<T> operator/(const Vec3<T> &lhs, const Vec3<T> &rhs);

template<typename T>
Vec3<T> operator+(const Vec3<T> &lhs, const Value<T> &rhs);
template<typename T>
Vec3<T> operator-(const Vec3<T> &lhs, const Value<T> &rhs);
template<typename T>
Vec3<T> operator*(const Vec3<T> &lhs, const Value<T> &rhs);
template<typename T>
Vec3<T> operator/(const Vec3<T> &lhs, const Value<T> &rhs);

template<typename T>
Vec3<T> operator+(const Value<T> &lhs, const Vec3<T> &rhs);
template<typename T>
Vec3<T> operator-(const Value<T> &lhs, const Vec3<T> &rhs);
template<typename T>
Vec3<T> operator*(const Value<T> &lhs, const Vec3<T> &rhs);
template<typename T>
Vec3<T> operator/(const Value<T> &lhs, const Vec3<T> &rhs);

template<typename T>
Value<T> dot(const Vec3<T> &a, const Vec3<T> &b);
template<typename T>
Value<T> cos(const Vec3<T> &a, const Vec3<T> &b);
template<typename T>
Vec3<T> cross(const Vec3<T> &a, const Vec3<T> &b);

CUJ_NAMESPACE_END(cuj::builtin::math)
