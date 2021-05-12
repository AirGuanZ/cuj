#pragma once

#include <cuj/ast/ast.h>

CUJ_NAMESPACE_BEGIN(cuj::builtin::math)

template<typename T>
class Vec1Impl : public ClassBase<Vec1Impl<T>>
{
public:

    CUJ_DEFINE_CLASS(Vec1Impl)

    $mem(T, x);

    using ClassBase<Vec1Impl>::ClassBase;

    explicit Vec1Impl(ClassAddress addr);

    Vec1Impl(ClassAddress addr, const Value<T> &v);
    
    Vec1Impl(ClassAddress addr, const ClassValue<Vec1Impl<T>> &other);

    Value<T> length_square() const;

    Value<T> length() const;

    Value<T> min_elem() const;

    Value<T> max_elem() const;

    ClassValue<Vec1Impl<T>> normalize() const;
};

template<typename T>
using Vec1 = ClassValue<Vec1Impl<T>>;

using Vec1f = Vec1<float>;
using Vec1d = Vec1<double>;
using Vec1i = Vec1<int>;

template<typename T>
Vec1<T> make_vec1();
template<typename T>
Vec1<T> make_vec1(const Value<T> &v);

inline Vec1f make_vec1f();
inline Vec1f make_vec1f(const f32 &v);

inline Vec1d make_vec1d();
inline Vec1d make_vec1d(const f64 &v);

inline Vec1i make_vec1i();
inline Vec1i make_vec1i(const i32 &v);

template<typename T>
Vec1<T> operator-(const Vec1<T> &v);

template<typename T>
Vec1<T> operator+(const Vec1<T> &lhs, const Vec1<T> &rhs);
template<typename T>
Vec1<T> operator-(const Vec1<T> &lhs, const Vec1<T> &rhs);
template<typename T>
Vec1<T> operator*(const Vec1<T> &lhs, const Vec1<T> &rhs);
template<typename T>
Vec1<T> operator/(const Vec1<T> &lhs, const Vec1<T> &rhs);

template<typename T>
Vec1<T> operator+(const Vec1<T> &lhs, const Value<T> &rhs);
template<typename T>
Vec1<T> operator-(const Vec1<T> &lhs, const Value<T> &rhs);
template<typename T>
Vec1<T> operator*(const Vec1<T> &lhs, const Value<T> &rhs);
template<typename T>
Vec1<T> operator/(const Vec1<T> &lhs, const Value<T> &rhs);

template<typename T>
Vec1<T> operator+(const Value<T> &lhs, const Vec1<T> &rhs);
template<typename T>
Vec1<T> operator-(const Value<T> &lhs, const Vec1<T> &rhs);
template<typename T>
Vec1<T> operator*(const Value<T> &lhs, const Vec1<T> &rhs);
template<typename T>
Vec1<T> operator/(const Value<T> &lhs, const Vec1<T> &rhs);

template<typename T>
Value<T> dot(const Vec1<T> &a, const Vec1<T> &b);
template<typename T>
Value<T> cos(const Vec1<T> &a, const Vec1<T> &b);

CUJ_NAMESPACE_END(cuj::builtin::math)
