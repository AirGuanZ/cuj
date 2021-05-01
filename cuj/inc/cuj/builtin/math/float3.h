#pragma once

#include <cuj/ast/ast.h>

CUJ_NAMESPACE_BEGIN(cuj::builtin::math)

class Float3Impl : public ast::ClassBase<Float3Impl>
{
public:

    $mem(float, x);
    $mem(float, y);
    $mem(float, z);

    using ClassBase::ClassBase;

    explicit Float3Impl(ClassAddress addr);

    Float3Impl(ClassAddress addr, const $float &v);

    Float3Impl(
        ClassAddress addr, const $float &_x, const $float &_y, const $float &_z);

    Float3Impl(ClassAddress addr, const Value<Float3Impl> &other);

    $float length_square() const;

    $float length() const;

    $float min_elem() const;

    $float max_elem() const;

    Value<Float3Impl> normalize() const;
};

using Float3 = Value<Float3Impl>;

Float3 make_float3();
Float3 make_float3(const $float &v);
Float3 make_float3(const $float &x, const $float &y, const $float &z);

Float3 operator+(const Float3 &lhs, const Float3 &rhs);
Float3 operator-(const Float3 &lhs, const Float3 &rhs);
Float3 operator*(const Float3 &lhs, const Float3 &rhs);
Float3 operator/(const Float3 &lhs, const Float3 &rhs);

Float3 operator+(const Float3 &lhs, const $float &rhs);
Float3 operator-(const Float3 &lhs, const $float &rhs);
Float3 operator*(const Float3 &lhs, const $float &rhs);
Float3 operator/(const Float3 &lhs, const $float &rhs);

Float3 operator+(const $float &lhs, const Float3 &rhs);
Float3 operator-(const $float &lhs, const Float3 &rhs);
Float3 operator*(const $float &lhs, const Float3 &rhs);
Float3 operator/(const $float &lhs, const Float3 &rhs);

$float dot(const Float3 &a, const Float3 &b);
$float cos(const Float3 &a, const Float3 &b);

Float3 cross(const Float3 &lhs, const Float3 &rhs);

CUJ_NAMESPACE_END(cuj::builtin::math)
