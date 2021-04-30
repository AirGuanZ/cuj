#pragma once

#include <cuj/ast/ast.h>

CUJ_NAMESPACE_BEGIN(cuj::builtin::math)

class Float4Impl : public ast::ClassBase<Float4Impl>
{
public:

    $mem(float, x);
    $mem(float, y);
    $mem(float, z);
    $mem(float, w);

    using ClassBase::ClassBase;

    explicit Float4Impl(ClassAddress addr);

    Float4Impl(ClassAddress addr, const $float &v);

    Float4Impl(
        ClassAddress addr,
        const $float &_x,
        const $float &_y,
        const $float &_z,
        const $float &_w);

    Float4Impl(ClassAddress addr, const Value<Float4Impl> &other);

    $float length_square() const;

    $float length() const;

    $float min_elem() const;

    $float max_elem() const;

    Value<Float4Impl> normalize() const;
};

using Float4 = Value<Float4Impl>;

Float4 make_float4();
Float4 make_float4(const $float &v);
Float4 make_float4(
    const $float &x, const $float &y, const $float &z, const $float &w);

Float4 operator+(const Float4 &lhs, const Float4 &rhs);
Float4 operator-(const Float4 &lhs, const Float4 &rhs);
Float4 operator*(const Float4 &lhs, const Float4 &rhs);
Float4 operator/(const Float4 &lhs, const Float4 &rhs);

Float4 operator+(const Float4 &lhs, const $float &rhs);
Float4 operator-(const Float4 &lhs, const $float &rhs);
Float4 operator*(const Float4 &lhs, const $float &rhs);
Float4 operator/(const Float4 &lhs, const $float &rhs);

Float4 operator+(const $float &lhs, const Float4 &rhs);
Float4 operator-(const $float &lhs, const Float4 &rhs);
Float4 operator*(const $float &lhs, const Float4 &rhs);
Float4 operator/(const $float &lhs, const Float4 &rhs);

$float dot(const Float4 &a, const Float4 &b);
$float cos(const Float4 &a, const Float4 &b);

CUJ_NAMESPACE_END(cuj::builtin::math)
