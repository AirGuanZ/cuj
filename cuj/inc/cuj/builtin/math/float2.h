#pragma once

#include <cuj/ast/ast.h>

CUJ_NAMESPACE_BEGIN(cuj::builtin::math)

class Float2Impl : public ast::ClassBase<Float2Impl>
{
public:

    $mem(float, x);
    $mem(float, y);

    using ClassBase::ClassBase;

    explicit Float2Impl(ClassAddress addr);

    Float2Impl(ClassAddress addr, $float v);

    Float2Impl(ClassAddress addr, $float _x, $float _y);

    Float2Impl(ClassAddress addr, const ast::Value<Float2Impl> &other);

    $float length_square() const;

    $float length() const;

    $float min_elem() const;

    $float max_elem() const;

    ast::Value<Float2Impl> normalize() const;
};

using Float2 = ast::Value<Float2Impl>;

Float2 make_float2();
Float2 make_float2($float v);
Float2 make_float2($float x, $float y);

Float2 operator+(const Float2 &lhs, const Float2 &rhs);
Float2 operator-(const Float2 &lhs, const Float2 &rhs);
Float2 operator*(const Float2 &lhs, const Float2 &rhs);
Float2 operator/(const Float2 &lhs, const Float2 &rhs);

Float2 operator+(const Float2 &lhs, $float rhs);
Float2 operator-(const Float2 &lhs, $float rhs);
Float2 operator*(const Float2 &lhs, $float rhs);
Float2 operator/(const Float2 &lhs, $float rhs);

Float2 operator+($float lhs, const Float2 &rhs);
Float2 operator-($float lhs, const Float2 &rhs);
Float2 operator*($float lhs, const Float2 &rhs);
Float2 operator/($float lhs, const Float2 &rhs);

$float dot(const Float2 &a, const Float2 &b);
$float cos(const Float2 &a, const Float2 &b);

CUJ_NAMESPACE_END(cuj::builtin::math)

CUJ_NAMESPACE_BEGIN(cuj::ast)

ClassValue(builtin::math::Float2) ->
ClassValue<builtin::math::Float2Impl>;

CUJ_NAMESPACE_END(cuj::ast)
