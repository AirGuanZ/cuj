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

    Float4Impl(ClassAddress addr, $float v);

    Float4Impl(ClassAddress addr, $float _x, $float _y, $float _z, $float w);

    Float4Impl(ClassAddress addr, const ast::Value<Float4Impl> &other);

    $float length_square() const;

    $float length() const;

    $float min_elem() const;

    $float max_elem() const;

    ast::Value<Float4Impl> normalize() const;
};

using Float4 = ast::Value<Float4Impl>;

Float4 make_float4();
Float4 make_float4($float v);
Float4 make_float4($float x, $float y, $float z, $float w);

Float4 operator+(const Float4 &lhs, const Float4 &rhs);
Float4 operator-(const Float4 &lhs, const Float4 &rhs);
Float4 operator*(const Float4 &lhs, const Float4 &rhs);
Float4 operator/(const Float4 &lhs, const Float4 &rhs);

Float4 operator+(const Float4 &lhs, $float rhs);
Float4 operator-(const Float4 &lhs, $float rhs);
Float4 operator*(const Float4 &lhs, $float rhs);
Float4 operator/(const Float4 &lhs, $float rhs);

Float4 operator+($float lhs, const Float4 &rhs);
Float4 operator-($float lhs, const Float4 &rhs);
Float4 operator*($float lhs, const Float4 &rhs);
Float4 operator/($float lhs, const Float4 &rhs);

$float dot(const Float4 &a, const Float4 &b);
$float cos(const Float4 &a, const Float4 &b);

CUJ_NAMESPACE_END(cuj::builtin::math)

CUJ_NAMESPACE_BEGIN(cuj::ast)

ClassValue(builtin::math::Float4) ->
    ClassValue<builtin::math::Float4Impl>;

CUJ_NAMESPACE_END(cuj::ast)
