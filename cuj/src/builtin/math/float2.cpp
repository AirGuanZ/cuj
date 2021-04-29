#include <cuj/builtin/math/basic.h>
#include <cuj/builtin/math/float2.h>

CUJ_NAMESPACE_BEGIN(cuj::builtin::math)

Float2Impl::Float2Impl(ClassAddress addr)
    : Float2Impl(std::move(addr), 0, 0)
{

}

Float2Impl::Float2Impl(ClassAddress addr, $float v)
    : Float2Impl(std::move(addr), v, v)
{

}

Float2Impl::Float2Impl(ClassAddress addr, $float _x, $float _y)
    : ClassBase<Float2Impl>(std::move(addr))
{
    x = _x;
    y = _y;
}

Float2Impl::Float2Impl(ClassAddress addr, const ast::Value<Float2Impl> &other)
    : Float2Impl(std::move(addr), other->x, other->y)
{

}

$float Float2Impl::length_square() const
{
    return x * x + y * y;
}

$float Float2Impl::length() const
{
    return sqrt(length_square());
}

$float Float2Impl::min_elem() const
{
    return min(x, y);
}

$float Float2Impl::max_elem() const
{
    return max(x, y);
}

Float2 Float2Impl::normalize() const
{
    $var(float, inv_len, 1.0f / length());
    return make_float2(x * inv_len, y * inv_len);
}

Float2 make_float2($float x, $float y)
{
    $var(Float2, ret, x, y);
    return ret;
}

Float2 make_float2($float v)
{
    return make_float2(v, v);
}

Float2 make_float2()
{
    return make_float2(0);
}

Float2 operator+(const Float2 &lhs, const Float2 &rhs)
{
    return make_float2(lhs->x + rhs->x, lhs->y + rhs->y);
}

Float2 operator-(const Float2 &lhs, const Float2 &rhs)
{
    return make_float2(lhs->x - rhs->x, lhs->y - rhs->y);
}

Float2 operator*(const Float2 &lhs, const Float2 &rhs)
{
    return make_float2(lhs->x * rhs->x, lhs->y * rhs->y);
}

Float2 operator/(const Float2 &lhs, const Float2 &rhs)
{
    return make_float2(lhs->x / rhs->x, lhs->y / rhs->y);
}

Float2 operator+(const Float2 &lhs, $float rhs)
{
    return make_float2(lhs->x + rhs, lhs->y + rhs);
}

Float2 operator-(const Float2 &lhs, $float rhs)
{
    return make_float2(lhs->x - rhs, lhs->y - rhs);
}

Float2 operator*(const Float2 &lhs, $float rhs)
{
    return make_float2(lhs->x * rhs, lhs->y * rhs);
}

Float2 operator/(const Float2 &lhs, $float rhs)
{
    return make_float2(lhs->x / rhs, lhs->y / rhs);
}

Float2 operator+($float lhs, const Float2 &rhs)
{
    return make_float2(lhs + rhs->x, lhs + rhs->y);
}

Float2 operator-($float lhs, const Float2 &rhs)
{
    return make_float2(lhs - rhs->x, lhs - rhs->y);
}

Float2 operator*($float lhs, const Float2 &rhs)
{
    return make_float2(lhs * rhs->x, lhs * rhs->y);
}

Float2 operator/($float lhs, const Float2 &rhs)
{
    return make_float2(lhs / rhs->x, lhs / rhs->y);
}

$float dot(const Float2 &a, const Float2 &b)
{
    return a->x * b->x + a->y * b->y;
}

$float cos(const Float2 &a, const Float2 &b)
{
    return dot(a, b) / (a->length() * b->length());
}

CUJ_NAMESPACE_END(cuj::builtin::math)
