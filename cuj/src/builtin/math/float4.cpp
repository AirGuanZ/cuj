#include <cuj/builtin/math/basic.h>
#include <cuj/builtin/math/float4.h>

CUJ_NAMESPACE_BEGIN(cuj::builtin::math)

Float4Impl::Float4Impl(ClassAddress addr)
    : Float4Impl(std::move(addr), 0, 0, 0, 0)
{
    
}

Float4Impl::Float4Impl(ClassAddress addr, $float v)
    : Float4Impl(std::move(addr), v, v, v, v)
{
    
}

Float4Impl::Float4Impl(
    ClassAddress addr, $float _x, $float _y, $float _z, $float _w)
    : ClassBase(std::move(addr))
{
    x = _x;
    y = _y;
    z = _z;
    w = _w;
}

Float4Impl::Float4Impl(ClassAddress addr, const ast::Value<Float4Impl> &other)
    : Float4Impl(std::move(addr), other->x, other->y, other->z, other->w)
{
    
}

$float Float4Impl::length_square() const
{
    return x * x + y * y + z * z + w * w;
}

$float Float4Impl::length() const
{
    return sqrt(length_square());
}

$float Float4Impl::min_elem() const
{
    return min(min(x, y), min(z, w));
}

$float Float4Impl::max_elem() const
{
    return max(max(x, y), max(z, w));
}

Float4 Float4Impl::normalize() const
{
    $float inv_len = 1.0f / length();
    return make_float4(x * inv_len, y * inv_len, z * inv_len, w * inv_len);
}

Float4 make_float4()
{
    return make_float4(0, 0, 0, 0);
}

Float4 make_float4($float v)
{
    return make_float4(v, v, v, v);
}

Float4 make_float4($float x, $float y, $float z, $float w)
{
    $var(Float4, ret, x, y, z, w);
    return ret;
}

Float4 operator+(const Float4 &lhs, const Float4 &rhs)
{
    return make_float4(
        lhs->x + rhs->x, lhs->y + rhs->y, lhs->z + rhs->z, lhs->w + rhs->w);
}

Float4 operator-(const Float4 &lhs, const Float4 &rhs)
{
    return make_float4(
        lhs->x - rhs->x, lhs->y - rhs->y, lhs->z - rhs->z, lhs->w - rhs->w);
}

Float4 operator*(const Float4 &lhs, const Float4 &rhs)
{
    return make_float4(
        lhs->x * rhs->x, lhs->y * rhs->y, lhs->z * rhs->z, lhs->w * rhs->w);
}

Float4 operator/(const Float4 &lhs, const Float4 &rhs)
{
    return make_float4(
        lhs->x / rhs->x, lhs->y / rhs->y, lhs->z / rhs->z, lhs->w / rhs->w);
}

Float4 operator+(const Float4 &lhs, $float rhs)
{
    return make_float4(lhs->x + rhs, lhs->y + rhs, lhs->z + rhs, lhs->w + rhs);
}

Float4 operator-(const Float4 &lhs, $float rhs)
{
    return make_float4(lhs->x - rhs, lhs->y - rhs, lhs->z - rhs, lhs->w - rhs);
}

Float4 operator*(const Float4 &lhs, $float rhs)
{
    return make_float4(lhs->x * rhs, lhs->y * rhs, lhs->z * rhs, lhs->w * rhs);
}

Float4 operator/(const Float4 &lhs, $float rhs)
{
    return make_float4(lhs->x / rhs, lhs->y / rhs, lhs->z / rhs, lhs->w / rhs);
}

Float4 operator+($float lhs, const Float4 &rhs)
{
    return make_float4(lhs + rhs->x, lhs + rhs->y, lhs + rhs->z, lhs + rhs->w);
}

Float4 operator-($float lhs, const Float4 &rhs)
{
    return make_float4(lhs - rhs->x, lhs - rhs->y, lhs - rhs->z, lhs - rhs->w);
}

Float4 operator*($float lhs, const Float4 &rhs)
{
    return make_float4(lhs * rhs->x, lhs * rhs->y, lhs * rhs->z, lhs * rhs->w);
}

Float4 operator/($float lhs, const Float4 &rhs)
{
    return make_float4(lhs / rhs->x, lhs / rhs->y, lhs / rhs->z, lhs / rhs->w);
}

$float dot(const Float4 &a, const Float4 &b)
{
    return a->x * b->x + a->y * b->y + a->z * b->z + a->w * b->w;
}

$float cos(const Float4 &a, const Float4 &b)
{
    return dot(a, b) / (a->length() * b->length());
}

CUJ_NAMESPACE_END(cuj::builtin::math)
