#include <cuj/builtin/math/basic.h>
#include <cuj/builtin/math/float3.h>

CUJ_NAMESPACE_BEGIN(cuj::builtin::math)

Float3Impl::Float3Impl(ClassAddress addr)
    : Float3Impl(std::move(addr), 0, 0, 0)
{
    
}

Float3Impl::Float3Impl(ClassAddress addr, $float v)
    : Float3Impl(std::move(addr), v, v, v)
{
    
}

Float3Impl::Float3Impl(ClassAddress addr, $float _x, $float _y, $float _z)
    : ClassBase(std::move(addr))
{
    x = _x;
    y = _y;
    z = _z;
}

Float3Impl::Float3Impl(ClassAddress addr, const ast::Value<Float3Impl> &other)
    : Float3Impl(std::move(addr), other->x, other->y, other->z)
{
    
}

$float Float3Impl::length_square() const
{
    return x * x + y * y + z * z;
}

$float Float3Impl::length() const
{
    return sqrt(length_square());
}

$float Float3Impl::min_elem() const
{
    return min(x, min(y, z));
}

$float Float3Impl::max_elem() const
{
    return max(x, max(y, z));
}

Float3 Float3Impl::normalize() const
{
    $var(float, inv_len , 1.0f / length());
    return make_float3(x * inv_len, y * inv_len, z * inv_len);
}

Float3 make_float3()
{
    return make_float3(0, 0, 0);
}

Float3 make_float3($float v)
{
    return make_float3(v, v, v);
}

Float3 make_float3($float x, $float y, $float z)
{
    $var(Float3, ret, x, y, z);
    return ret;
}

Float3 operator+(const Float3 &lhs, const Float3 &rhs)
{
    return make_float3(lhs->x + rhs->x, lhs->y + rhs->y, lhs->z + rhs->z);
}

Float3 operator-(const Float3 &lhs, const Float3 &rhs)
{
    return make_float3(lhs->x - rhs->x, lhs->y - rhs->y, lhs->z - rhs->z);
}

Float3 operator*(const Float3 &lhs, const Float3 &rhs)
{
    return make_float3(lhs->x * rhs->x, lhs->y * rhs->y, lhs->z * rhs->z);
}

Float3 operator/(const Float3 &lhs, const Float3 &rhs)
{
    return make_float3(lhs->x / rhs->x, lhs->y / rhs->y, lhs->z / rhs->z);
}

Float3 operator+(const Float3 &lhs, $float rhs)
{
    return make_float3(lhs->x + rhs, lhs->y + rhs, lhs->z + rhs);
}

Float3 operator-(const Float3 &lhs, $float rhs)
{
    return make_float3(lhs->x - rhs, lhs->y - rhs, lhs->z - rhs);
}

Float3 operator*(const Float3 &lhs, $float rhs)
{
    return make_float3(lhs->x * rhs, lhs->y * rhs, lhs->z * rhs);
}

Float3 operator/(const Float3 &lhs, $float rhs)
{
    return make_float3(lhs->x / rhs, lhs->y / rhs, lhs->z / rhs);
}

Float3 operator+($float lhs, const Float3 &rhs)
{
    return make_float3(lhs + rhs->x, lhs + rhs->y, lhs + rhs->z);
}

Float3 operator-($float lhs, const Float3 &rhs)
{
    return make_float3(lhs - rhs->x, lhs - rhs->y, lhs - rhs->z);
}

Float3 operator*($float lhs, const Float3 &rhs)
{
    return make_float3(lhs * rhs->x, lhs * rhs->y, lhs * rhs->z);
}

Float3 operator/($float lhs, const Float3 &rhs)
{
    return make_float3(lhs / rhs->x, lhs / rhs->y, lhs / rhs->z);
}

$float dot(const Float3 &a, const Float3 &b)
{
    return a->x * b->x + a->y * b->y + a->z * b->z;
}

$float cos(const Float3 &a, const Float3 &b)
{
    return dot(a, b) / (a->length() * b->length());
}

Float3 cross(const Float3 &lhs, const Float3 &rhs)
{
    return make_float3(
        lhs->y * rhs->z - lhs->z * rhs->y,
        lhs->z * rhs->x - lhs->x * rhs->z,
        lhs->x * rhs->y - lhs->y * rhs->x);
}

CUJ_NAMESPACE_END(cuj::builtin::math)
