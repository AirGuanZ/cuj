#pragma once

#include <cuj.h>

using namespace cuj;

CUJ_CLASS_BEGIN(Vec2f)
    CUJ_MEMBER_VARIABLE(f32, x)
    CUJ_MEMBER_VARIABLE(f32, y)
CUJ_CLASS_END

CUJ_CLASS_BEGIN(Vec3f)
    CUJ_MEMBER_VARIABLE(f32, x)
    CUJ_MEMBER_VARIABLE(f32, y)
    CUJ_MEMBER_VARIABLE(f32, z)
    ref<f32> operator[](i32 i)
    {
        return *(x.address() + i);
    }
CUJ_CLASS_END

inline Vec2f make_vec2f(f32 x, f32 y)
{
    Vec2f ret;
    ret.x = x;
    ret.y = y;
    return ret;
}

inline Vec3f make_vec3f(f32 x, f32 y, f32 z)
{
    Vec3f ret;
    ret.x = x;
    ret.y = y;
    ret.z = z;
    return ret;
}

inline f32 dot(const Vec3f &a, const Vec3f &b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline f32 dot(const Vec2f &a, const Vec2f &b)
{
    return a.x * b.x + a.y * b.y;
}

inline Vec3f operator+(const Vec3f &a, const Vec3f &b)
{
    return make_vec3f(a.x + b.x, a.y + b.y, a.z + b.z);
}

inline Vec3f operator-(const Vec3f &a, const Vec3f &b)
{
    return make_vec3f(a.x - b.x, a.y - b.y, a.z - b.z);
}

inline Vec3f operator*(f32 a, const Vec3f &b)
{
    return make_vec3f(a * b.x, a * b.y, a * b.z);
}

inline Vec3f operator*(const Vec3f &a, const Vec3f &b)
{
    return make_vec3f(a.x * b.x, a.y * b.y, a.z * b.z);
}

inline f32 length_square(const Vec3f &v)
{
    return dot(v, v);
}

inline f32 length_square(const Vec2f &v)
{
    return dot(v, v);
}

inline f32 length(const Vec3f &v)
{
    return cstd::sqrt(length_square(v));
}

inline f32 length(const Vec2f &v)
{
    return cstd::sqrt(length_square(v));
}

inline Vec3f abs(const Vec3f &v)
{
    return make_vec3f(cstd::abs(v.x), cstd::abs(v.y), cstd::abs(v.z));
}

inline Vec3f cross(const Vec3f &a, const Vec3f &b)
{
    return make_vec3f(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x);
}

inline Vec3f normalize(const Vec3f &v)
{
    var inv_len = cstd::rsqrt(length_square(v));
    return make_vec3f(inv_len * v.x, inv_len * v.y, inv_len * v.z);
}
