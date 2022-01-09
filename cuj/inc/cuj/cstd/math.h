#pragma once

#include <cuj/dsl/dsl.h>

CUJ_NAMESPACE_BEGIN(cuj::cstd)

f32 abs(f32 x);
f32 mod(f32 x, f32 y);
f32 rem(f32 x, f32 y);
f32 exp(f32 x);
f32 exp2(f32 x);
f32 exp10(f32 x);
f32 log(f32 x);
f32 log2(f32 x);
f32 log10(f32 x);
f32 pow(f32 x, f32 y);
f32 sqrt(f32 x);
f32 rsqrt(f32 x);
f32 sin(f32 x);
f32 cos(f32 x);
f32 tan(f32 x);
f32 asin(f32 x);
f32 acos(f32 x);
f32 atan(f32 x);
f32 atan2(f32 y, f32 x);
f32 ceil(f32 x);
f32 floor(f32 x);
f32 trunc(f32 x);
f32 round(f32 x);
boolean isfinite(f32 x);
boolean isinf(f32 x);
boolean isnan(f32 x);

f64 abs(f64 x);
f64 mod(f64 x, f64 y);
f64 rem(f64 x, f64 y);
f64 exp(f64 x);
f64 exp2(f64 x);
f64 exp10(f64 x);
f64 log(f64 x);
f64 log2(f64 x);
f64 log10(f64 x);
f64 pow(f64 x, f64 y);
f64 sqrt(f64 x);
f64 rsqrt(f64 x);
f64 sin(f64 x);
f64 cos(f64 x);
f64 tan(f64 x);
f64 asin(f64 x);
f64 acos(f64 x);
f64 atan(f64 x);
f64 atan2(f64 y, f64 x);
f64 ceil(f64 x);
f64 floor(f64 x);
f64 trunc(f64 x);
f64 round(f64 x);
boolean isfinite(f64 x);
boolean isinf(f64 x);
boolean isnan(f64 x);

f32 min(f32 a, f32 b);
f32 max(f32 a, f32 b);

f64 min(f64 a, f64 b);
f64 max(f64 a, f64 b);

template<typename T> requires dsl::is_cuj_var_v<T>
T select(
    const boolean &cond,
    const T       &a,
    const T       &b)
{
    T ret;
    $if(cond)
    {
        ret = a;
    }
    $else
    {
        ret = b;
    };
    return ret;
}

template<typename T>
T select(
    const boolean &cond,
    const var<T>  &a,
    const var<T>  &b)
{
    T ret;
    $if(cond)
    {
        ret = a;
    }
    $else
    {
        ret = b;
    };
    return ret;
}

template<typename T>
ref<T> select(
    const boolean &cond,
    const ref<T>  &a,
    const ref<T>  &b)
{
    var pa = a.address();
    var pb = b.address();
    ptr<T> pr;
    $if(cond)
    {
        pr = pa;
    }
    $else
    {
        pr = pb;
    };
    return *pr;
}

CUJ_NAMESPACE_END(cuj::cstd)
