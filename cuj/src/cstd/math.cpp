#include <cuj/cstd/math.h>

CUJ_NAMESPACE_BEGIN(cuj::cstd)

f32 abs(f32 x)
{
    return f32::_from_expr(core::CallFunc{
        .intrinsic = core::Intrinsic::f32_abs,
        .args      = { newRC<core::Expr>(x._load()) }
    });
}

f32 mod(f32 x, f32 y)
{
    return f32::_from_expr(core::CallFunc{
        .intrinsic = core::Intrinsic::f32_mod,
        .args      = { newRC<core::Expr>(x._load()), newRC<core::Expr>(y._load()) }
    });
}

f32 rem(f32 x, f32 y)
{
    return f32::_from_expr(core::CallFunc{
        .intrinsic = core::Intrinsic::f32_rem,
        .args      = { newRC<core::Expr>(x._load()), newRC<core::Expr>(y._load()) }
    });
}

f32 exp(f32 x)
{
    return f32::_from_expr(core::CallFunc{
        .intrinsic = core::Intrinsic::f32_exp,
        .args      = { newRC<core::Expr>(x._load()) }
    });
}

f32 exp2(f32 x)
{
    return f32::_from_expr(core::CallFunc{
        .intrinsic = core::Intrinsic::f32_exp2,
        .args      = { newRC<core::Expr>(x._load()) }
    });
}

f32 exp10(f32 x)
{
    return f32::_from_expr(core::CallFunc{
        .intrinsic = core::Intrinsic::f32_exp10,
        .args      = { newRC<core::Expr>(x._load()) }
    });
}

f32 log(f32 x)
{
    return f32::_from_expr(core::CallFunc{
        .intrinsic = core::Intrinsic::f32_log,
        .args      = { newRC<core::Expr>(x._load()) }
    });
}

f32 log2(f32 x)
{
    return f32::_from_expr(core::CallFunc{
        .intrinsic = core::Intrinsic::f32_log2,
        .args      = { newRC<core::Expr>(x._load()) }
    });
}

f32 log10(f32 x)
{
    return f32::_from_expr(core::CallFunc{
        .intrinsic = core::Intrinsic::f32_log10,
        .args      = { newRC<core::Expr>(x._load()) }
    });
}

f32 pow(f32 x, f32 y)
{
    return f32::_from_expr(core::CallFunc{
        .intrinsic = core::Intrinsic::f32_pow,
        .args      = { newRC<core::Expr>(x._load()), newRC<core::Expr>(y._load()) }
    });
}

f32 sqrt(f32 x)
{
    return f32::_from_expr(core::CallFunc{
        .intrinsic = core::Intrinsic::f32_sqrt,
        .args      = { newRC<core::Expr>(x._load()) }
    });
}

f32 rsqrt(f32 x)
{
    return f32::_from_expr(core::CallFunc{
        .intrinsic = core::Intrinsic::f32_rsqrt,
        .args      = { newRC<core::Expr>(x._load()) }
    });
}

f32 sin(f32 x)
{
    return f32::_from_expr(core::CallFunc{
        .intrinsic = core::Intrinsic::f32_sin,
        .args      = { newRC<core::Expr>(x._load()) }
    });
}

f32 cos(f32 x)
{
    return f32::_from_expr(core::CallFunc{
        .intrinsic = core::Intrinsic::f32_cos,
        .args      = { newRC<core::Expr>(x._load()) }
    });
}

f32 tan(f32 x)
{
    return f32::_from_expr(core::CallFunc{
        .intrinsic = core::Intrinsic::f32_tan,
        .args      = { newRC<core::Expr>(x._load()) }
    });
}

f32 asin(f32 x)
{
    return f32::_from_expr(core::CallFunc{
        .intrinsic = core::Intrinsic::f32_asin,
        .args      = { newRC<core::Expr>(x._load()) }
    });
}

f32 acos(f32 x)
{
    return f32::_from_expr(core::CallFunc{
        .intrinsic = core::Intrinsic::f32_acos,
        .args      = { newRC<core::Expr>(x._load()) }
    });
}

f32 atan(f32 x)
{
    return f32::_from_expr(core::CallFunc{
        .intrinsic = core::Intrinsic::f32_atan,
        .args      = { newRC<core::Expr>(x._load()) }
    });
}

f32 atan2(f32 y, f32 x)
{
    return f32::_from_expr(core::CallFunc{
        .intrinsic = core::Intrinsic::f32_atan2,
        .args      = { newRC<core::Expr>(y._load()), newRC<core::Expr>(x._load()) }
    });
}

f32 ceil(f32 x)
{
    return f32::_from_expr(core::CallFunc{
        .intrinsic = core::Intrinsic::f32_ceil,
        .args      = { newRC<core::Expr>(x._load()) }
    });
}

f32 floor(f32 x)
{
    return f32::_from_expr(core::CallFunc{
        .intrinsic = core::Intrinsic::f32_floor,
        .args      = { newRC<core::Expr>(x._load()) }
    });
}

f32 trunc(f32 x)
{
    return f32::_from_expr(core::CallFunc{
        .intrinsic = core::Intrinsic::f32_trunc,
        .args      = { newRC<core::Expr>(x._load()) }
    });
}

f32 round(f32 x)
{
    return f32::_from_expr(core::CallFunc{
        .intrinsic = core::Intrinsic::f32_round,
        .args      = { newRC<core::Expr>(x._load()) }
    });
}

boolean isfinite(f32 x)
{
    return i32::_from_expr(core::CallFunc{
        .intrinsic = core::Intrinsic::f32_isfinite,
        .args      = { newRC<core::Expr>(x._load()) }
    }) != i32(0);
}

boolean isinf(f32 x)
{
    return i32::_from_expr(core::CallFunc{
        .intrinsic = core::Intrinsic::f32_isinf,
        .args      = { newRC<core::Expr>(x._load()) }
    }) != i32(0);
}

boolean isnan(f32 x)
{
    return i32::_from_expr(core::CallFunc{
        .intrinsic = core::Intrinsic::f32_isnan,
        .args      = { newRC<core::Expr>(x._load()) }
    }) != i32(0);
}

f64 abs(f64 x)
{
    return f64::_from_expr(core::CallFunc{
        .intrinsic = core::Intrinsic::f64_abs,
        .args      = { newRC<core::Expr>(x._load()) }
    });
}

f64 mod(f64 x, f64 y)
{
    return f64::_from_expr(core::CallFunc{
        .intrinsic = core::Intrinsic::f64_mod,
        .args      = { newRC<core::Expr>(x._load()), newRC<core::Expr>(y._load()) }
    });
}

f64 rem(f64 x, f64 y)
{
    return f64::_from_expr(core::CallFunc{
        .intrinsic = core::Intrinsic::f64_rem,
        .args      = { newRC<core::Expr>(x._load()), newRC<core::Expr>(y._load()) }
    });
}

f64 exp(f64 x)
{
    return f64::_from_expr(core::CallFunc{
        .intrinsic = core::Intrinsic::f64_exp,
        .args      = { newRC<core::Expr>(x._load()) }
    });
}

f64 exp2(f64 x)
{
    return f64::_from_expr(core::CallFunc{
        .intrinsic = core::Intrinsic::f64_exp2,
        .args      = { newRC<core::Expr>(x._load()) }
    });
}

f64 exp10(f64 x)
{
    return f64::_from_expr(core::CallFunc{
        .intrinsic = core::Intrinsic::f64_exp10,
        .args      = { newRC<core::Expr>(x._load()) }
    });
}

f64 log(f64 x)
{
    return f64::_from_expr(core::CallFunc{
        .intrinsic = core::Intrinsic::f64_log,
        .args      = { newRC<core::Expr>(x._load()) }
    });
}

f64 log2(f64 x)
{
    return f64::_from_expr(core::CallFunc{
        .intrinsic = core::Intrinsic::f64_log2,
        .args      = { newRC<core::Expr>(x._load()) }
    });
}

f64 log10(f64 x)
{
    return f64::_from_expr(core::CallFunc{
        .intrinsic = core::Intrinsic::f64_log10,
        .args      = { newRC<core::Expr>(x._load()) }
    });
}

f64 pow(f64 x, f64 y)
{
    return f64::_from_expr(core::CallFunc{
        .intrinsic = core::Intrinsic::f64_pow,
        .args      = { newRC<core::Expr>(x._load()), newRC<core::Expr>(y._load()) }
    });
}

f64 sqrt(f64 x)
{
    return f64::_from_expr(core::CallFunc{
        .intrinsic = core::Intrinsic::f64_sqrt,
        .args      = { newRC<core::Expr>(x._load()) }
    });
}

f64 rsqrt(f64 x)
{
    return f64::_from_expr(core::CallFunc{
        .intrinsic = core::Intrinsic::f64_rsqrt,
        .args      = { newRC<core::Expr>(x._load()) }
    });
}

f64 sin(f64 x)
{
    return f64::_from_expr(core::CallFunc{
        .intrinsic = core::Intrinsic::f64_sin,
        .args      = { newRC<core::Expr>(x._load()) }
    });
}

f64 cos(f64 x)
{
    return f64::_from_expr(core::CallFunc{
        .intrinsic = core::Intrinsic::f64_cos,
        .args      = { newRC<core::Expr>(x._load()) }
    });
}

f64 tan(f64 x)
{
    return f64::_from_expr(core::CallFunc{
        .intrinsic = core::Intrinsic::f64_tan,
        .args      = { newRC<core::Expr>(x._load()) }
    });
}

f64 asin(f64 x)
{
    return f64::_from_expr(core::CallFunc{
        .intrinsic = core::Intrinsic::f64_asin,
        .args      = { newRC<core::Expr>(x._load()) }
    });
}

f64 acos(f64 x)
{
    return f64::_from_expr(core::CallFunc{
        .intrinsic = core::Intrinsic::f64_acos,
        .args      = { newRC<core::Expr>(x._load()) }
    });
}

f64 atan(f64 x)
{
    return f64::_from_expr(core::CallFunc{
        .intrinsic = core::Intrinsic::f64_atan,
        .args      = { newRC<core::Expr>(x._load()) }
    });
}

f64 atan2(f64 y, f64 x)
{
    return f64::_from_expr(core::CallFunc{
        .intrinsic = core::Intrinsic::f64_atan2,
        .args      = { newRC<core::Expr>(y._load()), newRC<core::Expr>(x._load()) }
    });
}

f64 ceil(f64 x)
{
    return f64::_from_expr(core::CallFunc{
        .intrinsic = core::Intrinsic::f64_ceil,
        .args      = { newRC<core::Expr>(x._load()) }
    });
}

f64 floor(f64 x)
{
    return f64::_from_expr(core::CallFunc{
        .intrinsic = core::Intrinsic::f64_floor,
        .args      = { newRC<core::Expr>(x._load()) }
    });
}

f64 trunc(f64 x)
{
    return f64::_from_expr(core::CallFunc{
        .intrinsic = core::Intrinsic::f64_trunc,
        .args      = { newRC<core::Expr>(x._load()) }
    });
}

f64 round(f64 x)
{
    return f64::_from_expr(core::CallFunc{
        .intrinsic = core::Intrinsic::f64_round,
        .args      = { newRC<core::Expr>(x._load()) }
    });
}

boolean isfinite(f64 x)
{
    return i32::_from_expr(core::CallFunc{
        .intrinsic = core::Intrinsic::f64_isfinite,
        .args      = { newRC<core::Expr>(x._load()) }
    }) != i32(0);
}

boolean isinf(f64 x)
{
    return i32::_from_expr(core::CallFunc{
        .intrinsic = core::Intrinsic::f64_isinf,
        .args      = { newRC<core::Expr>(x._load()) }
    }) != i32(0);
}

boolean isnan(f64 x)
{
    return i32::_from_expr(core::CallFunc{
        .intrinsic = core::Intrinsic::f64_isnan,
        .args      = { newRC<core::Expr>(x._load()) }
    }) != i32(0);
}

i32 min(i32 a, i32 b)
{
    return i32::_from_expr(core::CallFunc{
        .intrinsic = core::Intrinsic::i32_min,
        .args      = { newRC<core::Expr>(a._load()), newRC<core::Expr>(b._load()) }
    });
}

i32 max(i32 a, i32 b)
{
    return i32::_from_expr(core::CallFunc{
        .intrinsic = core::Intrinsic::i32_max,
        .args      = { newRC<core::Expr>(a._load()), newRC<core::Expr>(b._load()) }
    });
}

i64 min(i64 a, i64 b)
{
    return i64::_from_expr(core::CallFunc{
        .intrinsic = core::Intrinsic::i64_min,
        .args      = { newRC<core::Expr>(a._load()), newRC<core::Expr>(b._load()) }
    });
}

i64 max(i64 a, i64 b)
{
    return i64::_from_expr(core::CallFunc{
        .intrinsic = core::Intrinsic::i64_max,
        .args      = { newRC<core::Expr>(a._load()), newRC<core::Expr>(b._load()) }
    });
}

f32 min(f32 a, f32 b)
{
    return f32::_from_expr(core::CallFunc{
        .intrinsic = core::Intrinsic::f32_min,
        .args      = { newRC<core::Expr>(a._load()), newRC<core::Expr>(b._load()) }
    });
}

f32 max(f32 a, f32 b)
{
    return f32::_from_expr(core::CallFunc{
        .intrinsic = core::Intrinsic::f32_max,
        .args      = { newRC<core::Expr>(a._load()), newRC<core::Expr>(b._load()) }
    });
}

f64 min(f64 a, f64 b)
{
    return f64::_from_expr(core::CallFunc{
        .intrinsic = core::Intrinsic::f64_min,
        .args      = { newRC<core::Expr>(a._load()), newRC<core::Expr>(b._load()) }
    });
}

f64 max(f64 a, f64 b)
{
    return f64::_from_expr(core::CallFunc{
        .intrinsic = core::Intrinsic::f64_max,
        .args      = { newRC<core::Expr>(a._load()), newRC<core::Expr>(b._load()) }
    });
}

f32 clamp(f32 x, f32 minv, f32 maxv)
{
    return max(minv, min(x, maxv));
}

f64 clamp(f64 x, f64 minv, f64 maxv)
{
    return max(minv, min(x, maxv));
}

i32 clamp(i32 x, i32 minv, i32 maxv)
{
    return max(minv, min(x, maxv));
}

i64 clamp(i64 x, i64 minv, i64 maxv)
{
    return max(minv, min(x, maxv));
}

f32 saturate(f32 v)
{
    return f32::_from_expr(core::CallFunc{
        .intrinsic = core::Intrinsic::f32_saturate,
        .args      = { newRC<core::Expr>(v._load()) }
    });
}

f64 saturate(f64 v)
{
    return f64::_from_expr(core::CallFunc{
        .intrinsic = core::Intrinsic::f64_saturate,
        .args      = { newRC<core::Expr>(v._load()) }
    });
}

CUJ_NAMESPACE_END(cuj::cstd)
