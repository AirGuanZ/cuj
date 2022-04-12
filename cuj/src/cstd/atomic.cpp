#include <cuj/cstd/atomic.h>

CUJ_NAMESPACE_BEGIN(cuj::cstd)

i32 atomic_add(ptr<i32> dst, i32 val)
{
    return i32::_from_expr(core::CallFunc{
        .intrinsic = core::Intrinsic::atomic_add_i32,
        .args = {
            newRC<core::Expr>(dst._load()),
            newRC<core::Expr>(val._load())
        }
    });
}

u32 atomic_add(ptr<u32> dst, u32 val)
{
    return u32::_from_expr(core::CallFunc{
        .intrinsic = core::Intrinsic::atomic_add_u32,
        .args = {
            newRC<core::Expr>(dst._load()),
            newRC<core::Expr>(val._load())
        }
    });
}

f32 atomic_add(ptr<f32> dst, f32 val)
{
    return f32::_from_expr(core::CallFunc{
        .intrinsic = core::Intrinsic::atomic_add_f32,
        .args = {
            newRC<core::Expr>(dst._load()),
            newRC<core::Expr>(val._load())
        }
    });
}

i32 atomic_cmpxchg(ptr<i32> addr, i32 cmp, i32 new_val)
{
    return i32::_from_expr(core::CallFunc{
        .intrinsic = core::Intrinsic::cmpxchg_i32,
        .args = {
            newRC<core::Expr>(addr._load()),
            newRC<core::Expr>(cmp._load()),
            newRC<core::Expr>(new_val._load())
        }
    });
}

u32 atomic_cmpxchg(ptr<u32> addr, u32 cmp, u32 new_val)
{
    return u32::_from_expr(core::CallFunc{
        .intrinsic = core::Intrinsic::cmpxchg_u32,
        .args = {
            newRC<core::Expr>(addr._load()),
            newRC<core::Expr>(cmp._load()),
            newRC<core::Expr>(new_val._load())
        }
    });
}

u64 atomic_cmpxchg(ptr<u64> addr, u64 cmp, u64 new_val)
{
    return u64::_from_expr(core::CallFunc{
        .intrinsic = core::Intrinsic::cmpxchg_u64,
        .args = {
            newRC<core::Expr>(addr._load()),
            newRC<core::Expr>(cmp._load()),
            newRC<core::Expr>(new_val._load())
        }
    });
}

CUJ_NAMESPACE_END(cuj::cstd)
