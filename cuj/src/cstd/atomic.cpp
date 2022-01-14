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

CUJ_NAMESPACE_END(cuj::cstd)
