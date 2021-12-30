#include <cuj/cstd/ptx.h>

CUJ_NAMESPACE_BEGIN(cuj::cstd)

i32 thread_idx_x()
{
    return i32::_from_expr(core::CallFunc{
        .intrinsic = core::Intrinsic::thread_idx_x
    });
}

i32 thread_idx_y()
{
    return i32::_from_expr(core::CallFunc{
        .intrinsic = core::Intrinsic::thread_idx_y
    });
}

i32 thread_idx_z()
{
    return i32::_from_expr(core::CallFunc{
        .intrinsic = core::Intrinsic::thread_idx_z
    });
}

i32 block_idx_x()
{
    return i32::_from_expr(core::CallFunc{
        .intrinsic = core::Intrinsic::block_idx_x
    });
}

i32 block_idx_y()
{
    return i32::_from_expr(core::CallFunc{
        .intrinsic = core::Intrinsic::block_idx_y
    });
}

i32 block_idx_z()
{
    return i32::_from_expr(core::CallFunc{
        .intrinsic = core::Intrinsic::block_idx_z
    });
}

i32 block_dim_x()
{
    return i32::_from_expr(core::CallFunc{
        .intrinsic = core::Intrinsic::block_dim_x
    });
}

i32 block_dim_y()
{
    return i32::_from_expr(core::CallFunc{
        .intrinsic = core::Intrinsic::block_dim_y
    });
}

i32 block_dim_z()
{
    return i32::_from_expr(core::CallFunc{
        .intrinsic = core::Intrinsic::block_dim_z
    });
}

CUJ_NAMESPACE_END(cuj::cstd)
