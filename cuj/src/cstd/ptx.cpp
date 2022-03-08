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

void sample_texture2d_1f(u64 texture_object, f32 u, f32 v, ref<f32> r)
{
    f32 g, b, a;
    sample_texture2d_4f(texture_object, u, v, r, g, b, a);
}

void sample_texture2d_3f(u64 texture_object, f32 u, f32 v, ref<f32> r, ref<f32> g, ref<f32> b)
{
    f32 a;
    sample_texture2d_4f(texture_object, u, v, r, g, b, a);
}

void sample_texture2d_4f(u64 texture_object, f32 u, f32 v, ref<f32> r, ref<f32> g, ref<f32> b, ref<f32> a)
{
    auto func = dsl::FunctionContext::get_func_context();
    auto call = core::CallFunc{
        .intrinsic = core::Intrinsic::sample_tex_2d_f32,
        .args = {
            newRC<core::Expr>(texture_object._load()),
            newRC<core::Expr>(u._load()),
            newRC<core::Expr>(v._load()),
            newRC<core::Expr>(r.address()._load()),
            newRC<core::Expr>(g.address()._load()),
            newRC<core::Expr>(b.address()._load()),
            newRC<core::Expr>(a.address()._load())
        }
    };
    auto stat = core::CallFuncStat{ std::move(call) };
    func->append_statement(std::move(stat));
}

void sample_texture2d_1i(u64 texture_object, f32 u, f32 v, ref<i32> r)
{
    i32 g, b, a;
    sample_texture2d_4i(texture_object, u, v, r, g, b, a);
}

void sample_texture2d_3i(u64 texture_object, f32 u, f32 v, ref<i32> r, ref<i32> g, ref<i32> b)
{
    i32 a;
    sample_texture2d_4i(texture_object, u, v, r, g, b, a);
}

void sample_texture2d_4i(u64 texture_object, f32 u, f32 v, ref<i32> r, ref<i32> g, ref<i32> b, ref<i32> a)
{
    auto func = dsl::FunctionContext::get_func_context();
    auto call = core::CallFunc{
        .intrinsic = core::Intrinsic::sample_tex_2d_i32,
        .args = {
            newRC<core::Expr>(texture_object._load()),
            newRC<core::Expr>(u._load()),
            newRC<core::Expr>(v._load()),
            newRC<core::Expr>(r.address()._load()),
            newRC<core::Expr>(g.address()._load()),
            newRC<core::Expr>(b.address()._load()),
            newRC<core::Expr>(a.address()._load())
        }
    };
    auto stat = core::CallFuncStat{ std::move(call) };
    func->append_statement(std::move(stat));
}

void sample_texture3d_1f(u64 texture_object, f32 u, f32 v, f32 w, ref<f32> r)
{
    f32 g, b, a;
    sample_texture3d_4f(texture_object, u, v, w, r, g, b, a);
}

void sample_texture3d_3f(u64 texture_object, f32 u, f32 v, f32 w, ref<f32> r, ref<f32> g, ref<f32> b)
{
    f32 a;
    sample_texture3d_4f(texture_object, u, v, w, r, g, b, a);
}

void sample_texture3d_4f(u64 texture_object, f32 u, f32 v, f32 w, ref<f32> r, ref<f32> g, ref<f32> b, ref<f32> a)
{
    auto func = dsl::FunctionContext::get_func_context();
    auto call = core::CallFunc{
        .intrinsic = core::Intrinsic::sample_tex_3d_f32,
        .args = {
            newRC<core::Expr>(texture_object._load()),
            newRC<core::Expr>(u._load()),
            newRC<core::Expr>(v._load()),
            newRC<core::Expr>(w._load()),
            newRC<core::Expr>(r.address()._load()),
            newRC<core::Expr>(g.address()._load()),
            newRC<core::Expr>(b.address()._load()),
            newRC<core::Expr>(a.address()._load())
        }
    };
    auto stat = core::CallFuncStat{ std::move(call) };
    func->append_statement(std::move(stat));
}

void sample_texture3d_1i(u64 texture_object, f32 u, f32 v, f32 w, ref<i32> r)
{
    i32 g, b, a;
    sample_texture3d_4i(texture_object, u, v, w, r, g, b, a);
}

void sample_texture3d_3i(u64 texture_object, f32 u, f32 v, f32 w, ref<i32> r, ref<i32> g, ref<i32> b)
{
    i32 a;
    sample_texture3d_4i(texture_object, u, v, w, r, g, b, a);
}

void sample_texture3d_4i(u64 texture_object, f32 u, f32 v, f32 w, ref<i32> r, ref<i32> g, ref<i32> b, ref<i32> a)
{
    auto func = dsl::FunctionContext::get_func_context();
    auto call = core::CallFunc{
        .intrinsic = core::Intrinsic::sample_tex_3d_i32,
        .args = {
            newRC<core::Expr>(texture_object._load()),
            newRC<core::Expr>(u._load()),
            newRC<core::Expr>(v._load()),
            newRC<core::Expr>(w._load()),
            newRC<core::Expr>(r.address()._load()),
            newRC<core::Expr>(g.address()._load()),
            newRC<core::Expr>(b.address()._load()),
            newRC<core::Expr>(a.address()._load())
        }
    };
    auto stat = core::CallFuncStat{ std::move(call) };
    func->append_statement(std::move(stat));
}

CUJ_NAMESPACE_END(cuj::cstd)
