#include <cuj/cstd/memory.h>

CUJ_NAMESPACE_BEGIN(cuj::cstd)

namespace detail
{

    template<typename T>
    void create_vectorized_store(
        core::Intrinsic intrinsic, ptr<T> addr, std::initializer_list<T> vals)
    {
        std::vector<RC<core::Expr>> args = { newRC<core::Expr>(addr._load()) };
        args.reserve(1 + vals.size());
        for(auto &v : vals)
            args.push_back(newRC<core::Expr>(v._load()));
        dsl::FunctionContext::get_func_context()->append_statement(
            core::CallFuncStat{
                .call_expr = core::CallFunc{
                    .intrinsic = intrinsic,
                    .args      = std::move(args)
                }
            });
    }

    template<typename T>
    void create_vectorized_load(
        core::Intrinsic intrinsic, ptr<T> addr, std::initializer_list<ref<T>> vals)
    {
        std::vector<RC<core::Expr>> args = { newRC<core::Expr>(addr._load()) };
        args.reserve(1 + vals.size());
        for(auto &v : vals)
            args.push_back(newRC<core::Expr>(v.address()._load()));
        dsl::FunctionContext::get_func_context()->append_statement(
            core::CallFuncStat{
                .call_expr = core::CallFunc{
                    .intrinsic = intrinsic,
                    .args      = std::move(args)
                }
            });
    }

} // namespace detail

void store_f32x4(ptr<f32> addr, f32 a, f32 b, f32 c, f32 d)
{
    detail::create_vectorized_store(
        core::Intrinsic::store_f32x4, addr, { a, b, c, d });
}

void store_u32x4(ptr<u32> addr, u32 a, u32 b, u32 c, u32 d)
{
    detail::create_vectorized_store(
        core::Intrinsic::store_u32x4, addr, { a, b, c, d });
}

void store_i32x4(ptr<i32> addr, i32 a, i32 b, i32 c, i32 d)
{
    detail::create_vectorized_store(
        core::Intrinsic::store_i32x4, addr, { a, b, c, d });
}

void store_f32x3(ptr<f32> addr, f32 a, f32 b, f32 c)
{
    detail::create_vectorized_store(
        core::Intrinsic::store_f32x3, addr, { a, b, c });
}

void store_u32x3(ptr<u32> addr, u32 a, u32 b, u32 c)
{
    detail::create_vectorized_store(
        core::Intrinsic::store_u32x3, addr, { a, b, c });
}

void store_i32x3(ptr<i32> addr, i32 a, i32 b, i32 c)
{
    detail::create_vectorized_store(
        core::Intrinsic::store_i32x3, addr, { a, b, c });
}

void store_f32x2(ptr<f32> addr, f32 a, f32 b)
{
    detail::create_vectorized_store(
        core::Intrinsic::store_f32x2, addr, { a, b });
}

void store_u32x2(ptr<u32> addr, u32 a, u32 b)
{
    detail::create_vectorized_store(
        core::Intrinsic::store_u32x2, addr, { a, b });
}

void store_i32x2(ptr<i32> addr, i32 a, i32 b)
{
    detail::create_vectorized_store(
        core::Intrinsic::store_i32x2, addr, { a, b });
}

void load_f32x4(ptr<f32> addr, ref<f32> a, ref<f32> b, ref<f32> c, ref<f32> d)
{
    detail::create_vectorized_load(
        core::Intrinsic::load_f32x4, addr, { a, b, c, d });
}

void load_u32x4(ptr<u32> addr, ref<u32> a, ref<u32> b, ref<u32> c, ref<u32> d)
{
    detail::create_vectorized_load(
        core::Intrinsic::load_u32x4, addr, { a, b, c, d });
}

void load_i32x4(ptr<i32> addr, ref<i32> a, ref<i32> b, ref<i32> c, ref<i32> d)
{
    detail::create_vectorized_load(
        core::Intrinsic::load_i32x4, addr, { a, b, c, d });
}

void load_f32x3(ptr<f32> addr, ref<f32> a, ref<f32> b, ref<f32> c)
{
    detail::create_vectorized_load(
        core::Intrinsic::load_f32x3, addr, { a, b, c });
}

void load_u32x3(ptr<u32> addr, ref<u32> a, ref<u32> b, ref<u32> c)
{
    detail::create_vectorized_load(
        core::Intrinsic::load_u32x3, addr, { a, b, c });
}

void load_i32x3(ptr<i32> addr, ref<i32> a, ref<i32> b, ref<i32> c)
{
    detail::create_vectorized_load(
        core::Intrinsic::load_i32x3, addr, { a, b, c });
}

void load_f32x2(ptr<f32> addr, ref<f32> a, ref<f32> b)
{
    detail::create_vectorized_load(
        core::Intrinsic::load_f32x2, addr, { a, b });
}

void load_u32x2(ptr<u32> addr, ref<u32> a, ref<u32> b)
{
    detail::create_vectorized_load(
        core::Intrinsic::load_u32x2, addr, { a, b });
}

void load_i32x2(ptr<i32> addr, ref<i32> a, ref<i32> b)
{
    detail::create_vectorized_load(
        core::Intrinsic::load_i32x2, addr, { a, b });
}

void _memcpy_impl(ptr<u8> dst, ptr<u8> src, u64 bytes)
{
    auto call_expr = core::CallFunc{
            .intrinsic = core::Intrinsic::memcpy,
            .args = {
                newRC<core::Expr>(dst._load()),
                newRC<core::Expr>(src._load()),
                newRC<core::Expr>(bytes._load())
            }
    };
    auto call = core::CallFuncStat{ .call_expr = std::move(call_expr) };
    dsl::FunctionContext::get_func_context()->append_statement(std::move(call));
}

CUJ_NAMESPACE_END(cuj::cstd)
