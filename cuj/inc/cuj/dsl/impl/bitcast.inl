#pragma once

#include <cuj/core/expr.h>
#include <cuj/dsl/arithmetic.h>
#include <cuj/dsl/bitcast.h>
#include <cuj/dsl/function.h>

CUJ_NAMESPACE_BEGIN(cuj::dsl)

namespace bitcast_detail
{

    template<typename To, typename From>
    To bitcast_impl(From from)
    {
        auto func_ctx = FunctionContext::get_func_context();
        auto type_ctx = func_ctx->get_type_context();
        auto cast = core::BitwiseCast{
            .dst_type = type_ctx->get_type<To>(),
            .src_type = type_ctx->get_type<From>(),
            .src_val = newRC<core::Expr>(from._load())
        };
        return To::_from_expr(std::move(cast));
    }

} // namespace bitcast_detail

template<typename To, typename From>
    requires is_cuj_arithmetic_v<To> &&
             is_cuj_arithmetic_v<From> &&
             (sizeof(typename To::RawType) == sizeof(typename From::RawType))
To bitcast(From from)
{
    return bitcast_detail::bitcast_impl<To>(from);
}

template<typename To>
    requires is_cuj_pointer_v<To>
To bitcast(num<uint64_t> from)
{
    return bitcast_detail::bitcast_impl<To>(from);
}

template<typename To>
    requires is_cuj_pointer_v<To>
To bitcast(num<int64_t> from)
{
    return bitcast_detail::bitcast_impl<To>(from);
}

template<typename To, typename From>
    requires is_cuj_arithmetic_v<To> && is_cuj_pointer_v<From>
To bitcast(From from)
{
    static_assert(
        std::is_same_v<To, num<int64_t>> || std::is_same_v<To, num<uint64_t>>);
    return bitcast_detail::bitcast_impl<To>(from);
}

template<typename To, typename From>
    requires is_cuj_ref_v<From>
To bitcast(From from)
{
    remove_reference_t<From> from_val = from;
    return dsl::bitcast<To>(from_val);
}

CUJ_NAMESPACE_END(cuj::dsl)
