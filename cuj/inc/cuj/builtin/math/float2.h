#pragma once

#include <cuj/ast/ast.h>

CUJ_NAMESPACE_BEGIN(cuj::builtin::math)

namespace detail
{

    class Float2Impl : public ast::ClassBase<Float2Impl>
    {
    public:

        $mem(float, x);
        $mem(float, y);

        using ClassBase::ClassBase;

        explicit Float2Impl(ClassAddress addr)
            : Float2Impl(addr, 0, 0)
        {
            
        }

        Float2Impl(ClassAddress addr, $float v)
            : Float2Impl(addr, v, v)
        {
            
        }

        Float2Impl(ClassAddress addr, $float _x, $float _y)
            : ClassBase<Float2Impl>(addr)
        {
            x = _x;
            y = _y;
        }
    };

} // namespace detail

using Float2 = ast::Var<detail::Float2Impl>;

CUJ_NAMESPACE_END(cuj::builtin::math)

CUJ_NAMESPACE_BEGIN(cuj::ast)

ClassValue(builtin::math::Float2) ->
    ClassValue<builtin::math::detail::Float2Impl>;

CUJ_NAMESPACE_END(cuj::ast)

CUJ_NAMESPACE_BEGIN(cuj::builtin::math)

inline Float2 make_float2($float x, $float y)
{
    $var(Float2, ret, x, y);
    return ret;
}

inline Float2 make_float2($float v)
{
    return make_float2(v, v);
}

inline Float2 make_float2()
{
    return make_float2(0);
}

inline $float dot(const Float2 &a, const Float2 &b)
{
    return a->x * b->x + a->y * b->y;
}

CUJ_NAMESPACE_END(cuj::builtin::math)
