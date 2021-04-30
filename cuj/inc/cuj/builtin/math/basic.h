#pragma once

#include <cuj/ast/ast.h>

CUJ_NAMESPACE_BEGIN(cuj::builtin::math)

enum class IntrinsicBasicMathFunctionType
{
    abs,
    mod,
    remainder,
    exp,
    exp2,
    log,
    log2,
    log10,
    pow,
    sqrt,
    sin,
    cos,
    tan,
    asin,
    acos,
    atan,
    atan2,
    ceil,
    floor,
    trunc,
    round,
    isfinite,
    isinf,
    isnan
};

const char *intrinsic_basic_math_function_name(
    IntrinsicBasicMathFunctionType type);

namespace detail
{

    template<typename R, typename F>
    class InternalIntrinsicBasicMathFunction :
        public ast::InternalArithmeticValue<R>
    {
    public:

        IntrinsicBasicMathFunctionType      type;
        RC<ast::InternalArithmeticValue<F>> input1;
        RC<ast::InternalArithmeticValue<F>> input2;

        ir::BasicValue gen_ir(ir::IRBuilder &builder) const override;
    };

} // namespace detail

$float abs       (const $float &x);
$float mod       (const $float &x, const $float &y);
$float remainder (const $float &x, const $float &y);
$float exp       (const $float &x);
$float exp2      (const $float &x);
$float log       (const $float &x);
$float log2      (const $float &x);
$float log10     (const $float &x);
$float pow       (const $float &x, const $float &y);
$float sqrt      (const $float &x);
$float sin       (const $float &x);
$float cos       (const $float &x);
$float tan       (const $float &x);
$float asin      (const $float &x);
$float acos      (const $float &x);
$float atan      (const $float &x);
$float atan2     (const $float &y, const $float &x);
$float ceil      (const $float &x);
$float floor     (const $float &x);
$float trunc     (const $float &x);
$float round     (const $float &x);
$int   isfinite (const $float &x);
$int   isinf    (const $float &x);
$int   isnan    (const $float &x);

$double abs      (const $double &x);
$double mod      (const $double &x, const $double &y);
$double remainder(const $double &x, const $double &y);
$double exp      (const $double &x);
$double exp2     (const $double &x);
$double log      (const $double &x);
$double log2     (const $double &x);
$double log10    (const $double &x);
$double pow      (const $double &x, const $double &y);
$double sqrt     (const $double &x);
$double sin      (const $double &x);
$double cos      (const $double &x);
$double tan      (const $double &x);
$double asin     (const $double &x);
$double acos     (const $double &x);
$double atan     (const $double &x);
$double atan2    (const $double &y, const $double &x);
$double ceil     (const $double &x);
$double floor    (const $double &x);
$double trunc    (const $double &x);
$double round    (const $double &x);
$int    isfinite (const $double &x);
$int    isinf    (const $double &x);
$int    isnan    (const $double &x);

template<typename T>
ArithmeticValue<T> min(
    const ArithmeticValue<T> &lhs, const ArithmeticValue<T> &rhs)
{
    $var(T, ret);
    $if(lhs < rhs)
    {
        ret = lhs;
    }
    $else
    {
        ret = rhs;
    };
    return ret;
}

template<typename T>
ArithmeticValue<T> max(
    const ArithmeticValue<T> &lhs, const ArithmeticValue<T> &rhs)
{
    $var(T, ret);
    $if(lhs > rhs)
    {
        ret = lhs;
    }
    $else
    {
        ret = rhs;
    };
    return ret;
}

CUJ_NAMESPACE_END(cuj::builtin::math)
