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

$float abs       ($float x);
$float mod       ($float x, $float y);
$float remainder ($float x, $float y);
$float exp       ($float x);
$float exp2      ($float x);
$float log       ($float x);
$float log2      ($float x);
$float log10     ($float x);
$float pow       ($float x, $float y);
$float sqrt      ($float x);
$float sin       ($float x);
$float cos       ($float x);
$float tan       ($float x);
$float asin      ($float x);
$float acos      ($float x);
$float atan      ($float x);
$float atan2     ($float y, $float x);
$float ceil      ($float x);
$float floor     ($float x);
$float trunc     ($float x);
$float round     ($float x);
$int    isfinite ($float x);
$int    isinf    ($float x);
$int    isnan    ($float x);

$double abs      ($double x);
$double mod      ($double x, $double y);
$double remainder($double x, $double y);
$double exp      ($double x);
$double exp2     ($double x);
$double log      ($double x);
$double log2     ($double x);
$double log10    ($double x);
$double pow      ($double x, $double y);
$double sqrt     ($double x);
$double sin      ($double x);
$double cos      ($double x);
$double tan      ($double x);
$double asin     ($double x);
$double acos     ($double x);
$double atan     ($double x);
$double atan2    ($double y, $double x);
$double ceil     ($double x);
$double floor    ($double x);
$double trunc    ($double x);
$double round    ($double x);
$int    isfinite ($double x);
$int    isinf    ($double x);
$int    isnan    ($double x);

CUJ_NAMESPACE_END(cuj::builtin::math)
