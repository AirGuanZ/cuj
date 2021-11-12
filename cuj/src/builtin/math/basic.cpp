#include <cuj/builtin/math/basic.h>

CUJ_NAMESPACE_BEGIN(cuj::builtin::math)

namespace detail
{

    template<typename T>
    const char *intrinsic_math_comp_name()
    {
        if constexpr(std::is_same_v<T, int32_t>)
            return "i32";
        else if constexpr(std::is_same_v<T, float>)
            return "f32";
        else if constexpr(std::is_same_v<T, double>)
            return "f64";
        else
        {
            static_assert(std::is_same_v<T, int64_t>);
            return "i64";
        }
    }

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

    template<typename R, typename F>
    ir::BasicValue InternalIntrinsicBasicMathFunction<R, F>::gen_ir(
        ir::IRBuilder &builder) const
    {
        auto input1_val = input1->gen_ir(builder);

        std::string name = intrinsic_basic_math_function_name(type);
        auto intrinsic = ir::IntrinsicOp{
            name + "." + intrinsic_math_comp_name<F>(), {input1_val}
        };

        if(input2)
        {
            auto input2_val = input2->gen_ir(builder);
            intrinsic.args.push_back(input2_val);
        }

        auto ret_type = ast::get_current_context()->get_type<R>();
        auto ret = builder.gen_temp_value(ret_type);

        builder.append_assign(ret, intrinsic);
        return ret;
    }

} // namespace detail

namespace
{

    template<typename R, typename F>
    Value<R> create(
        IntrinsicBasicMathFunctionType      type,
        RC<ast::InternalArithmeticValue<F>> input1,
        RC<ast::InternalArithmeticValue<F>> input2 = {})
    {
        auto impl = newRC<detail::InternalIntrinsicBasicMathFunction<R, F>>();
        impl->type   = type;
        impl->input1 = std::move(input1);
        impl->input2 = std::move(input2);
        return ast::Value<R>(std::move(impl));
    }

} // namespace anonymous

const char *intrinsic_basic_math_function_name(
    IntrinsicBasicMathFunctionType type)
{
#define CUJ_INTRINSIC_NAME(NAME) \
    case IntrinsicBasicMathFunctionType::NAME: return "math." #NAME;

    switch(type)
    {
    CUJ_INTRINSIC_NAME(abs)
    CUJ_INTRINSIC_NAME(mod)
    CUJ_INTRINSIC_NAME(remainder)
    CUJ_INTRINSIC_NAME(exp)
    CUJ_INTRINSIC_NAME(exp2)
    CUJ_INTRINSIC_NAME(exp10)
    CUJ_INTRINSIC_NAME(log)
    CUJ_INTRINSIC_NAME(log2)
    CUJ_INTRINSIC_NAME(log10)
    CUJ_INTRINSIC_NAME(pow)
    CUJ_INTRINSIC_NAME(sqrt)
    CUJ_INTRINSIC_NAME(rsqrt)
    CUJ_INTRINSIC_NAME(sin)
    CUJ_INTRINSIC_NAME(cos)
    CUJ_INTRINSIC_NAME(tan)
    CUJ_INTRINSIC_NAME(asin)
    CUJ_INTRINSIC_NAME(acos)
    CUJ_INTRINSIC_NAME(atan)
    CUJ_INTRINSIC_NAME(atan2)
    CUJ_INTRINSIC_NAME(ceil)
    CUJ_INTRINSIC_NAME(floor)
    CUJ_INTRINSIC_NAME(trunc)
    CUJ_INTRINSIC_NAME(round)
    CUJ_INTRINSIC_NAME(isfinite)
    CUJ_INTRINSIC_NAME(isinf)
    CUJ_INTRINSIC_NAME(isnan)
    CUJ_INTRINSIC_NAME(min)
    CUJ_INTRINSIC_NAME(max)
    }
    unreachable();

#undef CUJ_INTRINSIC_NAME
}

#define CUJ_MATH_FUNC_1_ARG(NAME, RET, ARG)                                     \
    Variable<RET> NAME(const Variable<ARG> &x)                                  \
    {                                                                           \
        return create<RET, ARG>(                                                \
            IntrinsicBasicMathFunctionType::NAME, x.get_impl());                \
    }

#define CUJ_MATH_FUNC_2_ARG(NAME, RET, ARG, ARG2)                               \
    Variable<RET> NAME(const Variable<ARG> &x, const Variable<ARG> &y)          \
    {                                                                           \
        static_assert(std::is_same_v<ARG, ARG2>);                               \
        return create<RET, ARG>(                                                \
            IntrinsicBasicMathFunctionType::NAME, x.get_impl(), y.get_impl());  \
    }

CUJ_MATH_FUNC_1_ARG(abs,       float, float)
CUJ_MATH_FUNC_2_ARG(mod,       float, float, float)
CUJ_MATH_FUNC_2_ARG(remainder, float, float, float)
CUJ_MATH_FUNC_1_ARG(exp,       float, float)
CUJ_MATH_FUNC_1_ARG(exp2,      float, float)
CUJ_MATH_FUNC_1_ARG(exp10,     float, float)
CUJ_MATH_FUNC_1_ARG(log,       float, float)
CUJ_MATH_FUNC_1_ARG(log2,      float, float)
CUJ_MATH_FUNC_1_ARG(log10,     float, float)
CUJ_MATH_FUNC_2_ARG(pow,       float, float, float)
CUJ_MATH_FUNC_1_ARG(sqrt,      float, float)
CUJ_MATH_FUNC_1_ARG(rsqrt,     float, float)
CUJ_MATH_FUNC_1_ARG(sin,       float, float)
CUJ_MATH_FUNC_1_ARG(cos,       float, float)
CUJ_MATH_FUNC_1_ARG(tan,       float, float)
CUJ_MATH_FUNC_1_ARG(asin,      float, float)
CUJ_MATH_FUNC_1_ARG(acos,      float, float)
CUJ_MATH_FUNC_1_ARG(atan,      float, float)
CUJ_MATH_FUNC_2_ARG(atan2,     float, float, float)
CUJ_MATH_FUNC_1_ARG(ceil,      float, float)
CUJ_MATH_FUNC_1_ARG(floor,     float, float)
CUJ_MATH_FUNC_1_ARG(trunc,     float, float)
CUJ_MATH_FUNC_1_ARG(round,     float, float)
CUJ_MATH_FUNC_1_ARG(isfinite,  int,   float)
CUJ_MATH_FUNC_1_ARG(isinf,     int,   float)
CUJ_MATH_FUNC_1_ARG(isnan,     int,   float)

CUJ_MATH_FUNC_1_ARG(abs,       double, double)
CUJ_MATH_FUNC_2_ARG(mod,       double, double, double)
CUJ_MATH_FUNC_2_ARG(remainder, double, double, double)
CUJ_MATH_FUNC_1_ARG(exp,       double, double)
CUJ_MATH_FUNC_1_ARG(exp2,      double, double)
CUJ_MATH_FUNC_1_ARG(exp10,     double, double)
CUJ_MATH_FUNC_1_ARG(log,       double, double)
CUJ_MATH_FUNC_1_ARG(log2,      double, double)
CUJ_MATH_FUNC_1_ARG(log10,     double, double)
CUJ_MATH_FUNC_2_ARG(pow,       double, double, double)
CUJ_MATH_FUNC_1_ARG(sqrt,      double, double)
CUJ_MATH_FUNC_1_ARG(rsqrt,     double, double)
CUJ_MATH_FUNC_1_ARG(sin,       double, double)
CUJ_MATH_FUNC_1_ARG(cos,       double, double)
CUJ_MATH_FUNC_1_ARG(tan,       double, double)
CUJ_MATH_FUNC_1_ARG(asin,      double, double)
CUJ_MATH_FUNC_1_ARG(acos,      double, double)
CUJ_MATH_FUNC_1_ARG(atan,      double, double)
CUJ_MATH_FUNC_2_ARG(atan2,     double, double, double)
CUJ_MATH_FUNC_1_ARG(ceil,      double, double)
CUJ_MATH_FUNC_1_ARG(floor,     double, double)
CUJ_MATH_FUNC_1_ARG(trunc,     double, double)
CUJ_MATH_FUNC_1_ARG(round,     double, double)
CUJ_MATH_FUNC_1_ARG(isfinite,  int,   double)
CUJ_MATH_FUNC_1_ARG(isinf,     int,   double)
CUJ_MATH_FUNC_1_ARG(isnan,     int,   double)

CUJ_MATH_FUNC_2_ARG(min, int32_t, int32_t, int32_t)
CUJ_MATH_FUNC_2_ARG(min, int64_t, int64_t, int64_t)
CUJ_MATH_FUNC_2_ARG(min, float,   float,   float)
CUJ_MATH_FUNC_2_ARG(min, double,  double,  double)

CUJ_MATH_FUNC_2_ARG(max, int32_t, int32_t, int32_t)
CUJ_MATH_FUNC_2_ARG(max, int64_t, int64_t, int64_t)
CUJ_MATH_FUNC_2_ARG(max, float,   float,   float)
CUJ_MATH_FUNC_2_ARG(max, double,  double,  double)

CUJ_NAMESPACE_END(cuj::builtin::math)
