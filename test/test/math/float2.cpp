#include <cmath>

#include <test/test.h>

using math::Float2;
using math::make_float2;

struct Float2Data
{
    float x, y;
};

#define ADD_TEST_EXPR(EXPR)                                                     \
    do {                                                                        \
        ScopedContext ctx;                                                      \
        auto approx_eq_f = to_callable<bool>(                                   \
            [](const $float &a, const $float &b)                                \
        {                                                                       \
            $return(math::abs(a - b) < 0.001f);                                 \
        });                                                                     \
        auto approx_eq = to_callable<bool>(                                     \
            [](const Float2 &a, const Float2 &b)                                \
        {                                                                       \
            $return(                                                            \
                math::abs(a->x - b->x) < 0.001f &&                              \
                math::abs(a->y - b->y) < 0.001f);                               \
        });                                                                     \
        auto test = to_callable<bool>(                                          \
            [&]                                                                 \
        {                                                                       \
            $return(EXPR);                                                      \
        });                                                                     \
        auto jit = ctx.gen_native_jit();                                        \
        auto func = jit.get_symbol(test);                                       \
        REQUIRE(func);                                                          \
        if(func)                                                                \
        {                                                                       \
            REQUIRE(func() == true);                                            \
        }                                                                       \
    } while(false)

TEST_CASE("builtin.math.float2")
{
    SECTION("create")
    {
        ScopedContext ctx;

        auto test_make_float2_0 = to_callable<Float2>(
            []()
        {
            $return(make_float2());
        });

        auto test_make_float2_1 = to_callable<Float2>(
            []($f32 v)
        {
            $return(make_float2(v));
        });

        auto test_make_float2_2 = to_callable<Float2>(
            []($f32 x, $f32 y)
        {
            $return(make_float2(x, y));
        });

        auto jit = ctx.gen_native_jit();

        Float2Data float2_data = { 1, 2 };
        jit.get_symbol(test_make_float2_0)(&float2_data);
        REQUIRE(float2_data.x == Approx(0));
        REQUIRE(float2_data.y == Approx(0));

        float2_data = { 1, 2 };
        jit.get_symbol(test_make_float2_1)(&float2_data, 3);
        REQUIRE(float2_data.x == Approx(3));
        REQUIRE(float2_data.y == Approx(3));

        float2_data = { 1, 2 };
        jit.get_symbol(test_make_float2_2)(&float2_data, 3, 4);
        REQUIRE(float2_data.x == Approx(3));
        REQUIRE(float2_data.y == Approx(4));
    }

    SECTION("function")
    {
        ADD_TEST_EXPR(
            approx_eq_f(make_float2(1, 2)->length_square(), 5));

        ADD_TEST_EXPR(
            approx_eq_f(make_float2(1, 2)->length(), std::sqrt(5.0f)));

        ADD_TEST_EXPR(
            approx_eq_f(make_float2(1, 2)->min_elem(), 1));

        ADD_TEST_EXPR(
            approx_eq_f(make_float2(1, 2)->max_elem(), 2));

        ADD_TEST_EXPR(
            approx_eq(make_float2(2, 2)->normalize(), make_float2(1 / std::sqrt(2.0f))));
    }

    SECTION("operator")
    {
        ADD_TEST_EXPR(approx_eq(
            make_float2(1, 2) + make_float2(3, 4), make_float2(4, 6)));

        ADD_TEST_EXPR(approx_eq(
            make_float2(1, 2) - make_float2(3, 4), make_float2(-2, -2)));

        ADD_TEST_EXPR(approx_eq(
            make_float2(1, 2) * make_float2(3, 4), make_float2(3, 8)));

        ADD_TEST_EXPR(approx_eq(
            make_float2(1, 2) / make_float2(3, 4), make_float2(1.0f / 3, 0.5f)));

        ADD_TEST_EXPR(approx_eq(
            make_float2(1, 2) + 3.0f, make_float2(4, 5)));

        ADD_TEST_EXPR(approx_eq(
            make_float2(1, 2) - 3.0f, make_float2(-2, -1)));

        ADD_TEST_EXPR(approx_eq(
            make_float2(1, 2) * 3, make_float2(3, 6)));

        ADD_TEST_EXPR(approx_eq(
            make_float2(1, 2) / 3, make_float2(1.0f / 3, 2.0f / 3)));

        ADD_TEST_EXPR(approx_eq(
            3 + make_float2(1, 2), make_float2(4, 5)));

        ADD_TEST_EXPR(approx_eq(
            3 - make_float2(1, 2), make_float2(2, 1)));

        ADD_TEST_EXPR(approx_eq(
            3 * make_float2(1, 2), make_float2(3, 6)));

        ADD_TEST_EXPR(approx_eq(
            3 / make_float2(1, 2), make_float2(3, 1.5f)));

        ADD_TEST_EXPR(
            approx_eq_f(dot(make_float2(1, 2), make_float2(3, 4)), 11));

        ADD_TEST_EXPR(
            approx_eq_f(cos(make_float2(0, 2), make_float2(0, -3)), -1));
    }
}

#undef ADD_TEST_EXPR
