#include <cmath>

#include <test/test.h>

using math::Float4;
using math::make_float4;

struct Float4Data
{
    float x, y, z, w;
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
            [](const Float4 &a, const Float4 &b)                                \
        {                                                                       \
            $return(                                                            \
                math::abs(a->x - b->x) < 0.001f &&                              \
                math::abs(a->y - b->y) < 0.001f &&                              \
                math::abs(a->z - b->z) < 0.001f &&                              \
                math::abs(a->w - b->w) < 0.001f);                               \
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

TEST_CASE("builtin.math.float4")
{
    SECTION("create")
    {
        ScopedContext ctx;

        auto test_make_float4_0 = to_callable<Float4>(
            []()
        {
            $return(make_float4());
        });

        auto test_make_float4_1 = to_callable<Float4>(
            []($f32 v)
        {
            $return(make_float4(v));
        });

        auto test_make_float4_2 = to_callable<Float4>(
            []($f32 x, $f32 y, $f32 z, $f32 w)
        {
            $return(make_float4(x, y, z, w));
        });

        auto jit = ctx.gen_native_jit();

        Float4Data float4_data = { 1, 2, 3, 4 };
        jit.get_symbol(test_make_float4_0)(&float4_data);
        REQUIRE(float4_data.x == Approx(0));
        REQUIRE(float4_data.y == Approx(0));
        REQUIRE(float4_data.z == Approx(0));
        REQUIRE(float4_data.w == Approx(0));

        float4_data = { 1, 2, 3, 4 };
        jit.get_symbol(test_make_float4_1)(&float4_data, 3);
        REQUIRE(float4_data.x == Approx(3));
        REQUIRE(float4_data.y == Approx(3));
        REQUIRE(float4_data.z == Approx(3));
        REQUIRE(float4_data.w == Approx(3));

        float4_data = { 1, 2, 3, 4 };
        jit.get_symbol(test_make_float4_2)(&float4_data, 3, 4, 5, 6);
        REQUIRE(float4_data.x == Approx(3));
        REQUIRE(float4_data.y == Approx(4));
        REQUIRE(float4_data.z == Approx(5));
        REQUIRE(float4_data.w == Approx(6));
    }

    SECTION("function")
    {
        ADD_TEST_EXPR(
            approx_eq_f(make_float4(1, 2, 3, 4)->length_square(), 30));

        ADD_TEST_EXPR(
            approx_eq_f(make_float4(1, 2, 3, 4)->length(), std::sqrt(30.0f)));

        ADD_TEST_EXPR(
            approx_eq_f(make_float4(1, 2, 3, 4)->min_elem(), 1));

        ADD_TEST_EXPR(
            approx_eq_f(make_float4(1, 2, 3, 4)->max_elem(), 4));

        ADD_TEST_EXPR(
            approx_eq(make_float4(2, 2, 2, 2)->normalize(), make_float4(1 / std::sqrt(4.0f))));
    }

    SECTION("operator")
    {
        ADD_TEST_EXPR(approx_eq(
            make_float4(1, 2, 3, 4) + make_float4(3, 4, 5, 6), make_float4(4, 6, 8, 10)));

        ADD_TEST_EXPR(approx_eq(
            make_float4(1, 2, 3, 4) - make_float4(3, 4, 5, 6), make_float4(-2)));

        ADD_TEST_EXPR(approx_eq(
            make_float4(1, 2, 3, 4) * make_float4(3, 4, 5, 6), make_float4(3, 8, 15, 24)));

        ADD_TEST_EXPR(approx_eq(
            make_float4(1, 2, 3, 4) / make_float4(3, 4, 5, 6), make_float4(1.0f / 3, 0.5f, 3.0f / 5, 4.0f / 6)));

        ADD_TEST_EXPR(approx_eq(
            make_float4(1, 2, 3, 4) + 3.0f, make_float4(4, 5, 6, 7)));

        ADD_TEST_EXPR(approx_eq(
            make_float4(1, 2, 3, 4) - 3.0f, make_float4(-2, -1, 0, 1)));

        ADD_TEST_EXPR(approx_eq(
            make_float4(1, 2, 3, 4) * 3, make_float4(3, 6, 9, 12)));

        ADD_TEST_EXPR(approx_eq(
            make_float4(1, 2, 3, 4) / 3, make_float4(1.0f / 3, 2.0f / 3, 1, 4.0f / 3)));

        ADD_TEST_EXPR(approx_eq(
            3 + make_float4(1, 2, 3, 4), make_float4(4, 5, 6, 7)));

        ADD_TEST_EXPR(approx_eq(
            3 - make_float4(1, 2, 3, 4), make_float4(2, 1, 0, -1)));

        ADD_TEST_EXPR(approx_eq(
            3 * make_float4(1, 2, 3, 4), make_float4(3, 6, 9, 12)));

        ADD_TEST_EXPR(approx_eq(
            3 / make_float4(1, 2, 3, 4), make_float4(3, 1.5f, 1, 0.75f)));

        ADD_TEST_EXPR(
            approx_eq_f(dot(make_float4(1, 2, 3, 4), make_float4(3, 4, 6, 8)), 61));

        ADD_TEST_EXPR(
            approx_eq_f(cos(make_float4(0, 0, 0, 2), make_float4(0, 0, -3, 0)), 0));
    }
}

#undef ADD_TEST_EXPR
