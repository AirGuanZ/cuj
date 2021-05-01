#include <cmath>

#include <test/test.h>

using math::Float3;
using math::make_float3;

struct Float3Data
{
    float x, y, z;
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
            [](const Float3 &a, const Float3 &b)                                \
        {                                                                       \
            $return(                                                            \
                math::abs(a->x - b->x) < 0.001f &&                              \
                math::abs(a->y - b->y) < 0.001f &&                              \
                math::abs(a->z - b->z) < 0.001f);                               \
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

TEST_CASE("builtin.math.float3")
{
    SECTION("create")
    {
        ScopedContext ctx;

        auto test_make_float3_0 = to_callable<Float3>(
            []()
        {
            $return(make_float3());
        });

        auto test_make_float3_1 = to_callable<Float3>(
            []($f32 v)
        {
            $return(make_float3(v));
        });

        auto test_make_float3_2 = to_callable<Float3>(
            []($f32 x, $f32 y, $f32 z)
        {
            $return(make_float3(x, y, z));
        });

        auto jit = ctx.gen_native_jit();

        Float3Data float3_data = { 1, 2, 3 };
        jit.get_symbol(test_make_float3_0)(&float3_data);
        REQUIRE(float3_data.x == Approx(0));
        REQUIRE(float3_data.y == Approx(0));
        REQUIRE(float3_data.z == Approx(0));

        float3_data = { 1, 2, 3 };
        jit.get_symbol(test_make_float3_1)(&float3_data, 3);
        REQUIRE(float3_data.x == Approx(3));
        REQUIRE(float3_data.y == Approx(3));
        REQUIRE(float3_data.z == Approx(3));

        float3_data = { 1, 2, 3 };
        jit.get_symbol(test_make_float3_2)(&float3_data, 3, 4, 5);
        REQUIRE(float3_data.x == Approx(3));
        REQUIRE(float3_data.y == Approx(4));
        REQUIRE(float3_data.z == Approx(5));
    }

    SECTION("function")
    {
        ADD_TEST_EXPR(
            approx_eq_f(make_float3(1, 2, 3)->length_square(), 14));

        ADD_TEST_EXPR(
            approx_eq_f(make_float3(1, 2, 3)->length(), std::sqrt(14.0f)));

        ADD_TEST_EXPR(
            approx_eq_f(make_float3(1, 2, 3)->min_elem(), 1));

        ADD_TEST_EXPR(
            approx_eq_f(make_float3(1, 2, 3)->max_elem(), 3));

        ADD_TEST_EXPR(
            approx_eq(make_float3(2, 2, 2)->normalize(), make_float3(1 / std::sqrt(3.0f))));
    }

    SECTION("operator")
    {
        ADD_TEST_EXPR(approx_eq(
            make_float3(1, 2, 3) + make_float3(3, 4, 5), make_float3(4, 6, 8)));

        ADD_TEST_EXPR(approx_eq(
            make_float3(1, 2, 3) - make_float3(3, 4, 5), make_float3(-2)));

        ADD_TEST_EXPR(approx_eq(
            make_float3(1, 2, 3) * make_float3(3, 4, 5), make_float3(3, 8, 15)));

        ADD_TEST_EXPR(approx_eq(
            make_float3(1, 2, 3) / make_float3(3, 4, 5), make_float3(1.0f / 3, 0.5f, 3.0f / 5)));

        ADD_TEST_EXPR(approx_eq(
            make_float3(1, 2, 3) + 3.0f, make_float3(4, 5, 6)));

        ADD_TEST_EXPR(approx_eq(
            make_float3(1, 2, 3) - 3.0f, make_float3(-2, -1, 0)));

        ADD_TEST_EXPR(approx_eq(
            make_float3(1, 2, 3) * 3, make_float3(3, 6, 9)));

        ADD_TEST_EXPR(approx_eq(
            make_float3(1, 2, 3) / 3, make_float3(1.0f / 3, 2.0f / 3, 1)));

        ADD_TEST_EXPR(approx_eq(
            3 + make_float3(1, 2, 3), make_float3(4, 5, 6)));

        ADD_TEST_EXPR(approx_eq(
            3 - make_float3(1, 2, 3), make_float3(2, 1, 0)));

        ADD_TEST_EXPR(approx_eq(
            3 * make_float3(1, 2, 3), make_float3(3, 6, 9)));

        ADD_TEST_EXPR(approx_eq(
            3 / make_float3(1, 2, 3), make_float3(3, 1.5f, 1)));

        ADD_TEST_EXPR(
            approx_eq_f(dot(make_float3(1, 2, 3), make_float3(3, 4, 6)), 29));

        ADD_TEST_EXPR(
            approx_eq_f(cos(make_float3(0, 0, 2), make_float3(0, 0, -3)), -1));
    }
}

#undef ADD_TEST_EXPR
