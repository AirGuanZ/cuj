#include <cmath>

#include <test/test.h>

using math::Vec2f;
using math::make_vec2f;

struct Vec2Data
{
    float x, y;
};

#define ADD_TEST_EXPR(EXPR)                                                     \
    do {                                                                        \
        ScopedContext ctx;                                                      \
        auto approx_eq_f = to_callable<bool>(                                   \
            [](const f32 &a, const f32 &b)                                      \
        {                                                                       \
            $return(math::abs(a - b) < 0.001f);                                 \
        });                                                                     \
        auto approx_eq = to_callable<bool>(                                     \
            [](const Vec2f &a, const Vec2f &b)                                  \
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
        auto func = jit.get_function(test);                                     \
        REQUIRE(func);                                                          \
        if(func)                                                                \
        {                                                                       \
            REQUIRE(func() == true);                                            \
        }                                                                       \
    } while(false)

TEST_CASE("builtin.math.vec2f")
{
    SECTION("create")
    {
        ScopedContext ctx;

        auto test_make_vec2f_0 = to_callable<Vec2f>(
            []()
        {
            $return(make_vec2f());
        });

        auto test_make_vec2f_1 = to_callable<Vec2f>(
            [](f32 v)
        {
            $return(make_vec2f(v));
        });

        auto test_make_vec2f_2 = to_callable<Vec2f>(
            [](f32 x, f32 y)
        {
            $return(make_vec2f(x, y));
        });

        auto jit = ctx.gen_native_jit();

        Vec2Data vec2f_data = { 1, 2 };
        jit.get_function(test_make_vec2f_0)(&vec2f_data);
        REQUIRE(vec2f_data.x == Approx(0));
        REQUIRE(vec2f_data.y == Approx(0));

        vec2f_data = { 1, 2 };
        jit.get_function(test_make_vec2f_1)(&vec2f_data, 3);
        REQUIRE(vec2f_data.x == Approx(3));
        REQUIRE(vec2f_data.y == Approx(3));

        vec2f_data = { 1, 2 };
        jit.get_function(test_make_vec2f_2)(&vec2f_data, 3, 4);
        REQUIRE(vec2f_data.x == Approx(3));
        REQUIRE(vec2f_data.y == Approx(4));
    }

    SECTION("function")
    {
        ADD_TEST_EXPR(
            approx_eq_f(make_vec2f(1, 2)->length_square(), 5));

        ADD_TEST_EXPR(
            approx_eq_f(make_vec2f(1, 2)->length(), std::sqrt(5.0f)));

        ADD_TEST_EXPR(
            approx_eq_f(make_vec2f(1, 2)->min_elem(), 1));

        ADD_TEST_EXPR(
            approx_eq_f(make_vec2f(1, 2)->max_elem(), 2));

        ADD_TEST_EXPR(
            approx_eq(make_vec2f(2, 2)->normalize(), make_vec2f(1 / std::sqrt(2.0f))));
    }

    SECTION("operator")
    {
        ADD_TEST_EXPR(approx_eq(
            -make_vec2f(2, 3), make_vec2f(-2, -3)));

        ADD_TEST_EXPR(approx_eq(
            make_vec2f(1, 2) + make_vec2f(3, 4), make_vec2f(4, 6)));

        ADD_TEST_EXPR(approx_eq(
            make_vec2f(1, 2) - make_vec2f(3, 4), make_vec2f(-2, -2)));

        ADD_TEST_EXPR(approx_eq(
            make_vec2f(1, 2) * make_vec2f(3, 4), make_vec2f(3, 8)));

        ADD_TEST_EXPR(approx_eq(
            make_vec2f(1, 2) / make_vec2f(3, 4), make_vec2f(1.0f / 3, 0.5f)));

        ADD_TEST_EXPR(approx_eq(
            make_vec2f(1, 2) + 3.0f, make_vec2f(4, 5)));

        ADD_TEST_EXPR(approx_eq(
            make_vec2f(1, 2) - 3.0f, make_vec2f(-2, -1)));

        ADD_TEST_EXPR(approx_eq(
            make_vec2f(1, 2) * 3, make_vec2f(3, 6)));

        ADD_TEST_EXPR(approx_eq(
            make_vec2f(1, 2) / 3, make_vec2f(1.0f / 3, 2.0f / 3)));

        ADD_TEST_EXPR(approx_eq(
            3 + make_vec2f(1, 2), make_vec2f(4, 5)));

        ADD_TEST_EXPR(approx_eq(
            3 - make_vec2f(1, 2), make_vec2f(2, 1)));

        ADD_TEST_EXPR(approx_eq(
            3 * make_vec2f(1, 2), make_vec2f(3, 6)));

        ADD_TEST_EXPR(approx_eq(
            3 / make_vec2f(1, 2), make_vec2f(3, 1.5f)));

        ADD_TEST_EXPR(
            approx_eq_f(dot(make_vec2f(1, 2), make_vec2f(3, 4)), 11));

        ADD_TEST_EXPR(
            approx_eq_f(cos(make_vec2f(0, 2), make_vec2f(0, -3)), -1));
    }
}

#undef ADD_TEST_EXPR
