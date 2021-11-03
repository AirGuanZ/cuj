#include <cmath>

#include <test/test.h>

using math::Vec4f;
using math::make_vec4f;

struct Vec4fData
{
    float x, y, z, w;
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
            [](const Vec4f &a, const Vec4f &b)                                  \
        {                                                                       \
            $return(                                                            \
                math::abs(a.x - b.x) < 0.001f &&                                \
                math::abs(a.y - b.y) < 0.001f &&                                \
                math::abs(a.z - b.z) < 0.001f &&                                \
                math::abs(a.w - b.w) < 0.001f);                                 \
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

TEST_CASE("builtin.math.vec4f")
{
    SECTION("create")
    {
        ScopedContext ctx;

        auto test_make_vec4f_0 = to_callable<Vec4f>(
            []()
        {
            $return(make_vec4f());
        });

        auto test_make_vec4f_1 = to_callable<Vec4f>(
            [](f32 v)
        {
            $return(make_vec4f(v));
        });

        auto test_make_vec4f_2 = to_callable<Vec4f>(
            [](f32 x, f32 y, f32 z, f32 w)
        {
            $return(make_vec4f(x, y, z, w));
        });

        auto jit = ctx.gen_native_jit();

        Vec4fData vec4f_data = { 1, 2, 3, 4 };
        jit.get_function(test_make_vec4f_0)(&vec4f_data);
        REQUIRE(vec4f_data.x == Approx(0));
        REQUIRE(vec4f_data.y == Approx(0));
        REQUIRE(vec4f_data.z == Approx(0));
        REQUIRE(vec4f_data.w == Approx(0));

        vec4f_data = { 1, 2, 3, 4 };
        jit.get_function(test_make_vec4f_1)(&vec4f_data, 3);
        REQUIRE(vec4f_data.x == Approx(3));
        REQUIRE(vec4f_data.y == Approx(3));
        REQUIRE(vec4f_data.z == Approx(3));
        REQUIRE(vec4f_data.w == Approx(3));

        vec4f_data = { 1, 2, 3, 4 };
        jit.get_function(test_make_vec4f_2)(&vec4f_data, 3, 4, 5, 6);
        REQUIRE(vec4f_data.x == Approx(3));
        REQUIRE(vec4f_data.y == Approx(4));
        REQUIRE(vec4f_data.z == Approx(5));
        REQUIRE(vec4f_data.w == Approx(6));
    }

    SECTION("elem")
    {
        ADD_TEST_EXPR(
            approx_eq_f(make_vec4f(2, 3, 4, 5)[3], 5));
    }

    SECTION("write elem")
    {
        ScopedContext ctx;

        auto test = to_callable<bool>([]
        {
            Vec4f v = make_vec4f(1, 2, 3, 4);
            v[0] = 3;
            v[2] = 1;
            $return(v[0] == 3 && v[1] == 2 && v[2] == 1 && v[3] == 4);
        });

        auto jit = ctx.gen_native_jit();
        auto test_func = jit.get_function(test);

        REQUIRE(test_func);
        if(test_func)
            REQUIRE(test_func() == true);
    }

    SECTION("function")
    {
        ADD_TEST_EXPR(
            approx_eq_f(make_vec4f(1, 2, 3, 4).length_square(), 30));

        ADD_TEST_EXPR(
            approx_eq_f(make_vec4f(1, 2, 3, 4).length(), std::sqrt(30.0f)));

        ADD_TEST_EXPR(
            approx_eq_f(make_vec4f(1, 2, 3, 4).min_elem(), 1));

        ADD_TEST_EXPR(
            approx_eq_f(make_vec4f(1, 2, 3, 4).max_elem(), 4));

        ADD_TEST_EXPR(
            approx_eq(make_vec4f(2, 2, 2, 2).normalize(), make_vec4f(1 / std::sqrt(4.0f))));
    }

    SECTION("operator")
    {
        ADD_TEST_EXPR(approx_eq(
            -make_vec4f(2, 3, 4, 5), make_vec4f(-2, -3, -4, -5)));

        ADD_TEST_EXPR(approx_eq(
            make_vec4f(1, 2, 3, 4) + make_vec4f(3, 4, 5, 6), make_vec4f(4, 6, 8, 10)));

        ADD_TEST_EXPR(approx_eq(
            make_vec4f(1, 2, 3, 4) - make_vec4f(3, 4, 5, 6), make_vec4f(-2)));

        ADD_TEST_EXPR(approx_eq(
            make_vec4f(1, 2, 3, 4) * make_vec4f(3, 4, 5, 6), make_vec4f(3, 8, 15, 24)));

        ADD_TEST_EXPR(approx_eq(
            make_vec4f(1, 2, 3, 4) / make_vec4f(3, 4, 5, 6), make_vec4f(1.0f / 3, 0.5f, 3.0f / 5, 4.0f / 6)));

        ADD_TEST_EXPR(approx_eq(
            make_vec4f(1, 2, 3, 4) + 3.0f, make_vec4f(4, 5, 6, 7)));

        ADD_TEST_EXPR(approx_eq(
            make_vec4f(1, 2, 3, 4) - 3.0f, make_vec4f(-2, -1, 0, 1)));

        ADD_TEST_EXPR(approx_eq(
            make_vec4f(1, 2, 3, 4) * 3, make_vec4f(3, 6, 9, 12)));

        ADD_TEST_EXPR(approx_eq(
            make_vec4f(1, 2, 3, 4) / 3, make_vec4f(1.0f / 3, 2.0f / 3, 1, 4.0f / 3)));

        ADD_TEST_EXPR(approx_eq(
            3 + make_vec4f(1, 2, 3, 4), make_vec4f(4, 5, 6, 7)));

        ADD_TEST_EXPR(approx_eq(
            3 - make_vec4f(1, 2, 3, 4), make_vec4f(2, 1, 0, -1)));

        ADD_TEST_EXPR(approx_eq(
            3 * make_vec4f(1, 2, 3, 4), make_vec4f(3, 6, 9, 12)));

        ADD_TEST_EXPR(approx_eq(
            3 / make_vec4f(1, 2, 3, 4), make_vec4f(3, 1.5f, 1, 0.75f)));

        ADD_TEST_EXPR(
            approx_eq_f(dot(make_vec4f(1, 2, 3, 4), make_vec4f(3, 4, 6, 8)), 61));

        ADD_TEST_EXPR(
            approx_eq_f(cos(make_vec4f(0, 0, 0, 2), make_vec4f(0, 0, -3, 0)), 0));
    }
}

#undef ADD_TEST_EXPR
