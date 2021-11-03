#include <cmath>

#include <test/test.h>

using math::Vec1f;
using math::make_vec1f;

struct Vec1Data
{
    float x;
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
            [](const Vec1f &a, const Vec1f &b)                                  \
        {                                                                       \
            $return(                                                            \
                math::abs(a.x - b.x) < 0.001f);                                 \
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

TEST_CASE("builtin.math.vec1f")
{
    SECTION("create")
    {
        ScopedContext ctx;

        auto test_make_vec1f_0 = to_callable<Vec1f>(
            []()
        {
            $return(make_vec1f());
        });

        auto test_make_vec1f_1 = to_callable<Vec1f>(
            [](f32 v)
        {
            $return(make_vec1f(v));
        });

        auto jit = ctx.gen_native_jit();

        Vec1Data vec1f_data = { 1 };
        jit.get_function(test_make_vec1f_0)(&vec1f_data);
        REQUIRE(vec1f_data.x == Approx(0));

        vec1f_data = { 1 };
        jit.get_function(test_make_vec1f_1)(&vec1f_data, 3);
        REQUIRE(vec1f_data.x == Approx(3));
    }

    SECTION("elem")
    {
        ADD_TEST_EXPR(
            approx_eq_f(make_vec1f(2)[0], 2));
    }

    SECTION("write elem")
    {
        ScopedContext ctx;

        auto test = to_callable<bool>([]
        {
            Vec1f v = make_vec1f(2);
            v[0] = 1;
            $return(v[0] == 1);
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
            approx_eq_f(make_vec1f(2).length_square(), 4));

        ADD_TEST_EXPR(
            approx_eq_f(make_vec1f(2).length(), 2));

        ADD_TEST_EXPR(
            approx_eq_f(make_vec1f(2).min_elem(), 2));

        ADD_TEST_EXPR(
            approx_eq_f(make_vec1f(2).max_elem(), 2));

        ADD_TEST_EXPR(
            approx_eq(make_vec1f(-2).normalize(), make_vec1f(-1)));
    }

    SECTION("operator")
    {
        ADD_TEST_EXPR(approx_eq(
            -make_vec1f(2), make_vec1f(-2)));

        ADD_TEST_EXPR(approx_eq(
            make_vec1f(1) + make_vec1f(3), make_vec1f(4)));

        ADD_TEST_EXPR(approx_eq(
            make_vec1f(1) - make_vec1f(3), make_vec1f(-2)));

        ADD_TEST_EXPR(approx_eq(
            make_vec1f(1) * make_vec1f(3), make_vec1f(3)));

        ADD_TEST_EXPR(approx_eq(
            make_vec1f(1) / make_vec1f(3), make_vec1f(1.0f / 3)));

        ADD_TEST_EXPR(approx_eq(
            make_vec1f(1) + 3.0f, make_vec1f(4)));

        ADD_TEST_EXPR(approx_eq(
            make_vec1f(1) - 3.0f, make_vec1f(-2)));

        ADD_TEST_EXPR(approx_eq(
            make_vec1f(1) * 3, make_vec1f(3)));

        ADD_TEST_EXPR(approx_eq(
            make_vec1f(1) / 3, make_vec1f(1.0f / 3)));

        ADD_TEST_EXPR(approx_eq(
            3 + make_vec1f(1), make_vec1f(4)));

        ADD_TEST_EXPR(approx_eq(
            3 - make_vec1f(1), make_vec1f(2)));

        ADD_TEST_EXPR(approx_eq(
            3 * make_vec1f(1), make_vec1f(3)));

        ADD_TEST_EXPR(approx_eq(
            3 / make_vec1f(1), make_vec1f(3)));

        ADD_TEST_EXPR(
            approx_eq_f(dot(make_vec1f(1), make_vec1f(3)), 3));

        ADD_TEST_EXPR(
            approx_eq_f(cos(make_vec1f(1), make_vec1f(-2)), -1));
    }
}

#undef ADD_TEST_EXPR
