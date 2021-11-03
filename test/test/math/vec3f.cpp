#include <cmath>

#include <test/test.h>

using math::Vec3f;
using math::make_vec3f;

struct Vec3fData
{
    float x, y, z;
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
            [](const Vec3f &a, const Vec3f &b)                                  \
        {                                                                       \
            $return(                                                            \
                math::abs(a.x - b.x) < 0.001f &&                                \
                math::abs(a.y - b.y) < 0.001f &&                                \
                math::abs(a.z - b.z) < 0.001f);                                 \
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

TEST_CASE("builtin.math.vec3f")
{
    SECTION("create")
    {
        ScopedContext ctx;

        auto test_make_vec3f_0 = to_callable<Vec3f>(
            []()
        {
            $return(make_vec3f());
        });

        auto test_make_vec3f_1 = to_callable<Vec3f>(
            [](f32 v)
        {
            $return(make_vec3f(v));
        });

        auto test_make_vec3f_2 = to_callable<Vec3f>(
            [](f32 x, f32 y, f32 z)
        {
            $return(make_vec3f(x, y, z));
        });

        auto jit = ctx.gen_native_jit();

        Vec3fData vec3f_data = { 1, 2, 3 };
        jit.get_function(test_make_vec3f_0)(&vec3f_data);
        REQUIRE(vec3f_data.x == Approx(0));
        REQUIRE(vec3f_data.y == Approx(0));
        REQUIRE(vec3f_data.z == Approx(0));

        vec3f_data = { 1, 2, 3 };
        jit.get_function(test_make_vec3f_1)(&vec3f_data, 3);
        REQUIRE(vec3f_data.x == Approx(3));
        REQUIRE(vec3f_data.y == Approx(3));
        REQUIRE(vec3f_data.z == Approx(3));

        vec3f_data = { 1, 2, 3 };
        jit.get_function(test_make_vec3f_2)(&vec3f_data, 3, 4, 5);
        REQUIRE(vec3f_data.x == Approx(3));
        REQUIRE(vec3f_data.y == Approx(4));
        REQUIRE(vec3f_data.z == Approx(5));
    }

    SECTION("elem")
    {
        ADD_TEST_EXPR(
            approx_eq_f(make_vec3f(2, 3, 4)[2], 4));
    }

    SECTION("write elem")
    {
        ScopedContext ctx;

        auto test = to_callable<bool>([]
        {
            Vec3f v = make_vec3f(1, 2, 3);
            v[0] = 3;
            v[2] = 1;
            $return(v[0] == 3 && v[1] == 2 && v[2] == 1);
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
            approx_eq_f(make_vec3f(1, 2, 3).length_square(), 14));

        ADD_TEST_EXPR(
            approx_eq_f(make_vec3f(1, 2, 3).length(), std::sqrt(14.0f)));

        ADD_TEST_EXPR(
            approx_eq_f(make_vec3f(1, 2, 3).min_elem(), 1));

        ADD_TEST_EXPR(
            approx_eq_f(make_vec3f(1, 2, 3).max_elem(), 3));

        ADD_TEST_EXPR(
            approx_eq(make_vec3f(2, 2, 2).normalize(), make_vec3f(1 / std::sqrt(3.0f))));
    }

    SECTION("operator")
    {
        ADD_TEST_EXPR(approx_eq(
            -make_vec3f(2, 3, 4), make_vec3f(-2, -3, -4)));

        ADD_TEST_EXPR(approx_eq(
            make_vec3f(1, 2, 3) + make_vec3f(3, 4, 5), make_vec3f(4, 6, 8)));

        ADD_TEST_EXPR(approx_eq(
            make_vec3f(1, 2, 3) - make_vec3f(3, 4, 5), make_vec3f(-2)));

        ADD_TEST_EXPR(approx_eq(
            make_vec3f(1, 2, 3) * make_vec3f(3, 4, 5), make_vec3f(3, 8, 15)));

        ADD_TEST_EXPR(approx_eq(
            make_vec3f(1, 2, 3) / make_vec3f(3, 4, 5), make_vec3f(1.0f / 3, 0.5f, 3.0f / 5)));

        ADD_TEST_EXPR(approx_eq(
            make_vec3f(1, 2, 3) + 3.0f, make_vec3f(4, 5, 6)));

        ADD_TEST_EXPR(approx_eq(
            make_vec3f(1, 2, 3) - 3.0f, make_vec3f(-2, -1, 0)));

        ADD_TEST_EXPR(approx_eq(
            make_vec3f(1, 2, 3) * 3, make_vec3f(3, 6, 9)));

        ADD_TEST_EXPR(approx_eq(
            make_vec3f(1, 2, 3) / 3, make_vec3f(1.0f / 3, 2.0f / 3, 1)));

        ADD_TEST_EXPR(approx_eq(
            3 + make_vec3f(1, 2, 3), make_vec3f(4, 5, 6)));

        ADD_TEST_EXPR(approx_eq(
            3 - make_vec3f(1, 2, 3), make_vec3f(2, 1, 0)));

        ADD_TEST_EXPR(approx_eq(
            3 * make_vec3f(1, 2, 3), make_vec3f(3, 6, 9)));

        ADD_TEST_EXPR(approx_eq(
            3 / make_vec3f(1, 2, 3), make_vec3f(3, 1.5f, 1)));

        ADD_TEST_EXPR(
            approx_eq_f(dot(make_vec3f(1, 2, 3), make_vec3f(3, 4, 6)), 29));

        ADD_TEST_EXPR(
            approx_eq_f(cos(make_vec3f(0, 0, 2), make_vec3f(0, 0, -3)), -1));

        ADD_TEST_EXPR(
            approx_eq(cross(make_vec3f(1, 0, 0), make_vec3f(0, 1, 0)), make_vec3f(0, 0, 1)));
    }
}

#undef ADD_TEST_EXPR
