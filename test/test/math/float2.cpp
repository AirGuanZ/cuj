#include <cmath>

#include <test/test.h>

using math::Float2;
using math::make_float2;

struct Float2Data
{
    float x, y;
};

TEST_CASE("builtin.math.float2")
{
    SECTION("make_float2")
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
}
