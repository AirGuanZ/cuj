#include <test/test.h>

TEST_CASE("math.atomic.add")
{
    SECTION("f32")
    {
        ScopedContext ctx;

        auto test = to_callable<float>(
            []
        {
            f32 sum = 0;
            for(int i = 1; i < 5; ++i)
                atomic::atomic_add(sum.address(), i);
            $return(sum);
        });

        auto jit = ctx.gen_native_jit();
        auto test_func = jit.get_symbol(test);

        REQUIRE(test_func);
        if(test_func)
            REQUIRE(test_func() == Approx(10));
    }

    SECTION("f64")
    {
        ScopedContext ctx;

        auto test = to_callable<double>(
            []
        {
            f64 sum = 0;
            for(int i = 1; i < 5; ++i)
                atomic::atomic_add(sum.address(), i);
            $return(sum);
        });

        auto jit = ctx.gen_native_jit();
        auto test_func = jit.get_symbol(test);

        REQUIRE(test_func);
        if(test_func)
            REQUIRE(test_func() == Approx(10));
    }
}
