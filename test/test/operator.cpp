#include <test/test.h>

TEST_CASE("operator")
{
    SECTION("mod")
    {
        ScopedContext ctx;

        auto test = to_callable<bool>([]
        {
            boolean ret = true;

            ret = ret && (i32(7) % i32(5) == i32(2));
            ret = ret && (i32(-7) % i32(5) == i32(-2));
            ret = ret && (i32(7) % i32(-5) == i32(2));
            ret = ret && (i32(-7) % i32(-5) == i32(-2));

            $return(ret);
        });

        auto jit = ctx.gen_native_jit();
        auto test_func = jit.get_function(test);

        REQUIRE(test_func);
        if(test_func)
            REQUIRE(test_func() == true);
    }
}
