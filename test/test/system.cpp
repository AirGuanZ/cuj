#include <test/test.h>

TEST_CASE("builtin.system")
{
    SECTION("print")
    {
        ScopedContext ctx;

        auto test = to_callable<void>([]
        {
            system::print("output of 'host.system.print' test"_cuj);
        });

        auto jit = ctx.gen_native_jit();
        auto test_func = jit.get_function(test);

        REQUIRE(test_func);
        if(test_func)
            test_func();
    }

    SECTION("malloc & free")
    {
        ScopedContext ctx;

        auto test = to_callable<int>([]
        {
            Pointer<i32> px = ptr_cast<int>(system::malloc(sizeof(int)));
            px.deref() = 4;
            i32 ret = px.deref();
            system::free(px);
            $return(ret);
        });

        auto jit = ctx.gen_native_jit();
        auto test_func = jit.get_function(test);

        REQUIRE(test_func);
        if(test_func)
            REQUIRE(test_func() == 4);
    }
}
