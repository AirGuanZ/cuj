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

    SECTION("bitwise")
    {
        ScopedContext ctx;

        auto bitand = to_callable<int32_t>(
            [](i32 a, i32 b)
        {
            $return(a & b);
        });

        auto bitor = to_callable<int32_t>(
            [](i32 a, i32 b)
        {
            $return(a | b);
        });

        auto bitxor = to_callable<int32_t>(
            [](i32 a, i32 b)
        {
            $return(a ^ b);
        });

        auto jit = ctx.gen_native_jit();

        auto bitand_func = jit.get_function(bitand);
        auto bitor_func = jit.get_function(bitor );
        auto bitxor_func = jit.get_function(bitxor);

        REQUIRE(bitand_func);
        if(bitand_func)
            REQUIRE(bitand_func(34785, 45891) == (34785 & 45891));

        REQUIRE(bitor_func);
        if(bitor_func)
            REQUIRE(bitor_func(34785, 45891) == (34785 | 45891));

        REQUIRE(bitxor_func);
        if(bitxor_func)
            REQUIRE(bitxor_func(34785, 45891) == (34785 ^ 45891));
    }
}
