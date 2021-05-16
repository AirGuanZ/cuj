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

        auto band = to_callable<int32_t>(
            [](i32 a, i32 b)
        {
            $return(a & b);
        });

        auto bor = to_callable<int32_t>(
            [](i32 a, i32 b)
        {
            $return(a | b);
        });

        auto bxor = to_callable<int32_t>(
            [](i32 a, i32 b)
        {
            $return(a ^ b);
        });

        auto jit = ctx.gen_native_jit();

        auto bitand_func = jit.get_function(band);
        auto bitor_func = jit.get_function(bor);
        auto bitxor_func = jit.get_function(bxor);
        
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

    SECTION("const_data")
    {
        ScopedContext ctx;

        auto test = to_callable<bool>([]
        {
            const int data[] = { 1, 2, 3, 4, 5 };
            auto ptr = const_data<int>(data);

            $return(ptr[0] == 1 && ptr[1] == 2 && data[3] != 3 && data[4] == 5);
        });

        auto jit = ctx.gen_native_jit();
        auto test_func = jit.get_function(test);

        REQUIRE(test_func);
        if(test_func)
            REQUIRE(test_func() == true);
    }

    SECTION("select")
    {
        ScopedContext ctx;

        auto test = to_callable<float>(
            [](boolean cond, f32 a, f32 b)
        {
            $return(select(cond, a, b));
        });

        auto jit = ctx.gen_native_jit();
        auto test_func = jit.get_function(test);

        REQUIRE(test_func);
        if(test_func)
        {
            REQUIRE(test_func(true, 0, 1) == 0);
            REQUIRE(test_func(false, 0, 1) == 1);
        }
    }
}
