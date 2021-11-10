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

        auto band = to_callable<i32>(
            [](i32 a, i32 b)
        {
            $return(a & b);
        });

        auto bor = to_callable<i32>(
            [](i32 a, i32 b)
        {
            $return(a | b);
        });

        auto bxor = to_callable<i32>(
            [](i32 a, i32 b)
        {
            $return(a ^ b);
        });

        auto bnot = to_callable<i32>(
            [](i32 a)
        {
            $return(~a);
        });

        auto jit = ctx.gen_native_jit();

        auto bitand_func = jit.get_function(band);
        auto bitor_func = jit.get_function(bor);
        auto bitxor_func = jit.get_function(bxor);
        auto bitnot_func = jit.get_function(bnot);
        
        REQUIRE(bitand_func);
        if(bitand_func)
            REQUIRE(bitand_func(34785, 45891) == (34785 & 45891));
        
        REQUIRE(bitor_func);
        if(bitor_func)
            REQUIRE(bitor_func(34785, 45891) == (34785 | 45891));
        
        REQUIRE(bitxor_func);
        if(bitxor_func)
            REQUIRE(bitxor_func(34785, 45891) == (34785 ^ 45891));

        REQUIRE(bitnot_func);
        if(bitnot_func)
            REQUIRE(bitnot_func(32753) == ~32753);
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

    SECTION("shift")
    {
        ScopedContext ctx;

        auto lsh = to_callable<u32>(
            [](u32 a, i32 b)
        {
            $return(a << b);
        });

        auto rsh = to_callable<u32>(
            [](u64 a, u32 b)
        {
            $return(a >> b);
        });

        auto jit = ctx.gen_native_jit();
        auto lsh_func = jit.get_function(lsh);
        auto rsh_func = jit.get_function(rsh);

        REQUIRE(lsh_func);
        if(lsh_func)
            REQUIRE(lsh_func(327u, 4) == 327u << 4);

        REQUIRE(rsh_func);
        if(rsh_func)
            REQUIRE(rsh_func(32765u, 5) == 32765u >> 5);
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
