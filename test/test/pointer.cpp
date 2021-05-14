#include <test/test.h>

TEST_CASE("pointer")
{
    SECTION("comparison")
    {
        ScopedContext ctx;

        auto test = to_callable<bool>(
            []
        {
            i32 a = 0, b = 0;
            Pointer<int> pa = a.address(), pb = b.address(), pa2 = a.address();
            Pointer<int> nil = nullptr;
            
            boolean ret = true;
            ret = ret && (pa != pb);
            ret = ret && (pa == pa2);
            ret = ret && ((pa < pb) ^ (pa > pb));
            ret = ret && (pa <= pa2);
            ret = ret && !(pa < pa2);
            ret = ret && (pa > nullptr);
            ret = ret && (pa > nil);
            ret = ret && (pa >= nil);
            ret = ret && !!pa;
            ret = ret && !nil;

            $return(ret);
        });
        
        auto jit = ctx.gen_native_jit();
        auto test_func = jit.get_function(test);

        REQUIRE(test_func);
        if(test_func)
            REQUIRE(test_func() == true);
    }

    SECTION("+ and -")
    {
        ScopedContext ctx;

        auto test = to_callable<bool>(
            []
        {
            Array<int, 4> arr;
            boolean ret = true;

            ret = ret && (1 + arr[0].address() == arr[1].address());
            ret = ret && (arr[3].address() - 3 == arr[0].address());

            i64 diff = arr[2].address() - arr[0].address();
            ret = ret && (arr[2].address() - diff == arr[0].address());

            $return(ret);
        });

        auto jit = ctx.gen_native_jit();
        auto test_func = jit.get_function(test);

        REQUIRE(test_func);
        if(test_func)
            REQUIRE(test_func() == true);
    }

    SECTION("nested pointer")
    {
        ScopedContext ctx;

        auto test = to_callable<bool>(
            []
        {
            i32 a, b, c;

            Value<int *[4]> ptr_arr;
            
            ptr_arr[0] = a.address();
            ptr_arr[1] = a.address();
            ptr_arr[2] = b.address();
            ptr_arr[3] = c.address();

            ptr_arr[0].deref() = 1;
            ptr_arr[1].deref() = 2;
            ptr_arr[2].deref() = 3;
            ptr_arr[3].deref() = 4;

            boolean ret = true;
            ret = ret && (a == 2);
            ret = ret && (b == 3);
            ret = ret && (ptr_arr[0] && c == 4);

            Pointer<Pointer<int>> pp0 = ptr_arr[0].address();
            (pp0 + 2).deref().deref() = 0;

            ret = ret && (b == 0);

            $return(ret);
        });

        auto jit = ctx.gen_native_jit();
        auto test_func = jit.get_function(test);

        REQUIRE(test_func);
        if(test_func)
            REQUIRE(test_func() == true);
    }

    SECTION("cast")
    {
        ScopedContext ctx;

        auto test = to_callable<bool>([]
        {
            Value<float> x = 5, y = 7;
            Pointer<uint8_t> px = ptr_cast<uint8_t>(x.address());
            Pointer<uint8_t> py = ptr_cast<uint8_t>(y.address());
            for(int i = 0; i < 4; ++i)
                py[i] = px[i];
            $return(x == y);
        });

        auto jit = ctx.gen_native_jit();
        auto test_func = jit.get_function(test);

        REQUIRE(test_func);
        if(test_func)
            REQUIRE(test_func() == true);
    }

    SECTION("class pointer")
    {
        ScopedContext ctx;

        auto test = to_callable<void>(
            [](const Pointer<void>        &a,
               const Pointer<math::Vec3f> &b)
        {
            ptr_cast<math::Vec3f>(a).deref()->x = b.deref()->x;
        });

        auto jit = ctx.gen_native_jit();
        auto test_func = jit.get_function(test);

        REQUIRE(test_func);
        if(test_func)
        {
            struct Data { float x, y, z; };

            Data a = { 1, 2, 3 }, b = { 4, 5, 6 };
            test_func(&a, &b);

            REQUIRE(a.x == 4);
            REQUIRE(a.y == 2);
            REQUIRE(a.z == 3);

            REQUIRE(b.x == 4);
            REQUIRE(b.y == 5);
            REQUIRE(b.z == 6);
        }
    }
}
