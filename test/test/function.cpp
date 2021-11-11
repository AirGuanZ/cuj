#include <iostream>
#include <numeric>

#include <test/test.h>

TEST_CASE("function")
{
    SECTION("define function")
    {
        ScopedContext ctx;

        auto define_add_ints = [](Value<int *> int_ptr, Value<int> num)
        {
            i32 i = 0;
            i32 result = 0;
            $while(i < num)
            {
                result = result + int_ptr[i];
                i = i + 1;
            };
            $return(result);
        };

        to_callable<int>("add_ints_1", define_add_ints);

        ctx.begin_function<i32(Pointer<i32>, i32)>("add_ints_2");
        {
            $arg(int *, int_ptr);
            $arg(int,   num);
            define_add_ints(int_ptr, num);
        }
        ctx.end_function();

        ctx.add_function<int>("add_ints_3", define_add_ints);

        auto jit = ctx.gen_native_jit();
        auto add_ints_1 = jit.get_function_by_name<int(const int *, int)>("add_ints_1");
        auto add_ints_2 = jit.get_function_by_name<int(const int *, int)>("add_ints_2");
        auto add_ints_3 = jit.get_function_by_name<int(const int *, int)>("add_ints_3");

        REQUIRE(add_ints_1);
        REQUIRE(add_ints_2);
        REQUIRE(add_ints_3);

        const std::vector<int> ints = { 1, 3, 2, 4, 6, 7 };
        const int num = static_cast<int>(ints.size());
        const int expected = std::accumulate(ints.begin(), ints.end(), 0);

        if(add_ints_1)
        {
            const int sum = add_ints_1(ints.data(), num);
            REQUIRE(expected == sum);
        }

        if(add_ints_2)
        {
            const int sum = add_ints_2(ints.data(), num);
            REQUIRE(expected == sum);
        }

        if(add_ints_3)
        {
            const int sum = add_ints_3(ints.data(), num);
            REQUIRE(expected == sum);
        }
    }

    SECTION("class-type argument")
    {
        ScopedContext ctx;
        
        auto float2_sum = to_callable<float>(
            [](math::Vec2f a, math::Vec2f b, math::Vec2f c)
        {
            $return(a.x + a.y + b.x + b.y + c.x + c.y);
        });

        auto test_float2_sum_func = to_callable<float>(
            "test_float2_sum", [&]
        {
            math::Vec2f a(1, 2);
            math::Vec2f b(3, 4);
            math::Vec2f c(5, 6);
            $return(float2_sum(a, b, c));
        });

        auto jit = ctx.gen_native_jit();
        auto test_float2_sum = jit.get_function(test_float2_sum_func);

        REQUIRE(test_float2_sum);
        if(test_float2_sum)
        {
            REQUIRE(test_float2_sum() == Approx(21));
        }
    }

    SECTION("class-type return value")
    {
        ScopedContext ctx;

        auto my_make_vec2f = to_callable<math::Vec2f>(
            [](f32 x, f32 y)
        {
            $return(math::make_vec2f(x, y));
        });

        auto test_make_vec2f = to_callable<f32>(
            [&](f32 x, f32 y)
        {
            math::Vec2f v = my_make_vec2f(x, y);
            $return(v.x + v.y);
        });
        
        auto jit = ctx.gen_native_jit();
        auto test_make_vec2f_func = jit.get_function(test_make_vec2f);

        REQUIRE(test_make_vec2f_func);
        if(test_make_vec2f_func)
            REQUIRE(test_make_vec2f_func(1, 2) == Approx(3.0f));
    }

    SECTION("array-type return value")
    {
        ScopedContext ctx;

        auto my_make_arr4 = to_callable<int[4]>(
            [](i32 x, i32 y, i32 z, i32 w)
        {
            Array<int, 4> arr;
            arr[0] = x;
            arr[1] = y;
            arr[2] = z;
            arr[3] = w;
            $return(arr);
        });

        auto test_make_arr4 = to_callable<i32>(
            [&](i32 x, i32 y, i32 z, i32 w)
        {
            Array<int, 4> arr = my_make_arr4(x, y, z, w);
            $return(arr[0] + arr[1] + arr[2] + arr[3]);
        });

        auto jit = ctx.gen_native_jit();
        auto test_make_arr4_func = jit.get_function(test_make_arr4);

        REQUIRE(test_make_arr4_func);
        if(test_make_arr4_func)
            REQUIRE(test_make_arr4_func(1, 2, 3, 4) == 10);
    }

    SECTION("import host func")
    {
        ScopedContext ctx;

        void (*ptr1)();
        ptr1 = [] { std::cout << "output of 'import host printf' test\n"; };
        auto func1 = import_function(ptr1);

        int (*ptr2)(const int*);
        ptr2 = [](const int *x) { return *x * *x * *x; };
        auto func2 = import_function(ptr2);

        int func3_add_val = 5;
        auto func3 = import_function(
            [=](const int *pi) { return func3_add_val + *pi; });

        auto jit = ctx.gen_native_jit();
        
        auto test_func1 = jit.get_function(func1);
        REQUIRE(test_func1);
        if(test_func1)
            test_func1();

        auto test_func2 = jit.get_function(func2);
        static_assert(std::is_same_v<decltype(test_func2), decltype(ptr2)>);
        REQUIRE(test_func2);
        if(test_func2)
        {
            int x = 5;
            REQUIRE(test_func2(&x) == 5 * 5 * 5);
        }

        auto test_func3 = jit.get_function(func3);
        static_assert(std::is_same_v<decltype(test_func3), int(*)(const int *)>);
        REQUIRE(test_func3);
        if(test_func3)
        {
            int x = 3;
            REQUIRE(test_func3(&x) == func3_add_val + x);
        }
    }

    SECTION("while cond")
    {
        ScopedContext ctx;

        auto func = to_callable<i32>([](i32 n)
        {
            i32 c = 0, i = 0;

            auto cond = [&]
            {
                c = c + 1;
                boolean ret;
                $if(i < n)
                {
                    i = i + 1;
                    ret = true;
                }
                $else
                {
                    ret = false;
                };
                return ret;
            };

            $while(cond())
            {

            };

            $return(c);
        });
        
        auto jit = ctx.gen_native_jit();
        auto test_func = jit.get_function(func);

        REQUIRE(test_func);
        if(test_func)
            REQUIRE(test_func(10) == 11);
    }

    SECTION("switch")
    {
        ScopedContext ctx;

        auto entry = to_callable<int32_t>([](i32 i)
        {
            $switch(i)
            {
                for(int j = 1; j <= 4; ++j)
                    $case(j) { $return(j + 1); };
                $default{ $return(0); };
            };
        });

        auto jit = ctx.gen_native_jit();
        auto entry_func = jit.get_function(entry);

        REQUIRE(entry_func);
        if(entry_func)
        {
            REQUIRE(entry_func(1) == 2);
            REQUIRE(entry_func(2) == 3);
            REQUIRE(entry_func(3) == 4);
            REQUIRE(entry_func(4) == 5);
            REQUIRE(entry_func(5) == 0);
        }
    }

    SECTION("fallthrough")
    {
        ScopedContext ctx;

        auto entry = to_callable<void>(
            [](i32 i, Pointer<i32> a, Pointer<i32> b)
        {
            $switch(i)
            {
                $case(0) { *a = 1; $fallthrough; };
                $case(1) { *b = 1; };
                $case(2) { *a = 1; *b = 1; };
            };
        });

        auto jit = ctx.gen_native_jit();
        auto entry_func = jit.get_function(entry);

        REQUIRE(entry_func);
        if(entry_func)
        {
            int32_t a = 0, b = 0;
            entry_func(0, &a, &b);
            REQUIRE(a == 1);
            REQUIRE(b == 1);

            a = 0; b = 0;
            entry_func(1, &a, &b);
            REQUIRE(a == 0);
            REQUIRE(b == 1);

            a = 0; b = 0;
            entry_func(2, &a, &b);
            REQUIRE(a == 1);
            REQUIRE(b == 1);
        }
    }

    SECTION("recursive function")
    {
        ScopedContext ctx;

        auto fib = declare<i32(i32)>();

        fib.define([&](i32 i)
        {
            $if(i <= 0)
            {
                $return(0);
            }
            $elif(i == 1)
            {
                $return(1);
            }
            $else
            {
                $return(fib(i - 1) + fib(i - 2));
            };
        });

        auto jit = ctx.gen_native_jit();
        auto fib_func = jit.get_function(fib);

        REQUIRE(fib_func);
        if(fib_func)
        {
            REQUIRE(fib_func(0) == 0);
            REQUIRE(fib_func(1) == 1);
            REQUIRE(fib_func(2) == 1);
            REQUIRE(fib_func(3) == 2);
            REQUIRE(fib_func(4) == 3);
            REQUIRE(fib_func(5) == 5);
        }
    }
}
