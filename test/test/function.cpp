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

        ctx.begin_function<int(int*, int)>("add_ints_2");
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
            $return(a->x + a->y + b->x + b->y + c->x + c->y);
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

        auto test_make_vec2f = to_callable<float>(
            [&](f32 x, f32 y)
        {
            math::Vec2f v = my_make_vec2f(x, y);
            $return(v->x + v->y);
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

        auto test_make_arr4 = to_callable<int>(
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
}
