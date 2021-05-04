#include <cmath>

#include <test/test.h>

TEST_CASE("array")
{
    SECTION("basic")
    {
        ScopedContext ctx;

        auto check_array = to_callable<bool>(
            [](const Array<int, 5> &a)
        {
            for(size_t i = 0; i < a.size(); ++i)
            {
                $if(a[i] != i)
                {
                    $return(false);
                };
            }
            $return(true);
        });

        auto make_array = to_callable<Array<int, 5>>(
            []
        {
            Array<int, 5> arr;
            arr[0] = 0;
            arr[1] = 1;
            arr[2] = 2;
            arr[3] = 3;
            arr[4] = 4;
            $return(arr);
        });

        auto test_array1 = to_callable<bool>(
            [&]
        {
            Array<int, 5> arr1 = make_array();
            Array<int, 5> arr2 = arr1;
            $return(check_array(arr1) && check_array(arr2));
        });

        auto test_array2 = to_callable<bool>(
            [&]
        {
            Array<int, 5> arr1 = make_array();
            Array<int, 5> arr2 = arr1;
            arr2[0] = 1;
            $return(check_array(arr1) && !check_array(arr2));
        });

        auto jit = ctx.gen_native_jit();
        auto test_array1_func = jit.get_symbol(test_array1);
        auto test_array2_func = jit.get_symbol(test_array2);

        REQUIRE(test_array1_func);
        if(test_array1_func)
            REQUIRE(test_array1_func() == true);

        REQUIRE(test_array2_func);
        if(test_array2_func)
            REQUIRE(test_array2_func() == true);
    }

    SECTION("nested array")
    {
        ScopedContext ctx;

        auto make_arr = to_callable<Array<Array<int, 3>, 4>>([]
        {
            Array<Array<int, 3>, 4> arr;
            int v = 0;
            for(int i = 0; i < 4; ++i)
            {
                for(int j = 0; j < 3; ++j)
                    arr[i][j] = v++;
            }
            $return(arr);
        });

        auto sum_arr = to_callable<int>([&]
        {
            auto arr = make_arr();
            i32 ret = 0;
            for(int i = 0; i < 4; ++i)
            {
                for(int j = 0; j < 3; ++j)
                    ret = ret + arr[i][j];
            }
            $return(ret);
        });

        auto jit = ctx.gen_native_jit();
        auto sum_arr_func = jit.get_symbol(sum_arr);

        REQUIRE(sum_arr_func);
        if(sum_arr_func)
        {
            int expected_result = 0;
            int v = 0;
            for(int i = 0; i < 4; ++i)
            {
                for(int j = 0; j < 3; ++j)
                    expected_result += v++;
            }
            REQUIRE(sum_arr_func() == expected_result);
        }
    }

    SECTION("array of struct")
    {
        class Struct0 : public ClassBase<Struct0>
        {
        public:
            
            using ClassBase::ClassBase;
        
            $mem(int,    x);
            $mem(int[3], y);
            $mem(float,  z);
        };

        ScopedContext ctx;

        auto make_arr = to_callable<Array<Struct0, 2>>([]
        {
            Array<Struct0, 2> arr;
            arr[0]->x    = 0;
            arr[0]->y[0] = 1;
            arr[0]->y[1] = 2;
            arr[0]->y[2] = 3;
            arr[0]->z    = 4;
            arr[1]->x    = 5;
            arr[1]->y[0] = 6;
            arr[1]->y[1] = 7;
            arr[1]->y[2] = 8;
            arr[1]->z    = 9;
            $return(arr);
        });

        auto sum_arr = to_callable<float>([&]
        {
            auto arr = make_arr();
            f32 ret = 0;
            ret = ret + arr[0]->x;
            ret = ret + arr[0]->y[0];
            ret = ret + arr[0]->y[1];
            ret = ret + arr[0]->y[2];
            ret = ret + arr[0]->z;
            ret = ret + arr[1]->x;
            ret = ret + arr[1]->y[0];
            ret = ret + arr[1]->y[1];
            ret = ret + arr[1]->y[2];
            ret = ret + arr[1]->z;
            $return(ret);
        });

        auto jit = ctx.gen_native_jit();
        auto sum_arr_func = jit.get_symbol(sum_arr);

        REQUIRE(sum_arr_func);
        if(sum_arr_func)
            REQUIRE(sum_arr_func() == Approx(45.0f));
    }
}
