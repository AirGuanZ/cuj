#include "test.h"

TEST_CASE("control flow")
{
    SECTION("switch")
    {
        with_mcjit(
            [](i32 x)
        {
            i32 ret;
            $switch(x)
            {
            $case(0) { ret = 1; };
            $case(1) { ret = 2; };
            $case(2) { ret = 3; $fallthrough; };
            $case(3) { ret = 4; };
            $default { ret = 100; };
            };
            return ret;
        },
            [](auto f)
        {
            REQUIRE(f(0) == 1);
            REQUIRE(f(1) == 2);
            REQUIRE(f(2) == 4);
            REQUIRE(f(3) == 4);
            REQUIRE(f(4) == 100);
        });
    }

    SECTION("recursion")
    {
        ScopedModule mod;
        auto fib = declare<i32(i32)>();
        fib.define([&](i32 i)
        {
            i32 ret;
            $if(i <= 1)
            {
                $return(i);
            }
            $else
            {
                $return(ret = fib(i - 1) + fib(i - 2));
            };
        });

        MCJIT mcjit;
        mcjit.generate(mod);

        auto fib_c = mcjit.get_function(fib);
        REQUIRE(fib_c);
        if(fib_c)
        {
            REQUIRE(fib_c(0) == 0);
            REQUIRE(fib_c(1) == 1);
            REQUIRE(fib_c(2) == 1);
            REQUIRE(fib_c(3) == 2);
            REQUIRE(fib_c(4) == 3);
        }
    }

    SECTION("loop")
    {
        mcjit_require(
            [](i32 n)
        {
            i32 ret = 0, i = 0;
            $loop
            {
                $if(i >= n)
                {
                    $break;
                };
                ret = ret + i;
                i = i + 1;
            };
            return ret;
        },
            5, 1 + 2 + 3 + 4);

        mcjit_require(
            [](i32 n)
        {
            i32 ret = 0;
            $forrange(i, 0, n)
            {
                ret = ret + i;
            };
            return ret;
        },
            5, 0 + 1 + 2 + 3 + 4);
    }

    SECTION("while")
    {
        mcjit_require(
            [](i32 n)
        {
            auto cond = function([](i32 x, ref<i32> y)
            {
                y = y + 1;
                return x >= 0;
            });
            var y = 0;
            $while(cond(n, y))
            {
                n = n - 1;
            };
            return y;
        }, 5, 7);

        mcjit_require(
            [](i32 n)
        {
            var y = 0;
            auto cond = [&]
            {
                y = y + 1;
                return n >= 0;
            };
            $while(cond())
            {
                n = n - 1;
            };
            return y;
        }, 5, 7);
    }

    SECTION("if")
    {
        with_mcjit(
            [](i32 i)
        {
            var ret = 0;
            auto cond1 = [&]
            {
                ret = ret + 1;
                return i == 2;
            };
            $if(i == 1)
            {
                ret = ret + 100;
            }
            $elif(cond1())
            {
                ret = ret + 200;
            }
            $else
            {
                ret = ret + 300;
            };
            return ret;
        },
            [](auto f)
        {
            REQUIRE(f(1) == 100);
            REQUIRE(f(2) == 201);
            REQUIRE(f(3) == 301);
        });
    }
}
