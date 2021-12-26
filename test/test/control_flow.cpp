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
}
