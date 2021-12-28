#include "test/test.h"

TEST_CASE("misc")
{
    SECTION("return ref")
    {
        with_mcjit<ref<i32>>(
            [](ref<i32> a, ref<i32> b)
        {
            return cstd::select(a < b, a, b);
        },
            [](auto f)
        {
            int32_t a = 1, b = 2;
            REQUIRE(f(&a, &b) == &a);
            a = 2, b = 1;
            REQUIRE(f(&a, &b) == &b);
        });
    }
}
