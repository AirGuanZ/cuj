#include "test/test.h"

namespace
{

    struct S
    {
        int32_t a[5];
        
        bool operator==(const S &) const = default;
    };

    CUJ_CLASS(S, a);

} // namespace anonymous

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

    SECTION("return class object")
    {
        mcjit_require([]
        {
            cxx<S> s;
            s.a[0] = 0;
            s.a[1] = 1;
            s.a[2] = 2;
            s.a[3] = 3;
            s.a[4] = 4;
            return s;
        }, S{ { 0, 1, 2, 3, 4 } });
    }
}
