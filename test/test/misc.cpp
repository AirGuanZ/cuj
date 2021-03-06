#include <array>

#include "test/test.h"

namespace
{

    struct S
    {
        int32_t a[5];
        
        bool operator==(const S &) const = default;
    };

    CUJ_CLASS(S, a);

    CUJ_CLASS_BEGIN(AlignedClass64)
        CUJ_CLASS_ALIGNMENT(64)
        CUJ_MEMBER_VARIABLE(i32, x)
    CUJ_CLASS_END
    
    CUJ_CLASS_BEGIN(AlignedClass64_2)
        CUJ_MEMBER_VARIABLE(AlignedClass64, c)
        CUJ_MEMBER_VARIABLE(i32, x)
    CUJ_CLASS_END

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

    SECTION("const data")
    {
        with_mcjit(
            [](i32 x)
        {
            var lut = const_data<int32_t>(std::span<const int>(std::array{ 3, 4, 5, 6 }));
            return lut[x];
        },
            [](auto f)
        {
            REQUIRE(f(0) == 3);
            REQUIRE(f(1) == 4);
            REQUIRE(f(2) == 5);
            REQUIRE(f(3) == 6);
        });
    }

    SECTION("import pointer")
    {
        int32_t host_var;
        with_mcjit(
            [&](i32 x)
        {
            *import_pointer(&host_var) = x;
        },
            [&host_var](auto set_host_var)
        {
            set_host_var(1);
            REQUIRE(host_var == 1);
            set_host_var(2);
            REQUIRE(host_var == 2);
            set_host_var(99);
            REQUIRE(host_var == 99);
        });
    }

    SECTION("alignment")
    {
        mcjit_require(
            []
        {
            AlignedClass64 c;
            return bitcast<u64>(c.address()) % 64;
        }, 0);

        mcjit_require(
            []
        {
            arr<AlignedClass64_2, 2> c;
            return bitcast<u64>(c[1].address()) - bitcast<u64>(c[0].address());
        }, 128);
    }
}
