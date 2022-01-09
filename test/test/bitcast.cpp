#include "test.h"

TEST_CASE("bitcast")
{
    SECTION("num -> num")
    {
        mcjit_require(
            [](i32 x) { return bitcast<f32>(x); },
            12345, std::bit_cast<float>(12345));

        mcjit_require(
            [](f64 x) { return bitcast<u64>(x); },
            123456.789, std::bit_cast<uint64_t>(123456.789));
    }

    SECTION("num -> ptr")
    {
        float v = 0;
        mcjit_require(
            [](u64 x) { return bitcast<ptr<f32>>(x); },
            reinterpret_cast<uint64_t>(&v), &v);
    }

    SECTION("ptr -> num")
    {
        float v = 0;
        mcjit_require(
            [](ptr<f32> x) { return bitcast<i64>(x); },
            &v, reinterpret_cast<int64_t>(&v));
    }
}
