#include "test.h"

TEST_CASE("math")
{
    SECTION("mcjit.f32")
    {
        using D1 = f32(*)(f32);
        using D2 = f32(*)(f32, f32);

        mcjit_require(D1(&cstd::abs), -0.2f, 0.2f);
        mcjit_require(D2(&cstd::mod), 4.6f, 1.7f, std::fmod(4.6f, 1.7f));
        mcjit_require(D2(&cstd::rem), 4.6f, 1.7f, std::remainder(4.6f, 1.7f));
        mcjit_require(D1(&cstd::exp), 2.3f, std::exp(2.3f));
        mcjit_require(D1(&cstd::exp2), 2.3f, std::exp2(2.3f));
        mcjit_require(D1(&cstd::exp10), 2.3f, std::pow(10.0f, 2.3f));
        mcjit_require(D1(&cstd::log), 2.3f, std::log(2.3f));
        mcjit_require(D1(&cstd::log2), 2.3f, std::log2(2.3f));
        mcjit_require(D1(&cstd::log10), 2.3f, std::log10(2.3f));
        mcjit_require(D2(&cstd::pow), 3.5f, 4.7f, std::pow(3.5f, 4.7f));
        mcjit_require(D1(&cstd::sqrt), 3.7f, std::sqrt(3.7f));
        mcjit_require(D1(&cstd::rsqrt), 3.7f, 1.0f / std::sqrt(3.7f));
        mcjit_require(D1(&cstd::sin), 3.7f, std::sin(3.7f));
        mcjit_require(D1(&cstd::cos), 3.7f, std::cos(3.7f));
        mcjit_require(D1(&cstd::tan), 3.7f, std::tan(3.7f));
        mcjit_require(D1(&cstd::asin), 0.7f, std::asin(0.7f));
        mcjit_require(D1(&cstd::acos), 0.7f, std::acos(0.7f));
        mcjit_require(D1(&cstd::atan), 3.7f, std::atan(3.7f));
        mcjit_require(D2(&cstd::atan2), 2.2f, -3.4f, std::atan2(2.2f, -3.4f));
        mcjit_require(D1(&cstd::ceil), 2.2f, 3.0f);
        mcjit_require(D1(&cstd::floor), -0.7f, -1.0f);
        mcjit_require(D1(&cstd::trunc), -2.8f, -2.0f);
        mcjit_require(D1(&cstd::round), 0.6f, 1.0f);

        constexpr float f32_inf = std::numeric_limits<float>::infinity();
        constexpr float f32_nan = std::numeric_limits<float>::quiet_NaN();

        mcjit_require([](f32 x) { return i32(cstd::isfinite(x)); }, 2.4f, 1);
        mcjit_require([](f32 x) { return i32(cstd::isfinite(x)); }, f32_inf, 0);
        mcjit_require([](f32 x) { return i32(cstd::isfinite(x)); }, f32_nan, 0);
        mcjit_require([](f32 x) { return i32(cstd::isinf(x)); }, 2.4f, 0);
        mcjit_require([](f32 x) { return i32(cstd::isinf(x)); }, f32_inf, 1);
        mcjit_require([](f32 x) { return i32(cstd::isinf(x)); }, f32_nan, 0);
        mcjit_require([](f32 x) { return i32(cstd::isnan(x)); }, 2.4f, 0);
        mcjit_require([](f32 x) { return i32(cstd::isnan(x)); }, f32_inf, 0);
        mcjit_require([](f32 x) { return i32(cstd::isnan(x)); }, f32_nan, 1);
    }

    SECTION("mcjit.f64")
    {
        using D1 = f64(*)(f64);
        using D2 = f64(*)(f64, f64);

        mcjit_require(D1(&cstd::abs), -0.2, 0.2);
        mcjit_require(D2(&cstd::mod), 4.6, 1.7, std::fmod(4.6, 1.7));
        mcjit_require(D2(&cstd::rem), 4.6, 1.7, std::remainder(4.6, 1.7));
        mcjit_require(D1(&cstd::exp), 2.3, std::exp(2.3));
        mcjit_require(D1(&cstd::exp2), 2.3, std::exp2(2.3));
        mcjit_require(D1(&cstd::exp10), 2.3, std::pow(10.0, 2.3));
        mcjit_require(D1(&cstd::log), 2.3, std::log(2.3));
        mcjit_require(D1(&cstd::log2), 2.3, std::log2(2.3));
        mcjit_require(D1(&cstd::log10), 2.3, std::log10(2.3));
        mcjit_require(D2(&cstd::pow), 3.5, 4.7, std::pow(3.5, 4.7));
        mcjit_require(D1(&cstd::sqrt), 3.7, std::sqrt(3.7));
        mcjit_require(D1(&cstd::rsqrt), 3.7, 1.0 / std::sqrt(3.7));
        mcjit_require(D1(&cstd::sin), 3.7, std::sin(3.7));
        mcjit_require(D1(&cstd::cos), 3.7, std::cos(3.7));
        mcjit_require(D1(&cstd::tan), 3.7, std::tan(3.7));
        mcjit_require(D1(&cstd::asin), 0.7, std::asin(0.7));
        mcjit_require(D1(&cstd::acos), 0.7, std::acos(0.7));
        mcjit_require(D1(&cstd::atan), 3.7, std::atan(3.7));
        mcjit_require(D2(&cstd::atan2), 2.2, -3.4, std::atan2(2.2, -3.4));
        mcjit_require(D1(&cstd::ceil), 2.2, 3.0);
        mcjit_require(D1(&cstd::floor), -0.7, -1.0);
        mcjit_require(D1(&cstd::trunc), -2.8, -2.0);
        mcjit_require(D1(&cstd::round), 0.6, 1.0);

        constexpr double f64_inf = std::numeric_limits<double>::infinity();
        constexpr double f64_nan = std::numeric_limits<double>::quiet_NaN();

        mcjit_require([](f64 x) { return i32(cstd::isfinite(x)); }, 2.4, 1);
        mcjit_require([](f64 x) { return i32(cstd::isfinite(x)); }, f64_inf, 0);
        mcjit_require([](f64 x) { return i32(cstd::isfinite(x)); }, f64_nan, 0);
        mcjit_require([](f64 x) { return i32(cstd::isinf(x)); }, 2.4, 0);
        mcjit_require([](f64 x) { return i32(cstd::isinf(x)); }, f64_inf, 1);
        mcjit_require([](f64 x) { return i32(cstd::isinf(x)); }, f64_nan, 0);
        mcjit_require([](f64 x) { return i32(cstd::isnan(x)); }, 2.4, 0);
        mcjit_require([](f64 x) { return i32(cstd::isnan(x)); }, f64_inf, 0);
        mcjit_require([](f64 x) { return i32(cstd::isnan(x)); }, f64_nan, 1);
    }

#if CUJ_ENABLE_CUDA

    SECTION("cuda.f32")
    {
        using D1 = f32(*)(f32);
        using D2 = f32(*)(f32, f32);

        cuda_require(D1(&cstd::abs), -0.2f, 0.2f);
        cuda_require(D2(&cstd::mod), 4.6f, 1.7f, std::fmod(4.6f, 1.7f));
        cuda_require(D2(&cstd::rem), 4.6f, 1.7f, std::remainder(4.6f, 1.7f));
        cuda_require(D1(&cstd::exp), 2.3f, std::exp(2.3f));
        cuda_require(D1(&cstd::exp2), 2.3f, std::exp2(2.3f));
        cuda_require(D1(&cstd::exp10), 2.3f, std::pow(10.0f, 2.3f));
        cuda_require(D1(&cstd::log), 2.3f, std::log(2.3f));
        cuda_require(D1(&cstd::log2), 2.3f, std::log2(2.3f));
        cuda_require(D1(&cstd::log10), 2.3f, std::log10(2.3f));
        cuda_require(D2(&cstd::pow), 3.5f, 4.7f, std::pow(3.5f, 4.7f));
        cuda_require(D1(&cstd::sqrt), 3.7f, std::sqrt(3.7f));
        cuda_require(D1(&cstd::rsqrt), 3.7f, 1.0f / std::sqrt(3.7f));
        cuda_require(D1(&cstd::sin), 3.7f, std::sin(3.7f));
        cuda_require(D1(&cstd::cos), 3.7f, std::cos(3.7f));
        cuda_require(D1(&cstd::tan), 3.7f, std::tan(3.7f));
        cuda_require(D1(&cstd::asin), 0.7f, std::asin(0.7f));
        cuda_require(D1(&cstd::acos), 0.7f, std::acos(0.7f));
        cuda_require(D1(&cstd::atan), 3.7f, std::atan(3.7f));
        cuda_require(D2(&cstd::atan2), 2.2f, -3.4f, std::atan2(2.2f, -3.4f));
        cuda_require(D1(&cstd::ceil), 2.2f, 3.0f);
        cuda_require(D1(&cstd::floor), -0.7f, -1.0f);
        cuda_require(D1(&cstd::trunc), -2.8f, -2.0f);
        cuda_require(D1(&cstd::round), 0.6f, 1.0f);

        constexpr float f32_inf = std::numeric_limits<float>::infinity();
        constexpr float f32_nan = std::numeric_limits<float>::quiet_NaN();

        cuda_require([](f32 x) { return i32(cstd::isfinite(x)); }, 2.4f, 1);
        cuda_require([](f32 x) { return i32(cstd::isfinite(x)); }, f32_inf, 0);
        cuda_require([](f32 x) { return i32(cstd::isfinite(x)); }, f32_nan, 0);
        cuda_require([](f32 x) { return i32(cstd::isinf(x)); }, 2.4f, 0);
        cuda_require([](f32 x) { return i32(cstd::isinf(x)); }, f32_inf, 1);
        cuda_require([](f32 x) { return i32(cstd::isinf(x)); }, f32_nan, 0);
        cuda_require([](f32 x) { return i32(cstd::isnan(x)); }, 2.4f, 0);
        cuda_require([](f32 x) { return i32(cstd::isnan(x)); }, f32_inf, 0);
        cuda_require([](f32 x) { return i32(cstd::isnan(x)); }, f32_nan, 1);
    }

    SECTION("cuda.f64")
    {
        using D1 = f64(*)(f64);
        using D2 = f64(*)(f64, f64);

        cuda_require(D1(&cstd::abs), -0.2, 0.2);
        cuda_require(D2(&cstd::mod), 4.6, 1.7, std::fmod(4.6, 1.7));
        cuda_require(D2(&cstd::rem), 4.6, 1.7, std::remainder(4.6, 1.7));
        cuda_require(D1(&cstd::exp), 2.3, std::exp(2.3));
        cuda_require(D1(&cstd::exp2), 2.3, std::exp2(2.3));
        cuda_require(D1(&cstd::exp10), 2.3, std::pow(10.0, 2.3));
        cuda_require(D1(&cstd::log), 2.3, std::log(2.3));
        cuda_require(D1(&cstd::log2), 2.3, std::log2(2.3));
        cuda_require(D1(&cstd::log10), 2.3, std::log10(2.3));
        cuda_require(D2(&cstd::pow), 3.5, 4.7, std::pow(3.5, 4.7));
        cuda_require(D1(&cstd::sqrt), 3.7, std::sqrt(3.7));
        cuda_require(D1(&cstd::rsqrt), 3.7, 1.0 / std::sqrt(3.7));
        cuda_require(D1(&cstd::sin), 3.7, std::sin(3.7));
        cuda_require(D1(&cstd::cos), 3.7, std::cos(3.7));
        cuda_require(D1(&cstd::tan), 3.7, std::tan(3.7));
        cuda_require(D1(&cstd::asin), 0.7, std::asin(0.7));
        cuda_require(D1(&cstd::acos), 0.7, std::acos(0.7));
        cuda_require(D1(&cstd::atan), 3.7, std::atan(3.7));
        cuda_require(D2(&cstd::atan2), 2.2, -3.4, std::atan2(2.2, -3.4));
        cuda_require(D1(&cstd::ceil), 2.2, 3.0);
        cuda_require(D1(&cstd::floor), -0.7, -1.0);
        cuda_require(D1(&cstd::trunc), -2.8, -2.0);
        cuda_require(D1(&cstd::round), 0.6, 1.0);

        constexpr double f64_inf = std::numeric_limits<double>::infinity();
        constexpr double f64_nan = std::numeric_limits<double>::quiet_NaN();

        cuda_require([](f64 x) { return i32(cstd::isfinite(x)); }, 2.4, 1);
        cuda_require([](f64 x) { return i32(cstd::isfinite(x)); }, f64_inf, 0);
        cuda_require([](f64 x) { return i32(cstd::isfinite(x)); }, f64_nan, 0);
        cuda_require([](f64 x) { return i32(cstd::isinf(x)); }, 2.4, 0);
        cuda_require([](f64 x) { return i32(cstd::isinf(x)); }, f64_inf, 1);
        cuda_require([](f64 x) { return i32(cstd::isinf(x)); }, f64_nan, 0);
        cuda_require([](f64 x) { return i32(cstd::isnan(x)); }, 2.4, 0);
        cuda_require([](f64 x) { return i32(cstd::isnan(x)); }, f64_inf, 0);
        cuda_require([](f64 x) { return i32(cstd::isnan(x)); }, f64_nan, 1);
    }

#endif // #if CUJ_ENABLE_CUDA
}
