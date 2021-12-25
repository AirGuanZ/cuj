#include "test.h"

TEST_CASE("operator")
{
    SECTION("+")
    {
        mcjit_require([](i32 a, i32 b) { return a + b; }, 1, 2, 3);
        mcjit_require([](f32 a, f32 b) { return a + b; }, 1.0f, 2.0f, 3.0f);
        mcjit_require([](f32 a) { return a + 2.0f; }, 2.0f, 4.0f);
        mcjit_require([](f32 a) { return 2.0f + a; }, 2.0f, 4.0f);
        mcjit_require([](f32 a) { ref b = a; return b + 2.0f; }, 2.0f, 4.0f);
        mcjit_require([](f32 a) { ref b = a; return 2.0f + b; }, 2.0f, 4.0f);
    }

    SECTION("-")
    {
        mcjit_require([](i32 a, i32 b) { return a - b; }, 1, 2, -1);
        mcjit_require([](f32 a, f32 b) { return a - b; }, 1.0f, 2.0f, -1.0f);
        mcjit_require([](f32 a) { return a - 2.0f; }, 2.0f, 0.0f);
        mcjit_require([](f32 a) { return 2.0f - a; }, 2.0f, 0.0f);
        mcjit_require([](f32 a) { ref b = a; return b - 2.0f; }, 2.0f, 0.0f);
        mcjit_require([](f32 a) { ref b = a; return 2.0f - b; }, 2.0f, 0.0f);
    }

    SECTION("*")
    {
        mcjit_require([](i32 a, i32 b) { return a * b; }, 2, 3, 6);
        mcjit_require([](f32 a, f32 b) { return a * b; }, 2.0f, 3.0f, 6.0f);
        mcjit_require([](f32 a) { return a * 2.0f; }, 2.0f, 4.0f);
        mcjit_require([](f32 a) { return 2.0f * a; }, 2.0f, 4.0f);
        mcjit_require([](f32 a) { ref b = a; return b * 2.0f; }, 2.0f, 4.0f);
        mcjit_require([](f32 a) { ref b = a; return 2.0f * b; }, 2.0f, 4.0f);
    }

    SECTION("/")
    {
        mcjit_require([](i32 a, i32 b) { return a / b; }, 7, 3, 2);
        mcjit_require([](f32 a, f32 b) { return a / b; }, 2.0f, 3.0f, 2.0f / 3.0f);
        mcjit_require([](f32 a) { return a / 2.0f; }, 2.0f, 1.0f);
        mcjit_require([](f32 a) { return 2.0f / a; }, 2.0f, 1.0f);
        mcjit_require([](f32 a) { ref b = a; return b / 2.0f; }, 2.0f, 1.0f);
        mcjit_require([](f32 a) { ref b = a; return 2.0f / b; }, 2.0f, 1.0f);
    }

    SECTION("%")
    {
        with_mcjit(
            [](i32 a, i32 b)
        {
            return a % b;
        },
            [](auto f)
        {
            REQUIRE(f(1, 2) == 1);
            REQUIRE(f(5, 3) == 2);
            REQUIRE(f(0, 3) == 0);
            REQUIRE(f(-5, 4) == -1);
        });

        with_mcjit(
            [](u32 a, u32 b)
        {
            return a % b;
        },
            [](auto f)
        {
            REQUIRE(f(1, 2) == 1);
            REQUIRE(f(5, 3) == 2);
            REQUIRE(f(0, 3) == 0);
        });
    }

    SECTION("==")
    {
        with_mcjit(
            [](i32 a, i32 b)
        {
            return i32(a == b);
        },
            [](auto f)
        {
            REQUIRE(f(1, 2) == 0);
            REQUIRE(f(-5, -5) == 1);
        });
    }

    SECTION("!=")
    {
        with_mcjit(
            [](i32 a, i32 b)
        {
            return i32(a != b);
        },
            [](auto f)
        {
            REQUIRE(f(1, 2) == 1);
            REQUIRE(f(-5, -5) == 0);
        });
    }

    SECTION(">")
    {
        with_mcjit(
            [](i32 a, i32 b)
        {
            return i32(a > b);
        },
            [](auto f)
        {
            REQUIRE(f(1, 2) == 0);
            REQUIRE(f(-5, -5) == 0);
            REQUIRE(f(2, -7) == 1);
        });
    }

    SECTION(">=")
    {
        with_mcjit(
            [](i32 a, i32 b)
        {
            return i32(a >= b);
        },
            [](auto f)
        {
            REQUIRE(f(1, 2) == 0);
            REQUIRE(f(-5, -5) == 1);
            REQUIRE(f(2, -7) == 1);
        });
    }

    SECTION("<")
    {
        with_mcjit(
            [](i32 a, i32 b)
        {
            return i32(a < b);
        },
            [](auto f)
        {
            REQUIRE(f(1, 2) == 1);
            REQUIRE(f(-5, -5) == 0);
            REQUIRE(f(2, -7) == 0);
        });
    }

    SECTION("<=")
    {
        with_mcjit(
            [](i32 a, i32 b)
        {
            return i32(a <= b);
        },
            [](auto f)
        {
            REQUIRE(f(1, 2) == 1);
            REQUIRE(f(-5, -5) == 1);
            REQUIRE(f(2, -7) == 0);
        });
    }

    SECTION("<<")
    {
        with_mcjit(
            [](i32 a, i32 b)
        {
            return a << b;
        },
            [](auto f)
        {
            REQUIRE(f(2, 3) == 2 << 3);
            REQUIRE(f(13579, 12) == 13579 << 12);
        });
    }

    SECTION(">>")
    {
        with_mcjit(
            [](u32 a, u32 b)
        {
            return a >> b;
        },
            [](auto f)
        {
            REQUIRE(f(2u, 3u) == 2u >> 3u);
            REQUIRE(f(135797895u, 7u) == 135797895u >> 7u);
        });
    }

    SECTION("&")
    {
        with_mcjit(
            [](i32 a, i32 b)
        {
            return a & b;
        },
            [](auto f)
        {
            REQUIRE(f(0b1011011, 0b1011010) == (0b1011011 & 0b1011010));
            REQUIRE(f(123456, 234567) == (123456 & 234567));
        });
    }

    SECTION("|")
    {
        with_mcjit(
            [](i32 a, i32 b)
        {
            return a | b;
        },
            [](auto f)
        {
            REQUIRE(f(0b1011011, 0b1011010) == (0b1011011 | 0b1011010));
            REQUIRE(f(123456, 234567) == (123456 | 234567));
        });
    }

    SECTION("^")
    {
        with_mcjit(
            [](i32 a, i32 b)
        {
            return a ^ b;
        },
            [](auto f)
        {
            REQUIRE(f(0b1011011, 0b1011010) == (0b1011011 ^ 0b1011010));
            REQUIRE(f(123456, 234567) == (123456 ^ 234567));
        });
    }

    SECTION("-")
    {
        mcjit_require([](i32 x) { return -x; }, 1, -1);
        mcjit_require([](i32 x) { return -x; }, 0, 0);
        mcjit_require([](f32 x) { return -x; }, -5.0f, 5.0f);
    }

    SECTION("~")
    {
        mcjit_require([](i32 x) { return ~x; }, 0b1011011, ~0b1011011);
        mcjit_require([](u32 x) { return ~x; }, 234567u, ~234567u);
    }

    SECTION("!")
    {
        mcjit_require([](i32 x) { return i32(!(x != 0)); }, 1, 0);
        mcjit_require([](i32 x) { return i32(!(x != 0)); }, 0, 1);
    }
}
