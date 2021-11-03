#include <test/test.h>

struct TypeTestStruct0
{
    int x;
};

struct TypeTestStruct1
{
    int y;
    TypeTestStruct0 s0;
};

struct TypeTestStruct2
{
    int z;
};

namespace TypeTest
{
    struct TypeTestStruct3
    {
        float x, y, z;
    };
}

CUJ_CLASS(TypeTestStruct0, x)
{
    using CUJBase::CUJBase;
};

CUJ_CLASS(TypeTestStruct1, y, s0)
{
    using CUJBase::CUJBase;
};

CUJ_REGISTER_GLOBAL_CLASS(TypeTestStruct2, i32);

namespace TypeTest2
{
    CUJ_PROXY_CLASS(CTypeTestStruct3, TypeTest::TypeTestStruct3, x, y, z)
    {
        using CUJBase::CUJBase;
    };
}

CUJ_REGISTER_GLOBAL_CLASS(TypeTest::TypeTestStruct3, TypeTest2::CTypeTestStruct3);

static_assert(std::is_same_v<
    decltype(std::declval<CUJProxyTypeTestStruct1>().s0),
    CUJProxyTypeTestStruct0>);

static_assert(std::is_same_v<Variable<TypeTestStruct2>, i32>);

static_assert(std::is_same_v<
    Variable<TypeTest::TypeTestStruct3>,
    TypeTest2::CTypeTestStruct3>);

TEST_CASE("type")
{
    SECTION("nested class")
    {
        ScopedContext ctx;

        auto entry = to_callable<int>([](const Variable<TypeTestStruct1> &s)
        {
            $return(s.s0.x + s.y);
        });

        auto jit = ctx.gen_native_jit();
        auto entry_func = jit.get_function(entry);

        REQUIRE(entry_func);
        if(entry_func)
        {
            TypeTestStruct1 s = { 2, { 1 } };
            REQUIRE(entry_func(&s) == 3);
        }
    }
}
