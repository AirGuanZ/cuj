#include <test/test.h>

TEST_CASE("string")
{
    ScopedContext ctx;

    auto test = to_callable<bool>([]
    {
        auto s = "hello, world!"_cuj;

        boolean ret = true;
        ret = ret && (s[0] == 'h');
        ret = ret && (s[1] == 'e');
        ret = ret && (s[12] == '!');
        ret = ret && (s[13] == '\0');

        $return(ret);
    });

    auto jit = ctx.gen_native_jit();
    auto test_func = jit.get_function(test);

    REQUIRE(test_func);
    if(test_func)
        REQUIRE(test_func() == true);
}
