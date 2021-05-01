#include <test/test.h>

TEST_CASE("pointer")
{
    SECTION("comparison")
    {
        ScopedContext ctx;

        auto test = to_callable<bool>(
            []
        {
            $int a = 0, b = 0;
            Pointer<int> pa = a.address(), pb = b.address(), pa2 = a.address();
            Pointer<int> nil = nullptr;
            
            $bool ret = true;
            ret = ret && (pa != pb);
            ret = ret && (pa == pa2);
            ret = ret && ((pa < pb) ^ (pa > pb));
            ret = ret && (pa <= pa2);
            ret = ret && !(pa < pa2);
            ret = ret && (pa > nullptr);
            ret = ret && (pa > nil);
            ret = ret && (pa >= nil);
            ret = ret && !!pa;
            ret = ret && !nil;

            $return(ret);
        });
        
        auto jit = ctx.gen_native_jit();
        auto test_func = jit.get_symbol(test);

        REQUIRE(test_func);
        if(test_func)
            REQUIRE(test_func() == true);
    }
}
