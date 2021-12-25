#include "test.h"

#define CHECK_C_FUNC_TYPE_DEFAULT(TYPE, FUNC)                                   \
    do {                                                                        \
        ScopedModule mod;                                                       \
        Function func = FUNC;                                                   \
        MCJIT mcjit;                                                            \
        mcjit.generate(mod);                                                    \
        static_assert(std::is_same_v<                                           \
            TYPE, std::remove_pointer_t<decltype(mcjit.get_function(func))>>);  \
    } while(false)

#define CHECK_C_FUNC_TYPE(TYPE, FUNC)                                           \
    do {                                                                        \
        ScopedModule mod;                                                       \
        Function func = FUNC;                                                   \
        MCJIT mcjit;                                                            \
        mcjit.generate(mod);                                                    \
        REQUIRE(std::is_same_v<                                                 \
            TYPE, std::remove_pointer_t<                                        \
                decltype(mcjit.get_function<TYPE>(func))>>);                    \
    } while(false)

namespace
{

    struct A
    {
        int32_t a;
        float   b;
        void   *c;
    };

    struct B
    {
        A  arr[2];
        B *ptr;
    };

    CUJ_CLASS(A, a, b, c);
    CUJ_CLASS(B, arr, ptr);

} // namespace anonymous

TEST_CASE("mcjit")
{
    CHECK_C_FUNC_TYPE_DEFAULT(int32_t(int32_t, int32_t),    [](i32 a, i32 b)  { return a + b; });
    CHECK_C_FUNC_TYPE_DEFAULT(void(int32_t, int32_t),       [](i32, i32)      {});
    CHECK_C_FUNC_TYPE_DEFAULT(int32_t * (int32_t, int32_t), [](i32 a, i32)    { return a.address(); });
    CHECK_C_FUNC_TYPE_DEFAULT(int32_t(int32_t, int32_t),    [](i32 a, i32)    { ref b = a; return b; });
    CHECK_C_FUNC_TYPE_DEFAULT(void *(),                     []                { ptr ret = nullptr; return ret; });
    CHECK_C_FUNC_TYPE_DEFAULT(void *(void *),               [](ref<cxx<A>> a) { return a.address(); });
    CHECK_C_FUNC_TYPE_DEFAULT(void *(void *),               [](ref<cxx<B>> b) { return b.arr[0].address(); });
    CHECK_C_FUNC_TYPE_DEFAULT(void *(void *),               [](ptr<cxx<B>> b) { return b->ptr; });

    CHECK_C_FUNC_TYPE(const A * (A *),    [](ptr<cxx<A>> a) { return 1 + a; });
    CHECK_C_FUNC_TYPE(void * (const A *), [](ref<cxx<A>> a) { return a.address(); });
}
