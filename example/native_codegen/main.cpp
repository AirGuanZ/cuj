#include <iostream>

#include <cuj/cuj.h>

using namespace cuj;

void run()
{
    /*ScopedContext context;

    unsigned int n = 0;
    std::cout << "Enter n: ";
    std::cin >> n;

    auto pow_n = context.add_function<int64_t>(
        [n]($int x) mutable
    {
        $i64 result = 1;
        $i64 base = x;
        while(n)
        {
            if(n & 1)
                result = result * base;
            base = base * base;
            n >>= 1;
        }
        $return(result);
    });

    gen::LLVMIRGenerator llvm_gen;
    llvm_gen.generate(context.gen_ir());
    std::cout << std::endl << llvm_gen.get_string() << std::endl;

    auto codegen_result = context.gen_native_jit();
    auto pow_n_func = codegen_result.get_symbol(pow_n);

    std::cout << "n = " << n << std::endl;
    for(int i = 0; i <= 9; ++i)
        std::cout << i << " ^ n = " << pow_n_func(i) << std::endl;*/

    using namespace builtin;
    using namespace math;

#define ADD_TEST_EXPR(EXPR)                                                     \
    do {                                                                        \
        ScopedContext ctx;                                                      \
        auto approx_eq = to_callable<bool>(                                     \
            [](const Float2 &a, const Float2 &b)                                \
        {                                                                       \
            $return(                                                            \
                math::abs(a->x - b->x) < 0.01f &&                               \
                math::abs(a->y - b->y) < 0.01f);                                \
        });                                                                     \
        auto test = to_callable<bool>(                                          \
            [&]                                                                 \
        {                                                                       \
            $return(EXPR);                                                      \
        });                                                                     \
        auto jit = ctx.gen_native_jit();                                        \
        auto func = jit.get_symbol(test);                                       \
    } while(false)

    ADD_TEST_EXPR(approx_eq(
        make_float2(1, 2) + make_float2(3, 4), make_float2(4, 6)));

    ADD_TEST_EXPR(approx_eq(
        make_float2(1, 2) - make_float2(3, 4), make_float2(-2, -2)));

    ADD_TEST_EXPR(approx_eq(
        make_float2(1, 2) * make_float2(3, 4), make_float2(3, 8)));

    ADD_TEST_EXPR(approx_eq(
        make_float2(1, 2) / make_float2(3, 4), make_float2(1.0f / 3, 0.5f)));

    ADD_TEST_EXPR(approx_eq(
        make_float2(1, 2) + 3.0f, make_float2(4, 5)));

    ADD_TEST_EXPR(approx_eq(
        make_float2(1, 2) - 3.0f, make_float2(-2, -1)));

    ADD_TEST_EXPR(approx_eq(
        make_float2(1, 2) * 3, make_float2(3, 6)));

    ADD_TEST_EXPR(approx_eq(
        make_float2(1, 2) / 3, make_float2(1.0f / 3, 2.0f / 3)));

    ADD_TEST_EXPR(approx_eq(
        3 + make_float2(1, 2), make_float2(4, 5)));

    ADD_TEST_EXPR(approx_eq(
        3 - make_float2(1, 2), make_float2(2, 1)));

    ADD_TEST_EXPR(approx_eq(
        3 * make_float2(1, 2), make_float2(3, 6)));

    ADD_TEST_EXPR(approx_eq(
        3 / make_float2(1, 2), make_float2(3, 1.5f)));

    ADD_TEST_EXPR(
        math::abs(dot(make_float2(1, 2), make_float2(3, 4)) - 11) < 0.001f);

    ADD_TEST_EXPR(
        math::abs(cos(make_float2(0, 2), make_float2(0, -3)) + 1) < 0.001f);

#undef ADD_TEST_EXPR
}

int main()
{
    try
    {
        run();
    }
    catch(const std::exception &e)
    {
        std::cerr << e.what() << std::endl;
    }
}
