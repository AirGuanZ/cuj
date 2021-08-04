#pragma once

#include <cuj/ast/ast.h>

CUJ_NAMESPACE_BEGIN(cuj::builtin)

void assert_impl(
    RC<ast::Block>                         cond_block,
    RC<ast::InternalArithmeticValue<bool>> cond,
    const Pointer<char>                   &message,
    const Pointer<char>                   &file,
    const u32                             &line,
    const Pointer<char>                   &function);

#define CUJ_ASSERT_STR_HELPER(X) #X

#define CUJ_ASSERT(EXPR) \
    CUJ_ASSERT_IMPL(EXPR, CUJ_ASSERT_STR_HELPER(EXPR), __FILE__, __LINE__, __FUNCTION__)

#define CUJ_ASSERT_IMPL(EXPR, MSG, FILE, LINE, FUNC)                            \
    do {                                                                        \
        (void)::cuj::builtin::AssertBuilder(                                    \
            [&] { return (EXPR); }, MSG, FILE, LINE, FUNC);                     \
    } while(false)

class AssertBuilder
{
public:

    template<typename F>
    explicit AssertBuilder(
        const F           &calc_cond_func,
        const std::string &message,
        const std::string &file,
        uint32_t           line,
        const std::string &function)
    {
        auto func = get_current_function();
        auto cond_block = newRC<ast::Block>();

        func->push_block(cond_block);
        auto fail_cond = !calc_cond_func();
        func->pop_block();

        auto message_consts  = string_literial(message);
        auto file_consts     = string_literial(file);
        auto function_consts = string_literial(function);

        auto line_literial = ast::create_literial(line);

        builtin::assert_impl(
            cond_block, fail_cond.get_impl(),
            message_consts, file_consts,
            line_literial, function_consts);
    }
};

CUJ_NAMESPACE_END(cuj::builtin)
