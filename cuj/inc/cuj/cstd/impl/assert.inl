#pragma once

#include <cuj/dsl/function.h>

CUJ_NAMESPACE_BEGIN(cuj::cstd)

inline void _assert_fail(
    ptr<char_t> message, ptr<char_t> file, i32 line, ptr<char_t> function)
{
    auto func = dsl::FunctionContext::get_func_context();
    core::CallFuncStat call = {
        core::CallFunc{
            .intrinsic = core::Intrinsic::assert_fail,
            .args = {
                newRC<core::Expr>(message._load()),
                newRC<core::Expr>(file._load()),
                newRC<core::Expr>(line._load()),
                newRC<core::Expr>(function._load()),
            }
        }
    };
    func->append_statement(std::move(call));
}

CUJ_NAMESPACE_END(cuj::cstd)
