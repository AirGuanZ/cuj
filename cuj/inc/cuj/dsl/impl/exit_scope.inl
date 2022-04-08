#pragma once

#include <cuj/dsl/exit_scope.h>
#include <cuj/dsl/function.h>
#include <cuj/utils/scope_guard.h>

CUJ_NAMESPACE_BEGIN(cuj::dsl)

template<typename F>
void operator+(const ExitScopeBuilder &, F &&body_func)
{
    auto func = FunctionContext::get_func_context();
    auto block = newRC<core::Block>();
    {
        func->push_block(block);
        CUJ_SCOPE_EXIT{ func->pop_block(); };
        std::forward<F>(body_func)();
    }
    func->append_statement(core::MakeScope{
        .body = std::move(block)
    });
}

inline void _exit_current_scope()
{
    FunctionContext::get_func_context()->append_statement(core::ExitScope{});
}

CUJ_NAMESPACE_END(cuj::dsl)
