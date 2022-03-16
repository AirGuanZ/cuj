#pragma once

#include <cuj/core/stat.h>
#include <cuj/dsl/function.h>
#include <cuj/dsl/loop.h>
#include <cuj/utils/scope_guard.h>

CUJ_NAMESPACE_BEGIN(cuj::dsl)

template<typename F>
void LoopBuilder::operator+(F &&body_func)
{
    auto func = FunctionContext::get_func_context();
    auto block = newRC<core::Block>();
    {
        func->push_block(block);
        CUJ_SCOPE_EXIT{ func->pop_block(); };
        std::forward<F>(body_func)();
    }
    func->append_statement(newRC<core::Stat>(core::Loop{
        .body = std::move(block)
    }));
}

template<typename F>
    requires (!std::is_same_v<WhileBuilder, std::remove_cvref_t<F>>)
WhileBuilder::WhileBuilder(F &&cond_func)
{
    auto func = FunctionContext::get_func_context();
    cond_block_ = newRC<core::Block>();
    {
        func->push_block(cond_block_);
        CUJ_SCOPE_EXIT{ func->pop_block(); };
        cond_ = std::forward<F>(cond_func)()._load();
    }
}

template<typename F>
void WhileBuilder::operator+(F &&body_func)
{
    auto func = FunctionContext::get_func_context();
    auto body = newRC<core::Block>();
    {
        func->push_block(body);
        CUJ_SCOPE_EXIT{ func->pop_block(); };
        std::forward<F>(body_func)();
    }
    std::vector body_stats = {
        newRC<core::Stat>(core::If{
            .calc_cond = std::move(cond_block_),
            .cond      = std::move(cond_),
            .then_body = newRC<core::Stat>(std::move(*body)),
            .else_body = newRC<core::Stat>(core::Break{})
        })
    };
    func->append_statement(core::Loop{
        .body = newRC<core::Block>(core::Block{
            .stats = std::move(body_stats)
        })
    });
}

template<typename IT>
ForRangeBuilder<IT>::ForRangeBuilder(IT &idx, IT beg, IT end)
    : idx_(idx)
{
    beg_ = beg;
    end_ = end;
}

template<typename IT>
template<typename F>
void ForRangeBuilder<IT>::operator+(F &&body_func)
{
    IT next_idx = beg_;
    $loop
    {
        idx_ = next_idx;
        $if(idx_ >= end_)
        {
            $break;
        };
        next_idx = next_idx + IT(1);
        (void)std::forward<F>(body_func);
    };
}

inline void _add_break_statement()
{
    FunctionContext::get_func_context()
        ->append_statement(newRC<core::Stat>(core::Break{}));
}

inline void _add_continue_statement()
{
    FunctionContext::get_func_context()
        ->append_statement(newRC<core::Stat>(core::Continue{}));
}

CUJ_NAMESPACE_END(cuj::dsl)
