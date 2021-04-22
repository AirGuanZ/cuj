#pragma once

#include <cuj/ast/context.h>
#include <cuj/ast/stat_builder.h>

CUJ_NAMESPACE_BEGIN(cuj::ast)

inline IfBuilder::~IfBuilder()
{
    CUJ_ASSERT(!then_units_.empty());

    auto entry = newRC<If>();

    auto cur_if = entry;
    for(size_t i = 0; i < then_units_.size(); ++i)
    {
        cur_if->set_cond(then_units_[i].cond);
        cur_if->set_then(then_units_[i].block);

        if(i + 1 < then_units_.size())
        {
            auto next_block = newRC<Block>();
            auto next_if = newRC<If>();
            next_block->append(next_if);

            cur_if->set_else(next_block);
            cur_if = next_if;
        }
    }

    if(else_block_)
        cur_if->set_else(else_block_);

    get_current_function()->append_statement(std::move(entry));
}

template<typename T>
IfBuilder &IfBuilder::operator+(const ArithmeticValue<T> &cond)
{
    CUJ_ASSERT(then_units_.empty() || then_units_.back().block);
    CUJ_ASSERT(!else_block_);

    if constexpr(std::is_same_v<T, bool>)
        then_units_.push_back({ cond.get_impl(), nullptr });
    else
    {
        auto cast_impl = newRC<InternalCastArithmeticValue<T, bool>>();
        cast_impl->from = cond.get_impl();
        then_units_.push_back({ std::move(cast_impl), nullptr });
    }

    return *this;
}

inline IfBuilder &IfBuilder::operator+(const std::function<void()> &then_body)
{
    CUJ_ASSERT(!then_units_.empty() && !then_units_.back().block);
    CUJ_ASSERT(!else_block_);

    auto func = get_current_function();
    auto block = newRC<Block>();

    func->push_block(block);
    then_body();
    func->pop_block();

    then_units_.back().block = std::move(block);

    return *this;
}

inline IfBuilder &IfBuilder::operator-(const std::function<void()> &else_body)
{
    CUJ_ASSERT(!then_units_.empty() && then_units_.back().block);
    CUJ_ASSERT(!else_block_);

    auto func = get_current_function();
    auto block = newRC<Block>();

    func->push_block(block);
    else_body();
    func->pop_block();

    else_block_ = std::move(block);

    return *this;
}

template<typename T>
WhileBuilder::WhileBuilder(const ArithmeticValue<T> &cond)
{
    if constexpr(std::is_same_v<T, bool>)
        cond_ = cond.get_impl();
    else
    {
        auto cast_impl = newRC<InternalCastArithmeticValue<T, bool>>();
        cast_impl->from = cond.get_impl();
        cond_ = std::move(cast_impl);
    }
}

inline WhileBuilder::~WhileBuilder()
{
    CUJ_ASSERT(cond_ && block_);
    auto while_stat = newRC<While>(std::move(cond_), std::move(block_));
    get_current_function()->append_statement(std::move(while_stat));
}

inline void WhileBuilder::operator+(const std::function<void()> &body_func)
{
    CUJ_ASSERT(cond_ && !block_);

    auto func = get_current_function();
    auto block = newRC<Block>();

    func->push_block(block);
    body_func();
    func->pop_block();

    block_ = std::move(block);
}

CUJ_NAMESPACE_END(cuj::ast)
