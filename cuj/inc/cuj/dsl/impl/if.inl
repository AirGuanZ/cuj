#pragma once

#include <cassert>

#include <cuj/core/stat.h>
#include <cuj/dsl/function.h>
#include <cuj/dsl/if.h>
#include <cuj/utils/scope_guard.h>

CUJ_NAMESPACE_BEGIN(cuj::dsl)

inline IfBuilder::~IfBuilder()
{
    assert(then_units_.size());

    core::If ret, *last_stat = &ret;
    for(size_t i = 0; i < then_units_.size(); ++i)
    {
        last_stat->calc_cond = then_units_[i].cond_calc;
        last_stat->cond      = then_units_[i].cond;
        last_stat->then_body = then_units_[i].body;
        if(i < then_units_.size() - 1)
        {
            last_stat->else_body = newRC<core::Stat>(core::If{});
            last_stat = &last_stat->else_body->as<core::If>();
        }
    }

    if(else_body_)
        last_stat->else_body = else_body_;

    FunctionContext::get_func_context()->append_statement(std::move(ret));
}

template<typename F>
IfBuilder &IfBuilder::operator*(F &&cond_func)
{
    assert(then_units_.empty() || then_units_.back().body);
    assert(!else_body_);
    auto func = FunctionContext::get_func_context();
    auto cond_calc = newRC<core::Block>();
    num<bool> cond;
    {
        func->push_block(cond_calc);
        CUJ_SCOPE_EXIT{ func->pop_block(); };
        cond = cond_func();
    }
    then_units_.push_back(ThenUnit{
        std::move(cond_calc), cond._load(), {} });
    return *this;
}

template<typename F>
IfBuilder &IfBuilder::operator/(F &&then_func)
{
    assert(!then_units_.empty() && !then_units_.back().body);
    auto func = FunctionContext::get_func_context();
    auto block = newRC<core::Block>();
    {
        func->push_block(block);
        CUJ_SCOPE_EXIT{ func->pop_block(); };
        std::forward<F>(then_func)();
    }
    then_units_.back().body = newRC<core::Stat>(std::move(*block));
    return *this;
}

template<typename F>
void IfBuilder::operator-(F &&else_func)
{
    assert(!then_units_.empty() && then_units_.back().body && !else_body_);
    auto func = FunctionContext::get_func_context();
    auto block = newRC<core::Block>();
    {
        func->push_block(block);
        CUJ_SCOPE_EXIT{ func->pop_block(); };
        std::forward<F>(else_func)();
    }
    else_body_ = newRC<core::Stat>(std::move(*block));
}

CUJ_NAMESPACE_END(cuj::dsl)
