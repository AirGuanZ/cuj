#pragma once

#include <cassert>

#include <cuj/dsl/function.h>
#include <cuj/dsl/switch.h>
#include <cuj/utils/scope_guard.h>
#include <cuj/utils/unreachable.h>

CUJ_NAMESPACE_BEGIN(cuj::dsl)

namespace detail
{

    inline SwitchBuilder *&get_threadlocal_switch_builder()
    {
        static thread_local SwitchBuilder *ret = nullptr;
        return ret;
    }

} // namespace detail

inline SwitchBuilder *SwitchBuilder::get_current_switch_builder()
{
    return detail::get_threadlocal_switch_builder();
}

inline void SwitchBuilder::set_current_switch_builder(SwitchBuilder *builder)
{
    detail::get_threadlocal_switch_builder() = builder;
}

template<typename T> requires std::is_integral_v<T>
SwitchBuilder::SwitchBuilder(const Arithmetic<T> &value)
{
    val_type_ = core::arithmetic_to_builtin_v<T>;
    switch_s_.value = value._load();
}

template<typename F>
void SwitchBuilder::operator*(F &&stat_func)
{
    auto old_builder = get_current_switch_builder();
    CUJ_SCOPE_EXIT{ set_current_switch_builder(old_builder); };
    set_current_switch_builder(this);
    std::forward<F>(stat_func)();
    FunctionContext::get_func_context()->append_statement(switch_s_);
}

template<typename T> requires std::is_integral_v<T>
SwitchBuilder &SwitchBuilder::add_case(T cond)
{
    assert(switch_s_.branches.empty() || switch_s_.branches.back().body);
    assert(!switch_s_.default_body);
    switch_s_.branches.emplace_back();
    switch(val_type_)
    {
    case core::Builtin::U8:   switch_s_.branches.back().cond.value = uint8_t(cond);  break;
    case core::Builtin::U16:  switch_s_.branches.back().cond.value = uint16_t(cond); break;
    case core::Builtin::U32:  switch_s_.branches.back().cond.value = uint32_t(cond); break;
    case core::Builtin::U64:  switch_s_.branches.back().cond.value = uint64_t(cond); break;
    case core::Builtin::S8:   switch_s_.branches.back().cond.value = int8_t(cond);   break;
    case core::Builtin::S16:  switch_s_.branches.back().cond.value = int16_t(cond);  break;
    case core::Builtin::S32:  switch_s_.branches.back().cond.value = int32_t(cond);  break;
    case core::Builtin::S64:  switch_s_.branches.back().cond.value = int64_t(cond);  break;
    case core::Builtin::Char: switch_s_.branches.back().cond.value = char(cond);     break;
    case core::Builtin::Bool: switch_s_.branches.back().cond.value = bool(cond);     break;
    default: unreachable();
    }
    return *this;
}

template<typename F>
void SwitchBuilder::operator+(F &&case_body_func)
{
    assert(get_current_switch_builder() == this);
    assert(!switch_s_.branches.empty() && !switch_s_.branches.back().body);
    assert(!switch_s_.default_body);
    auto func = FunctionContext::get_func_context();
    auto body = newRC<core::Block>();
    {
        func->push_block(body);
        CUJ_SCOPE_EXIT{ func->pop_block(); };
        std::forward<F>(case_body_func)();
    }
    switch_s_.branches.back().body = std::move(body);
}

inline void SwitchBuilder::set_current_body_fallthrough()
{
    assert(!switch_s_.branches.empty() && !switch_s_.branches.back().body);
    assert(!switch_s_.default_body);
    switch_s_.branches.back().fallthrough = true;
}

template<typename F>
void SwitchBuilder::operator-(F &&default_body_func)
{
    assert(switch_s_.branches.empty() || switch_s_.branches.back().body);
    assert(!switch_s_.default_body);
    auto func = FunctionContext::get_func_context();
    auto body = newRC<core::Block>();
    {
        func->push_block(body);
        CUJ_SCOPE_EXIT{ func->pop_block(); };
        std::forward<F>(default_body_func)();
    }
    switch_s_.default_body = std::move(body);
}

CUJ_NAMESPACE_END(cuj::dsl)
