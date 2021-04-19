#pragma once

#include <functional>

#include <cuj/common.h>

CUJ_NAMESPACE_BEGIN(cuj)
   
class ScopeGuard
{
    std::function<void()> func_;

public:

    template<typename T, typename = std::enable_if_t<std::is_invocable_v<T>>>
    explicit ScopeGuard(T &&func)
        : func_(std::forward<T>(func))
    {

    }

    ScopeGuard(const ScopeGuard &) = delete;

    ScopeGuard &operator=(const ScopeGuard &) = delete;

    ~ScopeGuard()
    {
        if(func_)
            func_();
    }

    void dismiss()
    {
        func_ = std::function<void()>();
    }
};

template<typename T, typename = std::enable_if_t<std::is_invocable_v<T>>>
class FixedScopeGuard
{
    bool call_ = true;

    T func_;

public:

    explicit FixedScopeGuard(T &&func)
        : func_(std::forward<T>(func))
    {
        
    }

    FixedScopeGuard(const FixedScopeGuard &) = delete;

    FixedScopeGuard &operator=(const FixedScopeGuard &) = delete;

    ~FixedScopeGuard()
    {
        if(call_)
            func_();
    }

    void dismiss()
    {
        call_ = false;
    }
};

#define CUJ_SCOPE_GUARD(X) \
    CUJ_SCOPE_GUARD_IMPL0(X, __LINE__)
#define CUJ_SCOPE_GUARD_IMPL0(X, LINE) \
    CUJ_SCOPE_GUARD_IMPL1(X, LINE)
#define CUJ_SCOPE_GUARD_IMPL1(X, LINE) \
    ::cuj::FixedScopeGuard _auto_scope_guard##LINE([&] X)

CUJ_NAMESPACE_END(cuj)
