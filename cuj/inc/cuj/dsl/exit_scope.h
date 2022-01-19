#pragma once

#include <cuj/utils/anonymous_name.h>
#include <cuj/utils/uncopyable.h>

CUJ_NAMESPACE_BEGIN(cuj::dsl)

class ExitScopeBuilder { };

class ScopedExitScopeBuilder : public Uncopyable
{
    RC<core::Block> block_;

public:

    ScopedExitScopeBuilder();

    ~ScopedExitScopeBuilder();
};

template<typename F>
void operator+(const ExitScopeBuilder &, F &&body_func);

inline void _exit_current_scope();

#define $scope      ::cuj::dsl::ExitScopeBuilder{}+[&]
#define $exit_scope ::cuj::dsl::_exit_current_scope()

#define $declare_scope                                                          \
    auto CUJ_ANONYMOUS_NAME(_cuj_declare_scope) =                               \
        ::cuj::dsl::ScopedExitScopeBuilder{}

CUJ_NAMESPACE_END(cuj::dsl)
