#pragma once

#include <cuj/common.h>

CUJ_NAMESPACE_BEGIN(cuj::dsl)

class ExitScopeBuilder { };

template<typename F>
void operator+(const ExitScopeBuilder &, F &&body_func);

inline void _exit_current_scope();

#define $scope      ::cuj::dsl::ExitScopeBuilder{}+[&]
#define $exit_scope ::cuj::dsl::_exit_current_scope()

CUJ_NAMESPACE_END(cuj::dsl)
