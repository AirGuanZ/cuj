#pragma once

#include <cuj/dsl/variable_forward.h>

CUJ_NAMESPACE_BEGIN(cuj::dsl)

inline void _add_return_statement();

template<typename T> requires is_cuj_var_v<T> || is_cuj_ref_v<T>
void _add_return_statement(const T &val);

#define CUJ_RETURN(...) (::cuj::dsl::_add_return_statement(__VA_ARGS__))
#define $return CUJ_RETURN

CUJ_NAMESPACE_END(cuj::dsl)
