#pragma once

#include <cuj/dsl/dsl.h>

CUJ_NAMESPACE_BEGIN(cuj::cstd)

template<typename...Args>
i32 print(const std::string &format_string, Args...args);

CUJ_NAMESPACE_END(cuj::cstd)

#include <cuj/cstd/impl/system.inl>
