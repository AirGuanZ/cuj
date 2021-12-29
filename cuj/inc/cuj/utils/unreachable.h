#pragma once

#include <cuj/common.h>

CUJ_NAMESPACE_BEGIN(cuj)

[[noreturn]] inline void unreachable()
{
#if defined(_MSC_VER)
    __assume(0);
#elif defined(__clang__) || defined(__GNUC__)
    __builtin_unreachable();
#else
    std::terminate();
#endif
}

CUJ_NAMESPACE_END(cuj)
