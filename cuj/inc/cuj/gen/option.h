#pragma once

#include <cuj/common.h>

CUJ_NAMESPACE_BEGIN(cuj::gen)

enum class OptimizeLevel
{
    O0,
    O1,
    O2,
    O3,
    Default = O3
};

struct Options
{
    OptimizeLevel optimize_level     = OptimizeLevel::Default;
    bool          fast_math          = false;
    bool          auto_vectorization = true;
    bool          relocatable_code   = false;
};

CUJ_NAMESPACE_END(cuj::gen)
