#pragma once

#include <cuj/dsl/dsl.h>

CUJ_NAMESPACE_BEGIN(cuj::cstd)

struct LCGData
{
    uint32_t state;
};

CUJ_PROXY_CLASS_EX(LCG, LCGData, state)
{
    CUJ_BASE_CONSTRUCTORS

    LCG();

    explicit LCG(u32 seed);

    void set_seed(u32 seed);

    f32 uniform_float();

    f64 uniform_double();

    u32 next_state();
};

CUJ_NAMESPACE_END(cuj::cstd)
