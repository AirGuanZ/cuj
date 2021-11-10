#pragma once

#include <cuj/builtin/math/basic.h>

CUJ_NAMESPACE_BEGIN(cuj::builtin::math)

struct LCGData
{
    uint32_t state;
};

CUJ_PROXY_CLASS(LCG, LCGData, state)
{
    using CUJBase::CUJBase;

    LCG();

    explicit LCG(u32 seed);

    void set_seed(u32 seed);

    f32 uniform_float();

    f64 uniform_double();

    u32 next_state();
};

CUJ_NAMESPACE_END(cuj::builtin::math)
