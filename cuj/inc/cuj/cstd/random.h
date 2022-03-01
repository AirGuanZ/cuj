#pragma once

#include <cuj/dsl/dsl.h>

CUJ_NAMESPACE_BEGIN(cuj::cstd)

struct LCGData
{
    uint32_t state;

    LCGData();

    explicit LCGData(uint32_t seed);
};

CUJ_PROXY_CLASS_EX(LCG, LCGData, state)
{
    CUJ_BASE_CONSTRUCTORS

    using Data = LCGData;

    LCG();

    explicit LCG(u32 seed);

    void set_seed(u32 seed);

    f32 uniform_float();

    f64 uniform_double();

    u32 next_state();
};

struct PCGData
{
    uint64_t state, inc;

    PCGData();

    explicit PCGData(uint64_t seed);

    uint32_t next_state();
};

CUJ_PROXY_CLASS_EX(PCG, PCGData, state, inc)
{
    CUJ_BASE_CONSTRUCTORS

    using Data = PCGData;

    PCG();

    explicit PCG(u64 seed);

    f32 uniform_float();

    f64 uniform_double();

    u32 next_state();
};

CUJ_NAMESPACE_END(cuj::cstd)
