#pragma once

#include <cuj/builtin/math/basic.h>

CUJ_NAMESPACE_BEGIN(cuj::builtin::math)

struct PCGData
{
    uint64_t state;
    uint64_t inc;
};

CUJ_PROXY_CLASS(PCG, PCGData, state, inc)
{
    using CUJBase::CUJBase;

    PCG();

    explicit PCG(u64 sequence_index);

    void set_sequence(u64 sequence_index);

    u32 uniform_uint32();

    u32 uniform_uint32(u32 b);

    f32 uniform_float();

    f64 uniform_double();
};

CUJ_NAMESPACE_END(cuj::builtin::math)
