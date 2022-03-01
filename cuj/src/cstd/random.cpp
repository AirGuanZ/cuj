#include <cuj/cstd/math.h>
#include <cuj/cstd/random.h>

CUJ_NAMESPACE_BEGIN(cuj::cstd)

namespace
{
    constexpr uint32_t AX = 16807;
    constexpr uint32_t CX = 0;
    constexpr uint32_t MX = 2147483647;

    constexpr uint32_t MIN = 1;
    constexpr uint32_t MAX = MX - 1;

    constexpr double DoubleOneMinusEpsilon = 0x1.fffffffffffffp-1;
    constexpr float  FloatOneMinusEpsilon  = 0x1.fffffep-1;
}

LCGData::LCGData()
    : state(1)
{
    
}

LCGData::LCGData(uint32_t seed)
{
    seed = seed % MX;
    if(!seed)
        seed = 1;
    state = seed;
}

LCG::LCG()
    : LCG(u32(1))
{
    
}

LCG::LCG(u32 seed)
{
    set_seed(seed);
}

void LCG::set_seed(u32 seed)
{
    seed = seed % MX;
    $if(seed == 0)
    {
        seed = 1;
    };
    state = seed;
}

f32 LCG::uniform_float()
{
    constexpr float f = static_cast<float>(1.0 / (MAX - MIN));
    return cstd::min(f32(FloatOneMinusEpsilon), f32(next_state() - MIN) * f);
}

f64 LCG::uniform_double()
{
    return f64(next_state() - MIN) * (DoubleOneMinusEpsilon / (MAX - MIN));
}

u32 LCG::next_state()
{
    const u32 m = state * AX + CX;
    state = m % MX;
    return state;
}

namespace
{

    constexpr uint64_t PCG_DEFAULT_STATE  = 0x853c49e6748fea9bULL;
    constexpr uint64_t PCG_DEFAULT_STREAM = 0xda3e39cb94b95bdbULL;
    constexpr uint64_t PCG_MULT           = 0x5851f42d4c957f2dULL;

} // namespace anonymous

PCGData::PCGData()
{
    state = PCG_DEFAULT_STATE;
    inc = PCG_DEFAULT_STREAM;
}

PCGData::PCGData(uint64_t seed)
{
    state = 0;
    inc = (seed << 1u) | 1u;
    next_state();
    state = state + PCG_DEFAULT_STATE;
    next_state();
}

uint32_t PCGData::next_state()
{
    uint64_t old_state = state;
    state = old_state * PCG_MULT + inc;
    uint32_t xor_shifted = uint32_t(((old_state >> 18u) ^ old_state) >> 27u);
    uint32_t rot = uint32_t(old_state >> 59u);
    return (xor_shifted >> rot) | (xor_shifted << ((~rot + 1u) & 31));
}

PCG::PCG()
{
    state = PCG_DEFAULT_STATE;
    inc = PCG_DEFAULT_STREAM;
}

PCG::PCG(u64 seed)
{
    state = 0;
    inc = (seed << 1u) | 1u;
    next_state();
    state = state + PCG_DEFAULT_STATE;
    next_state();
}

f32 PCG::uniform_float()
{
    return cstd::min(f32(next_state()) * 0x1p-32f, f32(0x1.fffffep-1));
}

f64 PCG::uniform_double()
{
    return f64(uniform_float());
}

u32 PCG::next_state()
{
    u64 old_state = state;
    state = old_state * PCG_MULT + inc;
    u32 xor_shifted = u32(((old_state >> 18u) ^ old_state) >> 27u);
    u32 rot = u32(old_state >> 59u);
    return (xor_shifted >> rot) | (xor_shifted << ((~rot + 1u) & 31));
}

CUJ_NAMESPACE_END(cuj::cstd)
