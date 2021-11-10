#include <cuj/builtin/math/lcg.h>

CUJ_NAMESPACE_BEGIN(cuj::builtin::math)

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
    return cast<f32>(next_state() - MIN) * (FloatOneMinusEpsilon / (MAX - MIN));
}

f64 LCG::uniform_double()
{
    return cast<f64>(next_state() - MIN) * (DoubleOneMinusEpsilon / (MAX - MIN));
}

u32 LCG::next_state()
{
    const u32 m = state * AX + CX;
    state = m % MX;
    return state;
}

CUJ_NAMESPACE_END(cuj::builtin::math)
