#include <cuj/builtin/math/pcg.h>

CUJ_NAMESPACE_BEGIN(cuj::builtin::math)

#define CUJ_PCG32_DEFAULT_STATE  0x853c49e6748fea9bULL
#define CUJ_PCG32_DEFAULT_STREAM 0xda3e39cb94b95bdbULL
#define CUJ_PCG32_MULT           0x5851f42d4c957f2dULL

constexpr double DoubleOneMinusEpsilon = 0x1.fffffffffffffp-1;
constexpr float  FloatOneMinusEpsilon  = 0x1.fffffep-1;

PCG::PCG()
{
    state = uint64_t(CUJ_PCG32_DEFAULT_STATE);
    inc   = uint64_t(CUJ_PCG32_DEFAULT_STREAM);
}

PCG::PCG(u64 sequence_index)
{
    set_sequence(sequence_index);
}

void PCG::set_sequence(u64 sequence_index)
{
    state = 0u;
    inc = (sequence_index << 1u) | 1u;
    uniform_uint32();
    state = state + uint64_t(CUJ_PCG32_DEFAULT_STATE);
    uniform_uint32();
}

u32 PCG::uniform_uint32()
{
    u64 oldstate = state;
    state = oldstate * uint64_t(CUJ_PCG32_MULT) + inc;
    u32 xorshifted = cast<u32>(((oldstate >> 18u) ^ oldstate) >> 27u);
    u32 rot = cast<u32>(oldstate >> 59u);
    return (xorshifted >> rot) | (xorshifted << ((~rot + 1u) & 31));
}

u32 PCG::uniform_uint32(u32 b)
{
    u32 threshold = (~b + 1u) % b;
    u32 result;
    $while(true)
    {
        u32 r = uniform_uint32();
        $if(r >= threshold)
        {
            result = r % b;
            $break;
        };
    };
    return result;
}

f32 PCG::uniform_float()
{
    return min(
        f32(FloatOneMinusEpsilon), cast<f32>(uniform_uint32() * 0x1p-32f));
}

f64 PCG::uniform_double()
{
    return min(
        f64(DoubleOneMinusEpsilon), cast<f64>(uniform_uint32() * 0x1p-32f));
}

CUJ_NAMESPACE_END(cuj::builtin::math)
