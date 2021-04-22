#pragma once

#include <cuj/ir/alloc.h>
#include <cuj/ir/type.h>

CUJ_NAMESPACE_BEGIN(cuj::ir)

struct BasicTempValue
{
    const Type *type;
    int         index;

    bool operator<(const BasicTempValue &rhs) const
    {
        return index < rhs.index;
    }
};

struct BasicImmediateValue
{
    Variant<
        uint8_t, uint16_t, uint32_t, uint64_t,
        int8_t, int16_t, int32_t, int64_t,
        bool, float, double> value;
};

struct AllocAddress
{
    int alloc_index;
};

using BasicValue = Variant<
    BasicTempValue, BasicImmediateValue, AllocAddress>;

CUJ_NAMESPACE_END(cuj::ir)
