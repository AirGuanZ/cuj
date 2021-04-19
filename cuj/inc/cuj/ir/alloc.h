#pragma once

#include <cuj/ir/type.h>

CUJ_NAMESPACE_BEGIN(cuj::ir)

struct Allocation
{
    const Type *type;
};

CUJ_NAMESPACE_END(cuj::ir)
