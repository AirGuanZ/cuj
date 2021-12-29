#pragma once

#include <cuj/common.h>

CUJ_NAMESPACE_BEGIN(cuj::core)

enum class Intrinsic : int32_t
{
#define CUJ_INTRINSIC_TYPE(TYPE) TYPE,

#include "intrinsic_types.txt"

#undef CUJ_INTRINSIC_TYPE
};

const char *intrinsic_name(Intrinsic intrinsic);

CUJ_NAMESPACE_END(cuj::core)
