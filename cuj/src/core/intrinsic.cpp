#include <string>

#include <cuj/core/intrinsic.h>

CUJ_NAMESPACE_BEGIN(cuj::core)

const char *intrinsic_name(Intrinsic intrinsic)
{
    switch(intrinsic)
    {
#define CUJ_INTRINSIC_TYPE(TYPE) \
    case Intrinsic::TYPE: return "__cuj_intrinsic_" #TYPE;
#include <cuj/core/intrinsic_types.txt>
#undef CUJ_INTRINSIC_TYPE
    }

    throw CujException(
        "unknown intrinsic type: " + std::to_string(static_cast<
            std::underlying_type_t<Intrinsic>>(intrinsic)));
}

CUJ_NAMESPACE_END(cuj::core)
