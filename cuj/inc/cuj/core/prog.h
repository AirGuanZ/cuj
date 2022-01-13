#pragma once

#include <set>

#include <cuj/core/func.h>

CUJ_NAMESPACE_BEGIN(cuj::core)

struct Prog
{
    std::set<RC<GlobalVar>>     global_vars;
    RC<const TypeSet>           global_type_set;
    std::vector<RC<const Func>> funcs;
};

CUJ_NAMESPACE_END(cuj::core)
