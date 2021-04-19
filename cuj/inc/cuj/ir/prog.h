#pragma once

#include <typeindex>

#include <cuj/ir/func.h>

CUJ_NAMESPACE_BEGIN(cuj::ir)

struct Program
{
    std::map<std::type_index, RC<Type>> types;
    std::vector<RC<Function>>           funcs;
};

CUJ_NAMESPACE_END(cuj::ir)
