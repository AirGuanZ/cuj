#pragma once

#include <set>
#include <typeindex>

#include <cuj/ir/func.h>

CUJ_NAMESPACE_BEGIN(cuj::ir)

struct Program
{
    std::set<const Type *>                                       types;
    std::vector<Variant<RC<Function>, RC<ImportedHostFunction>>> funcs;
};

CUJ_NAMESPACE_END(cuj::ir)
