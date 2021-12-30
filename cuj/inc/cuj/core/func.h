#pragma once

#include <cuj/core/stat.h>

CUJ_NAMESPACE_BEGIN(cuj::core)

struct Func
{
    struct Argument
    {
        const Type *type  = nullptr;
        bool is_reference = false;
    };

    enum FuncType
    {
        Regular,
        Kernel,
    };

    std::string name;
    FuncType    type = Regular;

    RC<TypeSet> type_set;

    std::vector<Argument> argument_types;
    Argument              return_type;

    std::vector<const Type *> local_alloc_types;
    RC<Block>                 root_block;
};

CUJ_NAMESPACE_END(cuj::core)
