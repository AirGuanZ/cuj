#pragma once

#include <map>
#include <string>
#include <vector>

#include <cuj/ir/stat.h>

CUJ_NAMESPACE_BEGIN(cuj::ir)

struct Function
{
    enum class Type
    {
        Default,
        Kernel,
    };

    struct Arg
    {
        int alloc_index;
    };

    Type        type;
    std::string name;

    std::vector<Arg> args;
    const ir::Type  *ret_type;

    std::map<int, RC<Allocation>> index_to_allocs;

    RC<Block> body;
};

CUJ_NAMESPACE_END(cuj::ir)
