#pragma once

#include <map>
#include <string>
#include <vector>

#include <cuj/ir/stat.h>
#include <cuj/util/untyped_owner.h>

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

struct ImportedHostFunction
{
    RC<UntypedOwner> context_data;
    
    uint64_t    address;
    bool        is_external;
    std::string symbol_name;

    std::vector<const Type *> arg_types;
    const Type               *ret_type;
};

CUJ_NAMESPACE_END(cuj::ir)
