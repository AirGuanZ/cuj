#pragma once

#include <cuj/ir/type.h>

CUJ_NAMESPACE_BEGIN(cuj::ast)

class StructTypeRecorder
{
    ir::StructType *type_;

    explicit StructTypeRecorder(ir::StructType *type)
        : type_(type)
    {
        
    }

public:

    friend class Context;

    StructTypeRecorder(const StructTypeRecorder &) = delete;

    StructTypeRecorder &operator=(const StructTypeRecorder &) = delete;

    void add_member(const ir::Type *member_type)
    {
        type_->mem_types.push_back(member_type);
    }
};

CUJ_NAMESPACE_END(cuj::ast)
