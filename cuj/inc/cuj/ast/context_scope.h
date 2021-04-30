#pragma once

#include <cuj/ast/context.h>

CUJ_NAMESPACE_BEGIN(cuj::ast)

class ScopedContext : public Context
{
public:
    
    ScopedContext()
    {
        push_context(this);
    }

    ~ScopedContext()
    {
        pop_context();
    }
};

CUJ_NAMESPACE_END(cuj::ast)
