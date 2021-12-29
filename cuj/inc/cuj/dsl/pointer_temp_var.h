#pragma once

#include <any>

#include <cuj/dsl/pointer.h>

CUJ_NAMESPACE_BEGIN(cuj::dsl)

class PointerTempVarContext
{
public:

    static void push_context(PointerTempVarContext *context);

    static void pop_context();

    static PointerTempVarContext *get_context();

    void add_var(std::any var);

private:

    std::vector<std::any> vars_;
};

CUJ_NAMESPACE_END(cuj::dsl)
