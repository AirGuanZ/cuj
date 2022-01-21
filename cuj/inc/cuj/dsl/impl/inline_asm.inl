#pragma once

#include <cuj/dsl/inline_asm.h>

CUJ_NAMESPACE_BEGIN(cuj::dsl)

template<typename T>
OutputConstraint::OutputConstraint(std::string name, T &output_var)
    : name(std::move(name)), addr(output_var.address()._load())
{
    
}

template<typename T>
InputConstraint::InputConstraint(std::string name, const T &input_val)
    : name(std::move(name)), val(input_val._load())
{
    
}

CUJ_NAMESPACE_END(cuj::dsl)
