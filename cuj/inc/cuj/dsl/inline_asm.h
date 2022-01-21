#pragma once

#include <cuj/core/stat.h>

CUJ_NAMESPACE_BEGIN(cuj::dsl)

class OutputConstraint
{
public:

    std::string name;
    core::Expr  addr;

    template<typename T>
    OutputConstraint(std::string name, T &output_var);
};

class InputConstraint
{
public:

    std::string name;
    core::Expr  val;

    template<typename T>
    InputConstraint(std::string name, const T &input_val);
};

void inline_asm(
    std::string asm_code, bool additional_side_effects,
    const std::vector<OutputConstraint> &output_constraints,
    const std::vector<InputConstraint> &input_constraints,
    const std::vector<std::string> &clobber_constraints);

CUJ_NAMESPACE_END(cuj::dsl)
