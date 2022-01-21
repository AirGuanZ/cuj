#include <cuj/dsl/function.h>
#include <cuj/dsl/inline_asm.h>

CUJ_NAMESPACE_BEGIN(cuj::dsl)

void inline_asm(
    std::string asm_code, bool additional_side_effects,
    const std::vector<OutputConstraint> &output_constraints,
    const std::vector<InputConstraint> &input_constraints,
    const std::vector<std::string> &clobber_constraints)
{
    core::InlineAsm stat;
    stat.asm_string = std::move(asm_code);
    stat.side_effects = additional_side_effects;

    std::string output_constraints_str;
    for(size_t i = 0; i < output_constraints.size(); ++i)
    {
        if(i > 0)
            output_constraints_str += ",";
        auto &c = output_constraints[i];
        output_constraints_str += c.name;
        stat.output_addresses.push_back(c.addr);
    }
    stat.output_constraints = std::move(output_constraints_str);

    std::string input_constaints_str;
    for(size_t i = 0; i < input_constraints.size(); ++i)
    {
        if(i > 0)
            input_constaints_str += ",";
        auto &c = input_constraints[i];
        input_constaints_str += c.name;
        stat.input_values.push_back(c.val);
    }
    stat.input_constraints = std::move(input_constaints_str);

    std::string clobber_constraints_str;
    for(size_t i = 0; i < clobber_constraints.size(); ++i)
    {
        if(i > 0)
            clobber_constraints_str += ",";
        clobber_constraints_str += clobber_constraints[i];
    }
    stat.clobber_constraints = std::move(clobber_constraints_str);

    FunctionContext::get_func_context()
        ->append_statement(core::Stat{ std::move(stat) });
}

void inline_asm(
    std::string asm_code,
    const std::vector<OutputConstraint> &output_constraints,
    const std::vector<InputConstraint> &input_constraints,
    const std::vector<std::string> &clobber_constraints)
{
    inline_asm(
        std::move(asm_code), false,
        output_constraints, input_constraints, clobber_constraints);
}

void inline_asm_volatile(
    std::string asm_code,
    const std::vector<OutputConstraint> &output_constraints,
    const std::vector<InputConstraint> &input_constraints,
    const std::vector<std::string> &clobber_constraints)
{
    inline_asm(
        std::move(asm_code), true,
        output_constraints, input_constraints, clobber_constraints);
}

CUJ_NAMESPACE_END(cuj::dsl)
