#pragma once

#include <stack>

#include <cuj/ast/stat.h>
#include <cuj/ir/func.h>
#include <cuj/util/uncopyable.h>

CUJ_NAMESPACE_BEGIN(cuj::ast)

class Function : public Uncopyable
{
    struct StackAllocation
    {
        const ir::Type *type;
        int             alloc_index;
    };

    std::vector<StackAllocation> stack_allocs_;
    std::vector<int>             arg_indices_;

    std::stack<RC<Block>> blocks_;

    std::string        name_;
    ir::Function::Type type_ = ir::Function::Type::Default;

    template<typename T, typename...Args>
    Value<T> alloc_stack_var(bool is_arg, Args &&...args);

public:

    Function(std::string name, ir::Function::Type type);

    void set_name(std::string name);

    void set_type(ir::Function::Type type);

    void append_statement(RC<Statement> stat);

    void push_block(RC<Block> block);

    void pop_block();

    template<typename T, typename...Args>
    Value<T> create_stack_var(Args &&...args);
    
    RC<InternalStackAllocationValue> alloc_on_stack(const ir::Type *type);

    template<typename T>
    Value<T> create_arg();

    void gen_ir(ir::IRBuilder &builder) const;
};

CUJ_NAMESPACE_END(cuj::ast)
