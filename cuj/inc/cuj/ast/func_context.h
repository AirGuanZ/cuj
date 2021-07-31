#pragma once

#include <stack>

#include <cuj/ast/stat.h>
#include <cuj/ir/func.h>
#include <cuj/util/uncopyable.h>

CUJ_NAMESPACE_BEGIN(cuj::ast)

class FunctionContext : public Uncopyable
{
    struct StackAllocation
    {
        const ir::Type *type;
        int             alloc_index;
    };

    std::vector<StackAllocation> stack_allocs_;

    std::vector<const ir::Type*> arg_types_;
    std::vector<int>             arg_indices_;

    std::stack<RC<Block>> blocks_;

    std::string        name_;
    ir::Function::Type type_ = ir::Function::Type::Default;

    const ir::Type *ret_type_;

    template<typename T, typename...Args>
    RC<typename Value<T>::ImplType> alloc_stack_var(bool is_arg, Args &&...args);

public:

    FunctionContext(
        std::string                  name,
        ir::Function::Type           type,
        const ir::Type              *ret_type,
        std::vector<const ir::Type*> arg_types);

    void set_name(std::string name);

    void set_type(ir::Function::Type type);

    void append_statement(RC<Statement> stat);

    void push_block(RC<Block> block);

    void pop_block();

    std::string get_name() const;

    int get_arg_count() const;

    const ir::Type *get_arg_type(int index) const;

    const ir::Type *get_return_type() const;

    template<typename T, typename...Args>
    RC<typename Value<T>::ImplType> create_stack_var(Args &&...args);

    template<typename T>
    RC<InternalStackAllocationValue<T>> alloc_on_stack(const ir::Type *type);

    template<typename T>
    RC<typename Value<T>::ImplType> create_arg();

    void gen_ir(ir::IRBuilder &builder) const;
};

CUJ_NAMESPACE_END(cuj::ast)
