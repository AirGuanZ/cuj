#pragma once

#include <stack>

#include <cuj/ir/func.h>
#include <cuj/ir/prog.h>

CUJ_NAMESPACE_BEGIN(cuj::ir)

class IRBuilder
{
public:

    void set_assertion(bool enabled);

    bool is_assertion_enabled() const;

    const Program &get_prog() const;

    // type desc

    void add_type(const Type *type);

    // function signature

    void begin_function(
        std::string name, Function::Type type, const Type *ret_type);

    void end_function();

    void add_function_arg(int alloc_index);

    void add_host_imported_function(RC<ImportedHostFunction> func);

    // local allocation

    void add_alloc(int alloc_index, const Type *type);

    RC<Allocation> get_alloc(int alloc_index) const;

    // statement block

    void push_block(RC<Block> block);

    void pop_block();

    void append_statement(RC<Statement> stat);

    void append_assign(BasicTempValue lhs, Value rhs);

    BasicTempValue gen_temp_value(const Type *type);

private:

    Program prog_;

    RC<Function>          cur_func_;
    std::stack<RC<Block>> blocks_;

    int next_temp_value_ = 0;
    
#if defined(_DEBUG) || defined(DEBUG)
    bool enable_assertion_ = true;
#else
    bool enable_assertion_ = false;
#endif
};

CUJ_NAMESPACE_END(cuj::ir)
