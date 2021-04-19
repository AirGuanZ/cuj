#pragma once

#include <cuj/ast/expr.h>

CUJ_NAMESPACE_BEGIN(cuj::ast)

class Statement
{
public:

    virtual ~Statement() = default;

    virtual void gen_ir(ir::IRBuilder &builder) const = 0;
};

class Block : public Statement
{
    std::vector<RC<Statement>> stats_;

public:

    void append(RC<Statement> stat);

    auto begin() { return stats_.begin(); }
    auto end()   { return stats_.end();   }

    auto begin() const { return stats_.begin(); }
    auto end()   const { return stats_.end();   }

    void gen_ir(ir::IRBuilder &builder) const override;
};

template<typename L, typename R>
class Store : public Statement
{
    RC<InternalArithmeticValue<size_t>> lhs_;
    RC<InternalArithmeticValue<R>>      rhs_;

public:

    Store(
        RC<InternalArithmeticValue<size_t>> lhs,
        RC<InternalArithmeticValue<R>>      rhs);

    void gen_ir(ir::IRBuilder &builder) const override;
};

class If : public Statement
{
    struct ThenUnit
    {
        RC<InternalArithmeticValue<bool>> cond;
        RC<Block>                         then_block;
    };

    std::vector<ThenUnit> then_units_;
    RC<Block>             else_block_;

public:
    
    void add_then_unit(RC<InternalArithmeticValue<bool>> cond, RC<Block> block);

    void set_else(RC<Block> else_block);

    void gen_ir(ir::IRBuilder &builder) const override;
};

class While : public Statement
{
    RC<InternalArithmeticValue<bool>> cond_;
    RC<Block>                         body_;

public:

    While(RC<InternalArithmeticValue<bool>> cond, RC<Block> body);

    void gen_ir(ir::IRBuilder &builder) const override;
};

CUJ_NAMESPACE_END(cuj::ast)
