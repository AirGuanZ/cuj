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
    RC<InternalPointerValue<L>>    lhs_;
    RC<InternalArithmeticValue<R>> rhs_;

public:

    Store(
        RC<InternalPointerValue<L>>    lhs,
        RC<InternalArithmeticValue<R>> rhs);

    void gen_ir(ir::IRBuilder &builder) const override;
};

class If : public Statement
{
    RC<InternalArithmeticValue<bool>> cond_;
    RC<Block>                         then_block_;
    RC<Block>                         else_block_;

public:
    
    void set_cond(RC<InternalArithmeticValue<bool>> cond);

    void set_then(RC<Block> then_block);

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

class Break : public Statement
{
public:

    void gen_ir(ir::IRBuilder &builder) const override;
};

class Continue : public Statement
{
public:

    void gen_ir(ir::IRBuilder &builder) const override;
};

template<typename T>
class ReturnArithmetic : public Statement
{
    RC<InternalArithmeticValue<T>> value_;

public:

    explicit ReturnArithmetic(RC<InternalArithmeticValue<T>> value);

    void gen_ir(ir::IRBuilder &builder) const override;
};

template<typename T>
class ReturnPointer : public Statement
{
    RC<InternalPointerValue<T>> pointer_;

public:

    explicit ReturnPointer(RC<InternalPointerValue<T>> pointer);

    void gen_ir(ir::IRBuilder &builder) const override;
};

template<typename...Args>
class CallVoid : public Statement
{
    int                        func_index_;
    std::tuple<Value<Args>...> args_;

public:

    explicit CallVoid(int func_index, const Value<Args> &...args);

    void gen_ir(ir::IRBuilder &builder) const override;
};

CUJ_NAMESPACE_END(cuj::ast)
