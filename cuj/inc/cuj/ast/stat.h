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
    RC<InternalPointerValue<L>>     lhs_;
    RC<typename Value<R>::ImplType> rhs_;

public:

    Store(
        RC<InternalPointerValue<L>>     lhs,
        RC<typename Value<R>::ImplType> rhs);

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
    RC<Block>                         cond_block_;
    RC<InternalArithmeticValue<bool>> cond_;
    RC<Block>                         body_;

public:

    While(
        RC<Block>                         cond_block,
        RC<InternalArithmeticValue<bool>> cond,
        RC<Block>                         body);

    //While(RC<InternalArithmeticValue<bool>> cond, RC<Block> body);

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

template<typename T>
class ReturnClass : public Statement
{
    static_assert(is_cuj_class<T>);

    RC<InternalPointerValue<T>> pointer_;

public:

    explicit ReturnClass(RC<InternalPointerValue<T>> pointer);

    void gen_ir(ir::IRBuilder &builder) const override;
};

template<typename T>
class ReturnArray : public Statement
{
    static_assert(is_array<T>);

    RC<InternalPointerValue<T>> pointer_;

public:

    explicit ReturnArray(RC<InternalPointerValue<T>> pointer);

    void gen_ir(ir::IRBuilder &builder) const override;
};

template<typename...Args>
class CallVoid : public Statement
{
    int                                               func_index_;
    std::tuple<RC<typename Value<Args>::ImplType>...> args_;
    
public:

    explicit CallVoid(
        int func_index, const RC<typename Value<Args>::ImplType> &...args);

    void gen_ir(ir::IRBuilder &builder) const override;
};

template<typename Ret, typename...Args>
class CallClass : public Statement
{
    static_assert(is_cuj_class<Ret>);

    int                                               func_index_;
    PointerImpl<Ret>                                      ret_ptr_;
    std::tuple<RC<typename Value<Args>::ImplType>...> args_;

public:

    CallClass(
        int func_index, const PointerImpl<Ret> &ret_ptr,
        const RC<typename Value<Args>::ImplType> &...args);

    void gen_ir(ir::IRBuilder &builder) const override;
};

template<typename Ret, typename...Args>
class CallArray : public Statement
{
    static_assert(is_array<Ret>);

    int                                               func_index_;
    PointerImpl<Ret>                                      ret_ptr_;
    std::tuple<RC<typename Value<Args>::ImplType>...> args_;

public:

    CallArray(
        int func_index, const PointerImpl<Ret> &ret_ptr,
        const RC<typename Value<Args>::ImplType> &...args);

    void gen_ir(ir::IRBuilder &builder) const override;
};

CUJ_NAMESPACE_END(cuj::ast)
