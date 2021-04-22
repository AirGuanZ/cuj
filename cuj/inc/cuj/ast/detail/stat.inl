#pragma once

#include <cuj/ast/stat.h>

CUJ_NAMESPACE_BEGIN(cuj::ast)

inline void Block::append(RC<Statement> stat)
{
    stats_.push_back(std::move(stat));
}

inline void Block::gen_ir(ir::IRBuilder &builder) const
{
    for(auto &s : stats_)
        s->gen_ir(builder);
}

template<typename L, typename R>
Store<L, R>::Store(
    RC<InternalArithmeticValue<size_t>> lhs,
    RC<InternalArithmeticValue<R>>      rhs)
    : lhs_(std::move(lhs)), rhs_(std::move(rhs))
{
    
}

template<typename L, typename R>
void Store<L, R>::gen_ir(ir::IRBuilder &builder) const
{
    auto lhs_val = lhs_->gen_ir(builder);
    
    if constexpr(std::is_same_v<L, R>)
    {
        auto rhs_val = rhs_->gen_ir(builder);
        builder.append_statment(
            newRC<ir::Statement>(ir::Store{ lhs_val, rhs_val }));
    }
    else
    {
        auto origin_rhs_val = rhs_->gen_ir(builder);
        auto cast_op = ir::CastOp{ ir::to_builtin_type_value<L>, origin_rhs_val };

        auto lhs_type = get_current_context()->get_type<L>();
        auto rhs_val = builder.gen_temp_value(lhs_type);

        builder.append_assign(rhs_val, cast_op);

        builder.append_statment(
            newRC<ir::Statement>(ir::Store{ lhs_val, rhs_val }));
    }
}

inline void If::set_cond(RC<InternalArithmeticValue<bool>> cond)
{
    cond_ = std::move(cond);
}

inline void If::set_then(RC<Block> then_block)
{
    then_block_ = std::move(then_block);
}

inline void If::set_else(RC<Block> else_block)
{
    else_block_ = std::move(else_block);
}

inline void If::gen_ir(ir::IRBuilder &builder) const
{
    ir::If stat;

    stat.cond = cond_->gen_ir(builder);

    auto then_block = newRC<ir::Block>();
    builder.push_block(then_block);
    then_block_->gen_ir(builder);
    builder.pop_block();
    
    stat.then_block = std::move(then_block);
    
    if(else_block_)
    {
        auto else_block = newRC<ir::Block>();
        builder.push_block(else_block);
        else_block_->gen_ir(builder);
        builder.pop_block();

        stat.else_block = std::move(else_block);
    }

    builder.append_statment(newRC<ir::Statement>(std::move(stat)));
}

inline While::While(RC<InternalArithmeticValue<bool>> cond, RC<Block> body)
    : cond_(std::move(cond)), body_(std::move(body))
{
    
}

inline void While::gen_ir(ir::IRBuilder &builder) const
{
    ir::While result;

    auto calc_cond_block = newRC<ir::Block>();
    builder.push_block(calc_cond_block);
    auto cond = cond_->gen_ir(builder);
    builder.pop_block();

    auto body_block = newRC<ir::Block>();
    builder.push_block(body_block);
    body_->gen_ir(builder);
    builder.pop_block();

    result.calculate_cond = std::move(calc_cond_block);
    result.cond           = cond;
    result.body           = std::move(body_block);

    builder.append_statment(newRC<ir::Statement>(std::move(result)));
}

inline void Break::gen_ir(ir::IRBuilder &builder) const
{
    builder.append_statment(newRC<ir::Statement>(ir::Break{}));
}

inline void Continue::gen_ir(ir::IRBuilder &builder) const
{
    builder.append_statment(newRC<ir::Statement>(ir::Continue{}));
}

CUJ_NAMESPACE_END(cuj::ast)
