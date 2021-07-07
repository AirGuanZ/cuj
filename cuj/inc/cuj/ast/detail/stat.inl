#pragma once

#include <cuj/ast/stat.h>

#include <cuj/ast/detail/call_arg.inl>

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
    RC<InternalPointerValue<L>>     lhs,
    RC<typename Value<R>::ImplType> rhs)
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
        auto cast_op = ir::CastBuiltinOp{ ir::to_builtin_type_value<L>, origin_rhs_val };

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

inline While::While(
    RC<Block>                         cond_block,
    RC<InternalArithmeticValue<bool>> cond,
    RC<Block>                         body)
    : cond_block_(std::move(cond_block)),
      cond_(std::move(cond)),
      body_(std::move(body))
{
    
}

inline void While::gen_ir(ir::IRBuilder &builder) const
{
    ir::While result;

    auto calc_cond_block = newRC<ir::Block>();
    builder.push_block(calc_cond_block);
    cond_block_->gen_ir(builder);
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

template<typename T>
ReturnArithmetic<T>::ReturnArithmetic(RC<InternalArithmeticValue<T>> value)
    : value_(std::move(value))
{
    if(!value_)
    {
        if(get_current_function()->get_return_type() !=
           get_current_context()->get_type<void>())
            throw CUJException("return.type != function.type");
    }
    else
    {
        if(get_current_function()->get_return_type() !=
           get_current_context()->get_type<T>())
            throw CUJException("return.type != function.type");
    }
}

template<typename T>
void ReturnArithmetic<T>::gen_ir(ir::IRBuilder &builder) const
{
    if(!value_)
    {
        builder.append_statment(
            newRC<ir::Statement>(ir::Return{ std::nullopt }));
    }
    else
    {
        auto val = value_->gen_ir(builder);
        builder.append_statment(newRC<ir::Statement>(ir::Return{ val }));
    }
}

template<typename T>
ReturnPointer<T>::ReturnPointer(RC<InternalPointerValue<T>> pointer)
    : pointer_(std::move(pointer))
{
    
}

template<typename T>
void ReturnPointer<T>::gen_ir(ir::IRBuilder &builder) const
{
    auto val = pointer_->gen_ir(builder);
    builder.append_statment(newRC<ir::Statement>(ir::Return{ val }));
}

template<typename T>
ReturnClass<T>::ReturnClass(RC<InternalPointerValue<T>> pointer)
    : pointer_(std::move(pointer))
{
    
}

template<typename T>
void ReturnClass<T>::gen_ir(ir::IRBuilder &builder) const
{
    auto class_ptr = pointer_->gen_ir(builder);
    auto ret_stat = ir::ReturnClass{ class_ptr };
    builder.append_statment(newRC<ir::Statement>(ret_stat));
}

template<typename T>
ReturnArray<T>::ReturnArray(RC<InternalPointerValue<T>> pointer)
    : pointer_(std::move(pointer))
{
    
}

template<typename T>
void ReturnArray<T>::gen_ir(ir::IRBuilder &builder) const
{
    auto class_ptr = pointer_->gen_ir(builder);
    auto ret_stat = ir::ReturnArray{ class_ptr };
    builder.append_statment(newRC<ir::Statement>(ret_stat));
}

template<typename...Args>
CallVoid<Args...>::CallVoid(
    int func_index, const RC<typename Value<Args>::ImplType> &...args)
    : func_index_(func_index), args_{ args... }
{

}

template<typename...Args>
void CallVoid<Args...>::gen_ir(ir::IRBuilder &builder) const
{
    auto context = get_current_context();
    auto func = context->get_function_context(func_index_);

    auto ret_type = context->get_type<void>();

    std::vector<ir::BasicValue> arg_vals;

    std::apply(
        [&](const auto &...arg)
    {
        (call_detail::prepare_arg<
            typename detail::DeValueType<rm_cvref_t<Args>>::Type>(
                builder, arg, arg_vals), ...);
    }, args_);

    builder.append_statment(newRC<ir::Statement>(ir::Call{
        ir::CallOp{
            func->get_name(), std::move(arg_vals), ret_type
        }
    }));
}

template<typename Ret, typename ... Args>
CallClass<Ret, Args...>::CallClass(
    int func_index, const PointerImpl<Ret> &ret_ptr,
    const RC<typename Value<Args>::ImplType> &... args)
        : func_index_(func_index), ret_ptr_(ret_ptr), args_{ args... }
{
    
}

template<typename Ret, typename ... Args>
void CallClass<Ret, Args...>::gen_ir(ir::IRBuilder &builder) const
{
    auto context = get_current_context();
    auto func = context->get_function_context(func_index_);

    auto ret_type = context->get_type<void>();

    std::vector<ir::BasicValue> arg_vals;

    auto ret_ptr_val = ret_ptr_.get_impl()->gen_ir(builder);
    arg_vals.push_back(ret_ptr_val);

    std::apply(
        [&](const auto &...arg)
    {
        (call_detail::prepare_arg<
            typename detail::DeValueType<rm_cvref_t<Args>>::Type>(
                builder, arg, arg_vals), ...);
    }, args_);

    builder.append_statment(newRC<ir::Statement>(ir::Call{
        ir::CallOp{
            func->get_name(), std::move(arg_vals), ret_type
        }
    }));
}

template<typename Ret, typename ... Args>
CallArray<Ret, Args...>::CallArray(
    int func_index, const PointerImpl<Ret> &ret_ptr,
    const RC<typename Value<Args>::ImplType> &... args)
        : func_index_(func_index), ret_ptr_(ret_ptr), args_{ args... }
{
    
}

template<typename Ret, typename ... Args>
void CallArray<Ret, Args...>::gen_ir(ir::IRBuilder &builder) const
{
    auto context = get_current_context();
    auto func = context->get_function_context(func_index_);

    auto ret_type = context->get_type<void>();

    std::vector<ir::BasicValue> arg_vals;

    auto ret_ptr_val = ret_ptr_.get_impl()->gen_ir(builder);
    arg_vals.push_back(ret_ptr_val);

    std::apply(
        [&](const auto &...arg)
    {
        (call_detail::prepare_arg<
            typename detail::DeValueType<rm_cvref_t<Args>>::Type>(
                builder, arg, arg_vals), ...);
    }, args_);

    builder.append_statment(newRC<ir::Statement>(ir::Call{
        ir::CallOp{
            func->get_name(), std::move(arg_vals), ret_type
        }
    }));
}

CUJ_NAMESPACE_END(cuj::ast)
