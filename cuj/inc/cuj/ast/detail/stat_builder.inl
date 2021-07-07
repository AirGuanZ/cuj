#pragma once

#include <cuj/ast/context.h>
#include <cuj/ast/opr.h>
#include <cuj/ast/stat_builder.h>

CUJ_NAMESPACE_BEGIN(cuj::ast)

inline IfBuilder::~IfBuilder()
{
    CUJ_ASSERT(!then_units_.empty());

    auto entry = newRC<If>();

    auto cur_if = entry;
    for(size_t i = 0; i < then_units_.size(); ++i)
    {
        cur_if->set_cond(then_units_[i].cond);
        cur_if->set_then(then_units_[i].block);

        if(i + 1 < then_units_.size())
        {
            auto next_block = newRC<Block>();
            auto next_if = newRC<If>();
            next_block->append(next_if);

            cur_if->set_else(next_block);
            cur_if = next_if;
        }
    }

    if(else_block_)
        cur_if->set_else(else_block_);

    get_current_function()->append_statement(std::move(entry));
}

template<typename T>
IfBuilder &IfBuilder::operator+(const ArithmeticValue<T> &cond)
{
    CUJ_ASSERT(then_units_.empty() || then_units_.back().block);
    CUJ_ASSERT(!else_block_);

    if constexpr(std::is_same_v<T, bool>)
        then_units_.push_back({ cond.get_impl(), nullptr });
    else
    {
        auto cast_impl = newRC<InternalCastArithmeticValue<T, bool>>();
        cast_impl->from = cond.get_impl();
        then_units_.push_back({ std::move(cast_impl), nullptr });
    }

    return *this;
}

template<typename T>
IfBuilder &IfBuilder::operator+(const PointerImpl<T> &cond)
{
    return operator+(cond != nullptr);
}

inline IfBuilder &IfBuilder::operator+(const std::function<void()> &then_body)
{
    CUJ_ASSERT(!then_units_.empty() && !then_units_.back().block);
    CUJ_ASSERT(!else_block_);

    auto func = get_current_function();
    auto block = newRC<Block>();

    func->push_block(block);
    then_body();
    func->pop_block();

    then_units_.back().block = std::move(block);

    return *this;
}

inline IfBuilder &IfBuilder::operator-(const std::function<void()> &else_body)
{
    CUJ_ASSERT(!then_units_.empty() && then_units_.back().block);
    CUJ_ASSERT(!else_block_);

    auto func = get_current_function();
    auto block = newRC<Block>();

    func->push_block(block);
    else_body();
    func->pop_block();

    else_block_ = std::move(block);

    return *this;
}

template<typename T>
void WhileBuilder::init_cond(const ArithmeticValue<T> &cond)
{
    if constexpr(std::is_same_v<T, bool>)
        cond_ = cond.get_impl();
    else
    {
        auto cast_impl = newRC<InternalCastArithmeticValue<T, bool>>();
        cast_impl->from = cond.get_impl();
        cond_ = std::move(cast_impl);
    }
}

template<typename T>
void WhileBuilder::init_cond(const PointerImpl<T> &cond)
{
    this->init_cond(cond != nullptr);
}

template<typename T, typename>
void WhileBuilder::init_cond(T cond)
{
    this->init_cond(create_literial(cond));
}

template<typename F>
WhileBuilder::WhileBuilder(const F &calc_cond_func)
{
    auto func = get_current_function();
    auto cond_block = newRC<Block>();

    func->push_block(cond_block);
    auto cond = calc_cond_func();
    this->init_cond(std::move(cond));
    func->pop_block();

    calc_cond_ = std::move(cond_block);
}

inline WhileBuilder::~WhileBuilder()
{
    CUJ_ASSERT(calc_cond_ && cond_ && body_);
    auto while_stat = newRC<While>(
        std::move(calc_cond_), std::move(cond_), std::move(body_));
    get_current_function()->append_statement(std::move(while_stat));
}

inline void WhileBuilder::operator+(const std::function<void()> &body_func)
{
    CUJ_ASSERT(calc_cond_ && cond_ && !body_);

    auto func = get_current_function();
    auto block = newRC<Block>();

    func->push_block(block);
    body_func();
    func->pop_block();

    body_ = std::move(block);
}

inline ReturnBuilder::ReturnBuilder()
{
    auto func = get_current_function();
    if(func->get_return_type()->as<ir::BuiltinType>() != ir::BuiltinType::Void)
    {
        throw CUJException(
            "return void in a function with non-void return type");
    }

    auto ret = newRC<ReturnArithmetic<int>>(nullptr);
    func->append_statement(std::move(ret));
}

template<typename T>
ReturnBuilder::ReturnBuilder(const ArithmeticValue<T> &val)
{
    auto context = get_current_context();
    auto func = get_current_function();
    auto val_type = context->get_type<T>();

    if(val_type != func->get_return_type())
    {
#define CUJ_AUTO_CAST_RETURN(TYPE, TO) \
    case ir::BuiltinType::TYPE: (void)ReturnBuilder(cast<TO>(val)); return;

        switch(func->get_return_type()->as<ir::BuiltinType>())
        {
        case ir::BuiltinType::Void:
            throw CUJException(
                "convert value to void in return statement");
        CUJ_AUTO_CAST_RETURN(Char, char)
        CUJ_AUTO_CAST_RETURN(U8,   uint8_t)
        CUJ_AUTO_CAST_RETURN(U16,  uint16_t)
        CUJ_AUTO_CAST_RETURN(U32,  uint32_t)
        CUJ_AUTO_CAST_RETURN(U64,  uint64_t)
        CUJ_AUTO_CAST_RETURN(S8,   int8_t)
        CUJ_AUTO_CAST_RETURN(S16,  int16_t)
        CUJ_AUTO_CAST_RETURN(S32,  int32_t)
        CUJ_AUTO_CAST_RETURN(S64,  int64_t)
        CUJ_AUTO_CAST_RETURN(F32,  float)
        CUJ_AUTO_CAST_RETURN(F64,  double)
        CUJ_AUTO_CAST_RETURN(Bool, bool)
        }

#undef CUJ_AUTO_CAST_RETURN

        unreachable();
    }

    func->append_statement(newRC<ReturnArithmetic<T>>(val.get_impl()));
}

template<typename T>
ReturnBuilder::ReturnBuilder(const PointerImpl<T> &val)
{
    auto context = get_current_context();
    auto func = context->get_current_function();
    auto val_type = context->get_type<PointerImpl<T>>();
    if(val_type != func->get_return_type())
        throw CUJException("return.type != function.type");

    func->append_statement(newRC<ReturnPointer<T>>(val.get_impl()));
}

template<typename T>
ReturnBuilder::ReturnBuilder(const ClassValue<T> &val)
{
    auto context = get_current_context();
    auto func = context->get_current_function();

    auto val_type = context->get_type<T>();
    if(val_type != func->get_return_type())
        throw CUJException("return.type != function.type");

    auto ret_ptr = val.address();
    func->append_statement(newRC<ReturnClass<T>>(ret_ptr.get_impl()));
}

template<typename T, size_t N>
ReturnBuilder::ReturnBuilder(const ArrayImpl<T, N> &val)
{
    auto context = get_current_context();
    auto func = context->get_current_function();

    auto val_type = context->get_type<ArrayImpl<T, N>>();
    if(val_type != func->get_return_type())
        throw CUJException("return.type != function.type");

    auto ret_ptr = val.address();
    func->append_statement(newRC<ReturnArray<ArrayImpl<T, N>>>(ret_ptr.get_impl()));
}

template<typename T, typename>
ReturnBuilder::ReturnBuilder(T val)
{
    (void)ReturnBuilder(create_literial(val));
}

CUJ_NAMESPACE_END(cuj::ast)
