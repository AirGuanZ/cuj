#pragma once

#include <cuj/ast/context.h>
#include <cuj/ast/expr.h>
#include <cuj/ast/func_context.h>
#include <cuj/util/type_list.h>

#include <cuj/ast/detail/call_arg.inl>

CUJ_NAMESPACE_BEGIN(cuj::ast)

namespace detail
{

    template<typename From, typename To>
    ir::BasicValue gen_arithmetic_cast(
        ir::BasicValue input, ir::IRBuilder &builder)
    {
        static_assert(std::is_arithmetic_v<From>);
        static_assert(std::is_arithmetic_v<To>);

        if constexpr(std::is_same_v<From, To>)
            return input;

        auto to_type = get_current_context()->get_type<To>();
        auto cast_op = ir::CastBuiltinOp{ ir::to_builtin_type_value<To>, input };

        auto ret = builder.gen_temp_value(to_type);
        builder.append_assign(ret, cast_op);

        return ret;
    }

    template<typename From>
    ir::BasicValue gen_pointer_to_uint(
        ir::BasicValue input, ir::IRBuilder &builder)
    {
        static_assert(is_pointer<From>);

        auto ptr_type = get_current_context()->get_type<From>();
        auto to_type = get_current_context()->get_type<size_t>();

        auto cast_op = ir::PointerToUIntOp{ ptr_type, input };
        auto ret = builder.gen_temp_value(to_type);
        builder.append_assign(ret, cast_op);

        return ret;
    }

} // namespace detail

template<typename T>
RC<InternalPointerValue<T>> InternalArithmeticValue<T>::get_address() const
{
    throw CUJException("getting address of a non-left value");
}

template<typename T>
RC<InternalPointerValue<T>> InternalArithmeticLeftValue<T>::get_address() const
{
    return address;
}

template<typename T>
ir::BasicValue InternalArithmeticLeftValue<T>::gen_ir(ir::IRBuilder &builder) const
{
    auto addr = address->gen_ir(builder);
    auto type = get_current_context()->get_type<T>();
    auto load = ir::LoadOp{ type, addr };
    auto ret = builder.gen_temp_value(type);
    builder.append_assign(ret, load);
    return ret;
}

template<typename T>
RC<InternalPointerValue<T>> InternalClassLeftValue<T>::get_address() const
{
    return address;
}

template<typename T>
RC<InternalPointerValue<PointerImpl<T>>> InternalPointerValue<T>::get_address() const
{
    throw CUJException("getting address of a non-left pointer value");
}

template<typename T>
RC<InternalPointerValue<PointerImpl<T>>> InternalPointerLeftValue<T>::get_address() const
{
    return address;
}

template<typename T>
ir::BasicValue InternalPointerLeftValue<T>::gen_ir(ir::IRBuilder &builder) const
{
    auto addr = address->gen_ir(builder);
    auto type = get_current_context()->get_type<PointerImpl<T>>();
    auto load = ir::LoadOp{ type, addr };
    auto ret = builder.gen_temp_value(type);
    builder.append_assign(ret, load);
    return ret;
}

template<typename T>
ir::BasicValue InternalArrayAllocAddress<T>::gen_ir(
    ir::IRBuilder &builder) const
{
    auto ctx = get_current_context();

    auto arr = arr_alloc->gen_ir(builder);
    auto arr_type  = ctx->get_type<T>();
    auto elem_type = ctx->get_type<typename T::ElementType>();
    auto op = ir::ArrayElemAddrOp{ arr_type, elem_type, arr };

    auto ret = builder.gen_temp_value(elem_type);
    builder.append_assign(ret, op);

    return ret;
}

template<typename T, typename I>
ir::BasicValue InternalPointerValueOffset<T, I>::gen_ir(ir::IRBuilder &builder) const
{
    auto addr = pointer->gen_ir(builder);
    auto idx = index->gen_ir(builder);

    auto type = get_current_context()->get_type<T>();
    auto offset = ir::PointerOffsetOp{ type, addr, idx };

    auto ptr_type = get_current_context()->get_type<PointerImpl<T>>();
    auto ret = builder.gen_temp_value(ptr_type);

    builder.append_assign(ret, offset);
    return ret;
}

template<typename C, typename M>
ir::BasicValue InternalMemberPointerValueOffset<C, M>::gen_ir(
    ir::IRBuilder &builder) const
{
    auto addr = class_pointer->gen_ir(builder);

    auto type = get_current_context()->get_type<C>();
    auto mem_ptr = ir::MemberPtrOp{ addr, type, member_index };

    auto ptr_type = get_current_context()->get_type<PointerImpl<M>>();
    auto ret = builder.gen_temp_value(ptr_type);

    builder.append_assign(ret, mem_ptr);
    return ret;
}

template<typename T>
ir::BasicValue InternalPointerDiff<T>::gen_ir(ir::IRBuilder &builder) const
{
    auto lhs_val = lhs->gen_ir(builder);
    auto rhs_val = rhs->gen_ir(builder);

    auto ptr_type = get_current_context()->get_type<PointerImpl<T>>();
    auto ret_type = get_current_context()->get_type<int64_t>();

    auto ret = builder.gen_temp_value(ret_type);
    auto op = ir::PointerDiffOp{ ptr_type, lhs_val, rhs_val };
    builder.append_assign(ret, op);

    return ret;
}

template<typename T>
ir::BasicValue InternalArithmeticLoad<T>::gen_ir(ir::IRBuilder &builder) const
{
    auto ptr = pointer->gen_ir(builder);

    auto type = get_current_context()->get_type<T>();
    auto load = ir::LoadOp{ type, ptr };
    auto ret = builder.gen_temp_value(type);

    builder.append_assign(ret, load);
    return ret;
}

template<typename T>
ir::BasicValue InternalArithmeticLiterial<T>::gen_ir(ir::IRBuilder &builder) const
{
    return ir::BasicImmediateValue{ literial };
}

template<typename T>
ir::BasicValue InternalStackAllocationValue<T>::gen_ir(ir::IRBuilder &builder) const
{
    return ir::AllocAddress{ alloc_index };
}

template<typename T>
ir::BasicValue InternalEmptyPointer<T>::gen_ir(ir::IRBuilder &builder) const
{
    auto context = get_current_context();

    auto type = context->get_type<PointerImpl<T>>();
    auto op = ir::EmptyPointerOp{ type };
    auto ret = builder.gen_temp_value(type);

    builder.append_assign(ret, op);
    return ret;
}

template<typename From, typename To>
ir::BasicValue InternalCastArithmeticValue<From, To>::gen_ir(ir::IRBuilder &builder) const
{
    auto from_val = from->gen_ir(builder);
    return detail::gen_arithmetic_cast<From, To>(from_val, builder);
}

template<typename From, typename To>
ir::BasicValue InternalCastPointerValue<From, To>::gen_ir(ir::IRBuilder &builder) const
{
    auto context = get_current_context();
    auto from_type = context->get_type<Pointer<From>>();
    auto to_type   = context->get_type<Pointer<To>>();

    auto from_val = from->gen_ir(builder);
    auto op = ir::CastPointerOp{ from_type, to_type, from_val };

    auto ret = builder.gen_temp_value(to_type);
    builder.append_assign(ret, op);

    return ret;
}

template<typename R, typename...Args>
InternalArithmeticFunctionCall<R, Args...>::InternalArithmeticFunctionCall(
    int index, const RC<typename Value<Args>::ImplType> &...args)
    : func_index(index), args{ args... }
{
    
}

template<typename R, typename ...Args>
ir::BasicValue InternalArithmeticFunctionCall<R, Args...>::gen_ir(
    ir::IRBuilder &builder) const
{
    auto context = get_current_context();
    auto func = context->get_function_context(func_index);
    
    auto ret_type = context->get_type<R>();

    std::vector<ir::BasicValue> arg_vals;
    std::apply(
        [&](auto ...arg)
    {
        (call_detail::prepare_arg<
            typename detail::DeValueType<rm_cvref_t<Args>>::Type>(
                builder, arg, arg_vals), ...);
    }, args);
    
    auto ret = builder.gen_temp_value(ret_type);
    builder.append_assign(
        ret, ir::CallOp{ func->get_name(), std::move(arg_vals), ret_type });

    return ret;
}

template<typename R, typename ... Args>
InternalPointerFunctionCall<R, Args...>::InternalPointerFunctionCall(
    int index, const RC<typename Value<Args>::ImplType> &... args)
    : func_index(index), args{ args... }
{
    
}

template<typename R, typename ... Args>
ir::BasicValue InternalPointerFunctionCall<R, Args...>::gen_ir(
    ir::IRBuilder &builder) const
{
    static_assert(is_pointer<R>);

    auto context = get_current_context();
    auto func = context->get_function_context(func_index);

    auto ret_type = context->get_type<R>();

    std::vector<ir::BasicValue> arg_vals;
    std::apply(
        [&](const auto &...arg)
    {
        (call_detail::prepare_arg<
            typename detail::DeValueType<rm_cvref_t<Args>>::Type>(
                builder, arg, arg_vals), ...);
    }, args);

    auto ret = builder.gen_temp_value(ret_type);
    builder.append_assign(
        ret, ir::CallOp{ func->get_name(), std::move(arg_vals), ret_type });

    return ret;
}

template<typename T, typename L, typename R>
ir::BasicValue InternalBinaryOperator<T, L, R>::gen_ir(ir::IRBuilder &builder) const
{
    static_assert(std::is_arithmetic_v<L> || is_pointer<L>);
    static_assert(std::is_arithmetic_v<R> || is_pointer<R>);
    static_assert(std::is_arithmetic_v<L> == std::is_arithmetic_v<R>);

    auto lhs_val = lhs->gen_ir(builder);
    auto rhs_val = rhs->gen_ir(builder);

    if(type == ir::BinaryOp::Type::Add ||
       type == ir::BinaryOp::Type::Sub ||
       type == ir::BinaryOp::Type::Mul ||
       type == ir::BinaryOp::Type::Div)
    {
        CUJ_ASSERT(std::is_arithmetic_v<L> && std::is_arithmetic_v<R>);

        // arithmetic operators are available only to arithmetic types
        if constexpr(std::is_arithmetic_v<L> && std::is_arithmetic_v<R>)
        {
            lhs_val = detail::gen_arithmetic_cast<L, T>(lhs_val, builder);
            rhs_val = detail::gen_arithmetic_cast<R, T>(rhs_val, builder);
        }
        else
            unreachable();
    }
    else if(type == ir::BinaryOp::Type::And ||
            type == ir::BinaryOp::Type::Or  ||
            type == ir::BinaryOp::Type::XOr)
    {
        if constexpr(std::is_arithmetic_v<L>)
            lhs_val = detail::gen_arithmetic_cast<L, bool>(lhs_val, builder);
        else
        {
            lhs_val = detail::gen_pointer_to_uint<L>(lhs_val, builder);
            lhs_val = detail::gen_arithmetic_cast<size_t, bool>(lhs_val, builder);
        }

        if constexpr(std::is_arithmetic_v<R>)
            rhs_val = detail::gen_arithmetic_cast<R, bool>(rhs_val, builder);
        else
        {
            rhs_val = detail::gen_pointer_to_uint<L>(rhs_val, builder);
            rhs_val = detail::gen_arithmetic_cast<size_t, bool>(rhs_val, builder);
        }
    }
    else if constexpr(!std::is_same_v<L, bool> || !std::is_same_v<R, bool>)
    {
        CUJ_ASSERT(
            type == ir::BinaryOp::Type::Equal     ||
            type == ir::BinaryOp::Type::NotEqual  ||
            type == ir::BinaryOp::Type::Less      ||
            type == ir::BinaryOp::Type::LessEqual ||
            type == ir::BinaryOp::Type::Greater   ||
            type == ir::BinaryOp::Type::GreaterEqual);

        if constexpr(std::is_arithmetic_v<L>)
        {
            CUJ_ASSERT(std::is_arithmetic_v<R>);
            using AT = decltype(std::declval<L>() + std::declval<R>());
            lhs_val = detail::gen_arithmetic_cast<L, AT>(lhs_val, builder);
            rhs_val = detail::gen_arithmetic_cast<R, AT>(rhs_val, builder);
        }
        else
        {
            CUJ_ASSERT(is_pointer<L> && is_pointer<R>);
            lhs_val = detail::gen_pointer_to_uint<L>(lhs_val, builder);
            rhs_val = detail::gen_pointer_to_uint<L>(rhs_val, builder);
        }
    }

    auto binary_op = ir::BinaryOp{ type, lhs_val, rhs_val };

    auto ret_type = get_current_context()->get_type<T>();
    auto ret = builder.gen_temp_value(ret_type);

    builder.append_assign(ret, binary_op);
    return ret;
}

template<typename T, typename I>
ir::BasicValue InternalUnaryOperator<T, I>::gen_ir(ir::IRBuilder &builder) const
{
    auto input_val = input->gen_ir(builder);
    input_val = detail::gen_arithmetic_cast<I, T>(input_val, builder);

    auto unary_op = ir::UnaryOp{ type, input_val };

    auto type = get_current_context()->get_type<T>();
    auto ret = builder.gen_temp_value(type);

    builder.append_assign(ret, unary_op);
    return ret;
}

template<typename T>
std::enable_if_t<std::is_arithmetic_v<T>, ArithmeticValue<T>>
    create_literial(T val)
{
    auto impl = newRC<InternalArithmeticLiterial<T>>();
    impl->literial = val;
    return ArithmeticValue<T>(std::move(impl));
}

template<typename T, typename L, typename R>
RC<InternalArithmeticValue<T>> create_binary_operator(
    ir::BinaryOp::Type              type,
    RC<typename Value<L>::ImplType> lhs,
    RC<typename Value<R>::ImplType> rhs)
{
    auto ret = newRC<InternalBinaryOperator<T, L, R>>();
    ret->type = type;
    ret->lhs  = std::move(lhs);
    ret->rhs  = std::move(rhs);
    return ret;
}

template<typename T, typename I>
RC<InternalArithmeticValue<T>> create_unary_operator(
    ir::UnaryOp::Type              type,
    RC<InternalArithmeticValue<I>> input)
{
    auto ret = newRC<InternalUnaryOperator<T, I>>();
    ret->type  = type;
    ret->input = std::move(input);
    return ret;
}

template<typename T, typename I>
RC<InternalPointerValue<T>> create_pointer_offset(
    RC<InternalPointerValue<T>>    pointer,
    RC<InternalArithmeticValue<I>> index)
{
    auto ret = newRC<InternalPointerValueOffset<T, I>>();
    ret->pointer = std::move(pointer);
    ret->index   = std::move(index);
    return ret;
}

template<typename C, typename M>
RC<InternalPointerValue<M>> create_member_pointer_offset(
    RC<InternalPointerValue<C>> pointer,
    int                         member_index)
{
    auto ret = newRC<InternalMemberPointerValueOffset<C, M>>();
    ret->class_pointer = std::move(pointer);
    ret->member_index  = member_index;
    return ret;
}

CUJ_NAMESPACE_END(cuj::ast)
