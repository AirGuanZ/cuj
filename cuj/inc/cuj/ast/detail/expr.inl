#pragma once

#include <cuj/ast/context.h>
#include <cuj/ast/expr.h>
#include <cuj/ast/func_context.h>
#include <cuj/ast/stat.h>
#include <cuj/util/type_list.h>

#include <cuj/ast/detail/call_arg.inl>

CUJ_NAMESPACE_BEGIN(cuj::ast)

namespace detail
{

    template<typename From, typename To>
    ir::BasicValue gen_arithmetic_cast(
        ir::BasicValue input, ir::IRBuilder &builder)
    {
        if constexpr(std::is_same_v<From, To>)
            return input;

        auto to_type = get_current_context()->get_type<To>();
        auto cast_op = ir::CastOp{ ir::to_builtin_type_value<To>, input };

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
RC<InternalPointerValue<Pointer<T>>> InternalPointerValue<T>::get_address() const
{
    throw CUJException("getting address of a non-left pointer value");
}

template<typename T>
RC<InternalPointerValue<Pointer<T>>> InternalPointerLeftValue<T>::get_address() const
{
    return address;
}

template<typename T>
ir::BasicValue InternalPointerLeftValue<T>::gen_ir(ir::IRBuilder &builder) const
{
    auto addr = address->gen_ir(builder);
    auto type = get_current_context()->get_type<Pointer<T>>();
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

    auto ptr_type = get_current_context()->get_type<size_t>();
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

    auto ptr_type = get_current_context()->get_type<size_t>();
    auto ret = builder.gen_temp_value(ptr_type);

    builder.append_assign(ret, mem_ptr);
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

template<typename From, typename To>
ir::BasicValue InternalCastArithmeticValue<From, To>::gen_ir(ir::IRBuilder &builder) const
{
    auto from_val = from->gen_ir(builder);
    return detail::gen_arithmetic_cast<From, To>(from_val, builder);
}

template<typename R, typename...Args>
InternalArithmeticFunctionCall<R, Args...>::InternalArithmeticFunctionCall(
    int index, const Value<Args> &...args)
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
            typename detail::DeValueType<rm_cvref_t<decltype(arg)>>::Type>(
                builder, arg, arg_vals), ...);
    }, args);
    
    auto ret = builder.gen_temp_value(ret_type);
    builder.append_assign(
        ret, ir::CallOp{ func->get_name(), std::move(arg_vals), ret_type });

    return ret;
}

template<typename R, typename ... Args>
InternalPointerFunctionCall<R, Args...>::InternalPointerFunctionCall(
    int index, const Value<Args> &... args)
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
            typename detail::DeValueType<rm_cvref_t<decltype(arg)>>::Type>(
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
    auto lhs_val = lhs->gen_ir(builder);
    auto rhs_val = rhs->gen_ir(builder);

    if(type == ir::BinaryOp::Type::Add ||
       type == ir::BinaryOp::Type::Sub ||
       type == ir::BinaryOp::Type::Mul ||
       type == ir::BinaryOp::Type::Div)
    {
        lhs_val = detail::gen_arithmetic_cast<L, T>(lhs_val, builder);
        rhs_val = detail::gen_arithmetic_cast<R, T>(rhs_val, builder);
    }
    else if(type == ir::BinaryOp::Type::And ||
            type == ir::BinaryOp::Type::Or ||
            type == ir::BinaryOp::Type::XOr)
    {
        lhs_val = detail::gen_arithmetic_cast<L, bool>(lhs_val, builder);
        rhs_val = detail::gen_arithmetic_cast<R, bool>(rhs_val, builder);
    }
    else if constexpr(!std::is_same_v<L, bool> || !std::is_same_v<R, bool>)
    {
        using AT = decltype(std::declval<L>() + std::declval<R>());
        lhs_val = detail::gen_arithmetic_cast<L, AT>(lhs_val, builder);
        rhs_val = detail::gen_arithmetic_cast<R, AT>(rhs_val, builder);
    }

    auto binary_op = ir::BinaryOp{ type, lhs_val, rhs_val };

    auto type = get_current_context()->get_type<T>();
    auto ret = builder.gen_temp_value(type);

    builder.append_assign(ret, binary_op);
    return ret;
}

template<typename T, typename I>
ir::BasicValue InternalUnaryOperator<T, I>::gen_ir(ir::IRBuilder &builder) const
{
    auto input_val = input->gen_ir(builder);
    input_val = detail::gen_arithmetic_cast<I, T>(input_val);

    auto unary_op = ir::UnaryOp{ type, input_val };

    auto type = get_current_context()->get_type<T>();
    auto ret = builder.gen_temp_value(type);

    builder.append_assign(ret, unary_op);
    return ret;
}

template<typename T>
ArithmeticValue<T>::ArithmeticValue(RC<InternalArithmeticValue<T>> impl)
    : impl_(std::move(impl))
{
    
}

template<typename T>
template<typename I, typename>
ArithmeticValue<T>::ArithmeticValue(I val)
{
    auto literial = create_literial(val).get_impl();
    if constexpr(std::is_same_v<T, I>)
        impl_ = std::move(literial);
    else
    {
        auto cast = newRC<InternalCastArithmeticValue<I, T>>();
        cast->from = std::move(literial);
        impl_ = std::move(cast);
    }
}

template<typename T>
ArithmeticValue<T>::ArithmeticValue(const ArithmeticValue &rhs)
    : impl_(rhs.impl_)
{

}

template<typename T>
template<typename U>
ArithmeticValue<T> &ArithmeticValue<T>::operator=(const U &rhs)
{
    if constexpr(std::is_arithmetic_v<U>)
    {
        auto literial_impl = newRC<InternalArithmeticLiterial<U>>();
        literial_impl->literial = rhs;
        auto literial = ArithmeticValue<U>(std::move(literial_impl));
        *this = literial;
    }
    else
    {
        static_assert(
            std::is_same_v<ArithmeticValue<typename U::ArithmeticType>, U>);
        
        auto lhs_addr = impl_->get_address();
        auto rhs_impl = rhs.get_impl();

        get_current_function()->append_statement(
            newRC<Store<T, typename U::ArithmeticType>>(
                std::move(lhs_addr), std::move(rhs_impl)));
    }

    return *this;
}

template<typename T>
ArithmeticValue<T> &ArithmeticValue<T>::operator=(const ArithmeticValue &rhs)
{
    auto lhs_addr = impl_->get_address();
    auto rhs_impl = rhs.get_impl();

    get_current_function()->append_statement(
        newRC<Store<T, T>>(std::move(lhs_addr), std::move(rhs_impl)));

    return *this;
}

template<typename T>
Pointer<T> ArithmeticValue<T>::address() const
{
    return Pointer<T>(impl_->get_address());
}

template<typename T>
RC<InternalArithmeticValue<T>> ArithmeticValue<T>::get_impl() const
{
    return impl_;
}

template<typename T>
void ArithmeticValue<T>::set_impl(RC<InternalArithmeticValue<T>> impl)
{
    impl_ = std::move(impl);
}

template<typename T>
ClassValue<T>::ClassValue(RC<InternalClassLeftValue<T>> impl)
    : impl_(std::move(impl))
{
    
}

template<typename T>
ClassValue<T>::ClassValue(const ClassValue &rhs)
    : impl_(rhs.impl_)
{

}

template<typename T>
ClassValue<T> &ClassValue<T>::operator=(const ClassValue &rhs)
{
    *impl_->obj = *rhs.impl_->obj;
    return *this;
}

template<typename T>
Pointer<T> ClassValue<T>::address() const
{
    return Pointer<T>(impl_->get_address());
}

template<typename T>
RC<InternalClassLeftValue<T>> ClassValue<T>::get_impl() const
{
    return impl_;
}

template<typename T>
void ClassValue<T>::set_impl(RC<InternalClassLeftValue<T>> impl)
{
    impl_ = std::move(impl);
}

template<typename T>
T *ClassValue<T>::operator->() const
{
    return impl_->obj.get();
}

template<typename T, size_t N>
Array<T, N>::Array(RC<InternalArrayValue<T, N>> impl)
    : impl_(std::move(impl))
{

}

template<typename T, size_t N>
Array<T, N>::Array(const Array &rhs)
    : impl_(rhs.impl_)
{

}

template<typename T, size_t N>
Array<T, N> &Array<T, N>::operator=(const Array &rhs)
{
    for(size_t i = 0; i < impl_->elem_count; ++i)
    {
        auto literial_i = newRC<InternalArithmeticLiterial<size_t>>();
        literial_i->literial = i;
        
        auto lval_ptr = impl_->data_ptr->offset(literial_i);
        auto rval_ptr = rhs.impl_->data_ptr->offset(literial_i);

        auto lval = Pointer<T>(lval_ptr).dereference();
        auto rval = Pointer<T>(rval_ptr).dereference();

        lval = rval;
    }    

    return *this;
}

template<typename T, size_t N>
Pointer<Array<T, N>> Array<T, N>::address() const
{
    auto impl = newRC<InternalPointerValue<Array<T, N>>>();
    impl->value = impl_->data_ptr->value;
    return Pointer<Array<T, N>>(std::move(impl));
}

template<typename T, size_t N>
RC<InternalArrayValue<T, N>> Array<T, N>::get_impl() const
{
    return impl_;
}

template<typename T, size_t N>
void Array<T, N>::set_impl(RC<InternalArrayValue<T, N>> impl)
{
    impl_ = std::move(impl);
}

template<typename T, size_t N>
template<typename I, typename>
Pointer<T> Array<T, N>::get_element_ptr(const ArithmeticValue<I> &index) const
{
    return Pointer<T>(impl_->data_ptr).offset(index);
}

template<typename T, size_t N>
template<typename I, typename>
Value<T> Array<T, N>::operator[](const ArithmeticValue<I> &index) const
{
    return get_element_ptr(index).deref();
}

template<typename T, size_t N>
template<typename I, typename>
Value<T> Array<T, N>::operator[](I index) const
{
    return this->operator[](create_literial<I>(index));
}

template<typename T>
Pointer<T>::Pointer(RC<InternalPointerValue<T>> impl)
    : impl_(std::move(impl))
{

}

template<typename T>
Pointer<T>::Pointer(const Pointer &rhs)
    : impl_(rhs.impl_)
{

}

template<typename T>
Pointer<T> &Pointer<T>::operator=(const Pointer &rhs)
{
    auto lhs_addr = impl_->get_address();
    auto rhs_val  = rhs.impl_->value;

    auto store = newRC<Store<size_t, size_t>>(std::move(lhs_addr), std::move(rhs_val));
    get_current_function()->append_statement(std::move(store));

    return *this;
}

template<typename T>
Value<T> Pointer<T>::deref() const
{
    static_assert(
        is_array<T>             ||
        is_pointer<T>           ||
        std::is_arithmetic_v<T> ||
        is_cuj_class<T>);

    if constexpr(is_array<T>)
    {
        auto impl = newRC<InternalArrayValue<
            typename T::ElementType, T::ElementCount>>();
        impl->data_ptr = impl_;
        return T(std::move(impl));
    }
    else if constexpr(is_pointer<T>)
    {
        auto value = newRC<InternalArithmeticLeftValue<size_t>>();
        value->address = impl_;
        auto impl = newRC<InternalPointerValue<typename T::PointedType>>();
        impl->value = std::move(value);
        return T(std::move(impl));
    }
    else if constexpr(std::is_arithmetic_v<T>)
    {
        auto impl = newRC<InternalArithmeticLeftValue<T>>();
        impl->address = impl_;
        return ArithmeticValue<T>(std::move(impl));
    }
    else
    {
        auto addr_value = impl_;
        auto impl = newRC<InternalClassLeftValue<T>>();
        impl->address = addr_value;
        impl->obj     = newBox<T>(addr_value);
        return ClassValue<T>(std::move(impl));
    }
}

template<typename T>
Pointer<Pointer<T>> Pointer<T>::address() const
{
    auto impl = newRC<InternalPointerValue<Pointer<T>>>();
    impl->value = impl_->value->address();
    return Pointer<Pointer<T>>(std::move(impl));
}

template<typename T>
RC<InternalPointerValue<T>> Pointer<T>::get_impl() const
{
    return impl_;
}

template<typename T>
void Pointer<T>::set_impl(RC<InternalPointerValue<T>> impl)
{
    impl_ = std::move(impl);
}

template<typename T>
template<typename I, typename>
Pointer<T> Pointer<T>::offset(const ArithmeticValue<I> &index) const
{
    return Pointer<T>(create_pointer_offset(impl_, index.get_impl()));
}

template<typename T>
template<typename I, typename>
Value<T> Pointer<T>::operator[](const ArithmeticValue<I> &index) const
{
    return this->offset(index).deref();
}

template<typename T>
template<typename I, typename>
Value<T> Pointer<T>::operator[](I index) const
{
    return this->operator[](create_literial(index));
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
    ir::BinaryOp::Type             type,
    RC<InternalArithmeticValue<L>> lhs,
    RC<InternalArithmeticValue<R>> rhs)
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
