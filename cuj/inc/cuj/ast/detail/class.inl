#pragma once

#include <cuj/ast/class.h>
#include <cuj/ast/context.h>
#include <cuj/ast/expr.h>

CUJ_NAMESPACE_BEGIN(cuj::ast)

template<typename C>
template<typename T_, typename...Args>
RC<typename Value<T_>::ImplType> ClassBase<C>::new_member(Args &&...args)
{
    using T = typename detail::DeValueType<T_>::Type;

    if(type_recorder_)
    {
        CUJ_ASSERT(!ref_pointer_);
        auto context = get_current_context();
        type_recorder_->add_member(context->get_type<T>());
        return nullptr;
    }

    CUJ_ASSERT(ref_pointer_);

    auto address = create_member_pointer_offset<C, T>(
        ref_pointer_, member_count_++);

    static_assert(!is_array<T>             || sizeof...(args) == 0);
    static_assert(!is_pointer<T>           || sizeof...(args) <= 1);
    static_assert(!std::is_arithmetic_v<T> || sizeof...(args) <= 1);

    if constexpr(is_array<T>)
    {
        auto alloc_addr = newRC<InternalArrayAllocAddress<T>>();
        alloc_addr->arr_alloc = address;
        
        auto impl = newRC<InternalArrayValue<
            typename T::ElementType, T::ElementCount>>();
        impl->data_ptr = alloc_addr;

        return impl;
    }
    else if constexpr(is_pointer<T>)
    {
        auto impl_value = newRC<InternalArithmeticLeftValue<size_t>>();
        impl_value->address = std::move(address);

        auto impl = newRC<InternalPointerValue<typename T::PointedType>>();
        impl->value = std::move(impl_value);

        if constexpr(sizeof...(args) == 1)
        {
            Value<T> val(impl);
            val = (std::forward<Args>(args), ...);
        }

        return impl;
    }
    else if constexpr(std::is_arithmetic_v<T>)
    {
        auto impl = newRC<InternalArithmeticLeftValue<T>>();
        impl->address = std::move(address);

        if constexpr(sizeof...(args) == 1)
        {
            Value<T> val(impl);
            val = (std::forward<Args>(args), ...);
        }

        return impl;
    }
    else if constexpr(is_intrinsic<T>)
    {
        return T(address, std::forward<Args>(args)...);
    }
    else
    {
        auto impl = newRC<InternalClassLeftValue<T>>();
        impl->address = address;
        impl->obj     = newBox<T>(std::move(address), std::forward<Args>(args)...);
        return impl;
    }
}

template<typename C>
ClassBase<C>::ClassBase(StructTypeRecorder *type_recorder)
    : type_recorder_(type_recorder)
{

}

template<typename C>
ClassBase<C>::ClassBase(ClassAddress ref_pointer)
    : type_recorder_(nullptr), ref_pointer_(std::move(ref_pointer))
{

}

template<typename C>
ClassBase<C>::ClassBase(ClassAddress ref_pointer, UninitializeFlag)
    : ClassBase(std::move(ref_pointer))
{
    
}

template<typename C>
ClassBase<C> &ClassBase<C>::operator=(const ClassBase &)
{
    return *this;
}

CUJ_NAMESPACE_END(cuj::ast)
