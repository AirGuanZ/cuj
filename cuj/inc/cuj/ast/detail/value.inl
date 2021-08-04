#pragma once

#include <cuj/ast/context.h>
#include <cuj/ast/func_context.h>
#include <cuj/ast/stat.h>
#include <cuj/ast/value.h>

#include <cuj/ast/detail/call_arg.inl>

CUJ_NAMESPACE_BEGIN(cuj::ast)

template<typename T>
void ArithmeticValue<T>::init_as_stack_var()
{
    CUJ_INTERNAL_ASSERT(!impl_);
    impl_ = get_current_function()->create_stack_var<T>();
}

template<typename T>
ArithmeticValue<T>::ArithmeticValue()
{
    init_as_stack_var();
}

template<typename T>
ArithmeticValue<T>::ArithmeticValue(RC<InternalArithmeticValue<T>> impl)
    : impl_(std::move(impl))
{
    
}

template<typename T>
template<typename U, typename>
ArithmeticValue<T>::ArithmeticValue(U other)
{
    this->init_as_stack_var();
    this->operator=(other);
}

template<typename T>
ArithmeticValue<T>::ArithmeticValue(const ArithmeticValue &other)
{
    this->init_as_stack_var();
    this->operator=(other);
    return;
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
        this->operator=(literial);
    }
    else
    {
        static_assert(
            std::is_base_of_v<ArithmeticValue<typename U::ArithmeticType>, U>);
        
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
PointerImpl<T> ArithmeticValue<T>::address() const
{
    return Variable<PointerImpl<T>>(impl_->get_address());
}

template<typename T>
RC<InternalArithmeticValue<T>> ArithmeticValue<T>::get_impl() const
{
    return impl_;
}

template<typename T>
void ArithmeticValue<T>::set_impl(const ArithmeticValue<T> &val)
{
    this->set_impl(val.get_impl());
}

template<typename T>
void ArithmeticValue<T>::set_impl(RC<InternalArithmeticValue<T>> impl)
{
    impl_ = std::move(impl);
}

template<typename T>
void ArithmeticValue<T>::swap_impl(const ArithmeticValue<T> &other) noexcept
{
    std::swap(impl_, other.impl_);
}

template<typename T>
template<typename...Args>
void ClassValue<T>::init_as_stack_var(const Args &...args)
{
    CUJ_INTERNAL_ASSERT(!impl_);
    impl_ = get_current_function()->create_stack_var<T>(args...);
}

template<typename T>
ClassValue<T>::ClassValue()
{
    init_as_stack_var();
}

template<typename T>
template<typename U, typename...Args>
ClassValue<T>::ClassValue(const U &other, const Args &...args)
{
    using RU = rm_cvref_t<U>;

    static_assert(!std::is_same_v<RU, UninitializeFlag> ||
                  sizeof...(args) == 0);
    static_assert(!std::is_convertible_v<RU, RC<InternalClassLeftValue<T>>> ||
                  sizeof...(args) == 0);

    if constexpr(std::is_same_v<RU, UninitializeFlag>)
    {
        return;
    }
    else if constexpr(std::is_convertible_v<RU, RC<InternalClassLeftValue<T>>>)
    {
        impl_ = other;
        return;
    }
    else
    {
        this->init_as_stack_var(other, args...);
        return;
    }
}

template<typename T>
ClassValue<T>::ClassValue(RC<InternalClassLeftValue<T>> impl)
    : impl_(std::move(impl))
{
    
}

template<typename T>
ClassValue<T>::ClassValue(const ClassValue &rhs)
{
    this->init_as_stack_var();
    this->operator=(rhs);
}

template<typename T>
ClassValue<T> &ClassValue<T>::operator=(const ClassValue &rhs)
{
    *impl_->obj = *rhs.impl_->obj;
    return *this;
}

template<typename T>
PointerImpl<T> ClassValue<T>::address() const
{
    return Variable<PointerImpl<T>>(impl_->get_address());
}

template<typename T>
T *ClassValue<T>::operator->() const
{
    return impl_->obj.get();
}

template<typename T>
RC<InternalClassLeftValue<T>> ClassValue<T>::get_impl() const
{
    return impl_;
}

template<typename T>
void ClassValue<T>::set_impl(const ClassValue<T> &val)
{
    this->set_impl(val.get_impl());
}

template<typename T>
void ClassValue<T>::set_impl(RC<InternalClassLeftValue<T>> impl)
{
    impl_ = std::move(impl);
}

template<typename T>
void ClassValue<T>::swap_impl(ClassValue<T> &other) noexcept
{
    std::swap(impl_, other.impl_);
}

template<typename T, size_t N>
void ArrayImpl<T, N>::init_as_stack_var()
{
    CUJ_INTERNAL_ASSERT(!impl_);
    impl_ = get_current_function()->create_stack_var<ArrayImpl<T, N>>();
}

template<typename T, size_t N>
template<typename I, typename>
PointerImpl<T> ArrayImpl<T, N>::get_element_ptr(const ArithmeticValue<I> &index) const
{
    return PointerImpl<T>(impl_->data_ptr).offset(index);
}

template<typename T, size_t N>
ArrayImpl<T, N>::ArrayImpl()
{
    init_as_stack_var();
}

template<typename T, size_t N>
template<typename U>
ArrayImpl<T, N>::ArrayImpl(const U &other)
{
    using RU = rm_cvref_t<U>;

    if constexpr(std::is_same_v<RU, UninitializeFlag>)
    {
        return;
    }
    else if constexpr(std::is_convertible_v<RU, RC<InternalArrayValue<T, N>>>)
    {
        impl_ = other;
        return;
    }
    else
    {
        assert((std::is_base_of_v<ArrayImpl, RU>));
        this->init_as_stack_var();
        this->operator=(other);
        return;
    }
}

template<typename T, size_t N>
ArrayImpl<T, N>::ArrayImpl(const ArrayImpl &other)
{
    this->init_as_stack_var();
    this->operator=(other);
}

template<typename T, size_t N>
ArrayImpl<T, N> &ArrayImpl<T, N>::operator=(const ArrayImpl &rhs)
{
    for(size_t i = 0; i < N; ++i)
        this->operator[](i) = rhs[i];

    return *this;
}

template<typename T, size_t N>
PointerImpl<ArrayImpl<T, N>> ArrayImpl<T, N>::address() const
{
    return Variable<PointerImpl<ArrayImpl<T, N>>>(impl_->data_ptr->arr_alloc);
}

template<typename T, size_t N>
constexpr size_t ArrayImpl<T, N>::size() const
{
    return N;
}

template<typename T, size_t N>
template<typename I, typename>
Value<T> ArrayImpl<T, N>::operator[](const ArithmeticValue<I> &index) const
{
    return get_element_ptr(index).deref();
}

template<typename T, size_t N>
template<typename I, typename>
Value<T> ArrayImpl<T, N>::operator[](I index) const
{
    return get_element_ptr(create_literial(index)).deref();
}

template<typename T, size_t N>
RC<InternalArrayValue<T, N>> ArrayImpl<T, N>::get_impl() const
{
    return impl_;
}

template<typename T, size_t N>
void ArrayImpl<T, N>::set_impl(const ArrayImpl<T, N> &val)
{
    this->set_impl(val.get_impl());
}

template<typename T, size_t N>
void ArrayImpl<T, N>::set_impl(RC<InternalArrayValue<T, N>> impl)
{
    impl_ = std::move(impl);
}

template<typename T, size_t N>
void ArrayImpl<T, N>::swap_impl(ArrayImpl<T, N> &other) noexcept
{
    std::swap(impl_, other.impl_);
}

template<typename T>
void PointerImpl<T>::init_as_stack_var()
{
    CUJ_INTERNAL_ASSERT(!impl_);
    impl_ = get_current_function()->create_stack_var<PointerImpl<T>>();
}

template<typename T>
PointerImpl<T>::PointerImpl()
{
    init_as_stack_var();
}

template<typename T>
PointerImpl<T>::PointerImpl(const std::nullptr_t &)
{
    init_as_stack_var();
    this->operator=(nullptr);
}

template<typename T>
PointerImpl<T>::PointerImpl(RC<InternalPointerValue<T>> impl)
    : impl_(std::move(impl))
{

}

template<typename T>
PointerImpl<T>::PointerImpl(const PointerImpl &other)
{
    init_as_stack_var();
    this->operator=(other);
}

template<typename T>
PointerImpl<T> &PointerImpl<T>::operator=(const PointerImpl &rhs)
{
    auto lhs_addr = impl_->get_address();
    auto rhs_val  = rhs.impl_;

    auto store = newRC<Store<PointerImpl<T>, PointerImpl<T>>>(std::move(lhs_addr), std::move(rhs_val));
    get_current_function()->append_statement(std::move(store));

    return *this;
}

template<typename T>
PointerImpl<T> &PointerImpl<T>::operator=(const std::nullptr_t &)
{
    this->operator=(PointerImpl<T>(newRC<InternalEmptyPointer<T>>()));
    return *this;
}

template<typename T>
template<typename U, typename>
PointerImpl<T> &PointerImpl<T>::operator=(const PointerImpl<U> &other)
{
    this->operator=(ptr_cast<void>(other));
    return *this;
}

template<typename T>
Value<T> PointerImpl<T>::deref() const
{
    static_assert(
        is_array<T>             ||
        is_pointer<T>           ||
        std::is_arithmetic_v<T> ||
        is_cuj_class<T>);

    if constexpr(is_array<T>)
    {
        auto arr_addr = newRC<InternalArrayAllocAddress<T>>();
        arr_addr->arr_alloc = impl_;

        auto impl = newRC<InternalArrayValue<
            typename T::ElementType, T::ElementCount>>();
        impl->data_ptr = arr_addr;

        return Value<T>(std::move(impl));
    }
    else if constexpr(is_pointer<T>)
    {
        auto impl = newRC<InternalPointerLeftValue<typename T::PointedType>>();
        impl->address = impl_;
        return Value<T>(std::move(impl));
    }
    else if constexpr(std::is_arithmetic_v<T>)
    {
        auto impl = newRC<InternalArithmeticLeftValue<T>>();
        impl->address = impl_;
        return Value<T>(std::move(impl));
    }
    else
    {
        auto addr_value = impl_;
        auto impl = newRC<InternalClassLeftValue<T>>();
        impl->address = addr_value;
        impl->obj     = newBox<T>(addr_value, UNINIT);
        return Value<T>(std::move(impl));
    }
}

template<typename T>
PointerImpl<T>::operator PointerImpl<void>() const
{
    return ptr_cast<void>(*this);
}

template<typename T>
PointerImpl<PointerImpl<T>> PointerImpl<T>::address() const
{
    auto left = std::dynamic_pointer_cast<InternalPointerLeftValue<T>>(impl_);
    if(!left)
        throw CUJException("getting address of a non-left pointer value");
    return Variable<PointerImpl<PointerImpl<T>>>(left->address);
}

template<typename T>
template<typename I, typename>
PointerImpl<T> PointerImpl<T>::offset(const ArithmeticValue<I> &index) const
{
    return Variable<PointerImpl<T>>(create_pointer_offset(impl_, index.get_impl()));
}

template<typename T>
template<typename I, typename>
Value<T> PointerImpl<T>::operator[](const ArithmeticValue<I> &index) const
{
    return this->offset(index).deref();
}

template<typename T>
template<typename I, typename>
Value<T> PointerImpl<T>::operator[](I index) const
{
    return this->operator[](create_literial(index));
}

template<typename T>
RC<InternalPointerValue<T>> PointerImpl<T>::get_impl() const
{
    return impl_;
}

template<typename T>
void PointerImpl<T>::set_impl(const PointerImpl<T> &val)
{
    this->set_impl(val.get_impl());
}

template<typename T>
void PointerImpl<T>::set_impl(RC<InternalPointerValue<T>> impl)
{
    impl_ = std::move(impl);
}

template<typename T>
void PointerImpl<T>::swap_impl(PointerImpl<T> &other) noexcept
{
    std::swap(impl_, other.impl_);
}

CUJ_NAMESPACE_END(cuj::ast)
