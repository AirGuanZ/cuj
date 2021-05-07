#pragma once

#include <cuj/ast/expr.h>

CUJ_NAMESPACE_BEGIN(cuj::ast)

template<typename T>
class ArithmeticValue
{
    RC<InternalArithmeticValue<T>> impl_;

    void init_as_stack_var();

public:

    using ArithmeticType = T;

    using ImplType = InternalArithmeticValue<T>;

    ArithmeticValue();

    template<typename U>
    ArithmeticValue(const U &other);

    ArithmeticValue(const ArithmeticValue &other);

    template<typename U>
    ArithmeticValue &operator=(const U &rhs);

    ArithmeticValue &operator=(const ArithmeticValue &rhs);

    PointerImpl<T> address() const;

    RC<InternalArithmeticValue<T>> get_impl() const;

    void set_impl(const ArithmeticValue<T> &val);

    void set_impl(RC<InternalArithmeticValue<T>> impl);

    void swap_impl(const ArithmeticValue<T> &other) noexcept;
};

template<typename T>
class ClassValue
{
    RC<InternalClassLeftValue<T>> impl_;

    template<typename...Args>
    void init_as_stack_var(const Args &...args);

public:

    using ImplType = InternalClassLeftValue<T>;

    ClassValue();

    template<typename U, typename...Args>
    explicit ClassValue(const U &other, const Args &...args);

    ClassValue(RC<InternalClassLeftValue<T>> impl);

    ClassValue(const ClassValue &rhs);
    
    ClassValue &operator=(const ClassValue &rhs);

    PointerImpl<T> address() const;

    T *operator->() const;

    RC<InternalClassLeftValue<T>> get_impl() const;

    void set_impl(const ClassValue<T> &val);

    void set_impl(RC<InternalClassLeftValue<T>> impl);

    void swap_impl(ClassValue<T> &other) noexcept;
};

template<typename T, size_t N>
class ArrayImpl
{
    RC<InternalArrayValue<T, N>> impl_;

    void init_as_stack_var();

    template<typename I, typename = std::enable_if_t<std::is_integral_v<I>>>
    PointerImpl<T> get_element_ptr(const ArithmeticValue<I> &index) const;

public:

    using ImplType = InternalArrayValue<T, N>;

    using ElementType = T;

    static constexpr size_t ElementCount = N;

    ArrayImpl();

    template<typename U>
    ArrayImpl(const U &other);

    ArrayImpl(const ArrayImpl &other);

    ArrayImpl &operator=(const ArrayImpl &rhs);

    PointerImpl<ArrayImpl<T, N>> address() const;

    constexpr size_t size() const;

    template<typename I, typename = std::enable_if_t<std::is_integral_v<I>>>
    Value<T> operator[](const ArithmeticValue<I> &index) const;

    template<typename I, typename = std::enable_if_t<std::is_integral_v<I>>>
    Value<T> operator[](I index) const;

    RC<InternalArrayValue<T, N>> get_impl() const;

    void set_impl(const ArrayImpl<T, N> &val);

    void set_impl(RC<InternalArrayValue<T, N>> impl);

    void swap_impl(ArrayImpl<T, N> &other) noexcept;
};

template<typename T, size_t N>
using Array = ArrayImpl<typename detail::DeValueType<T>::Type, N>;

template<typename T>
class PointerImpl
{
    static_assert(
        is_array<T>             ||
        is_pointer<T>           ||
        std::is_arithmetic_v<T> ||
        is_cuj_class<T>);

    RC<InternalPointerValue<T>> impl_;

    void init_as_stack_var();

public:

    using ImplType = InternalPointerValue<T>;

    struct CUJPointerTag { };

    using PointedType = T;

    PointerImpl();

    template<typename U>
    PointerImpl(const U &other);

    PointerImpl(const PointerImpl &other);

    PointerImpl &operator=(const PointerImpl &rhs);

    PointerImpl &operator=(const std::nullptr_t &);

    Value<T> deref() const;

    Value<T> operator*() const { return this->deref(); }

    PointerImpl<PointerImpl<T>> address() const;

    template<typename I, typename = std::enable_if_t<std::is_integral_v<I>>>
    PointerImpl<T> offset(const ArithmeticValue<I> &index) const;

    template<typename I, typename = std::enable_if_t<std::is_integral_v<I>>>
    Value<T> operator[](const ArithmeticValue<I> &index) const;

    template<typename I, typename = std::enable_if_t<std::is_integral_v<I>>>
    Value<T> operator[](I index) const;

    RC<InternalPointerValue<T>> get_impl() const;

    void set_impl(const PointerImpl<T> &val);

    void set_impl(RC<InternalPointerValue<T>> impl);

    void swap_impl(PointerImpl<T> &other) noexcept;
};

template<typename T>
using Pointer = PointerImpl<typename detail::DeValueType<T>::Type>;

CUJ_NAMESPACE_END(cuj::ast)
