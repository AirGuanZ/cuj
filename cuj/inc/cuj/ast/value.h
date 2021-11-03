#pragma once

#include <cuj/ast/expr.h>

CUJ_NAMESPACE_BEGIN(cuj::ast)

template<typename T>
class ArithmeticVariable;

template<typename T, size_t N>
class ArrayVariable;

template<typename T>
class PointerVariable;

template<typename T>
class ArithmeticValue
{
    RC<InternalArithmeticValue<T>> impl_;

    void init_as_stack_var();

public:

    using ArithmeticType = T;

    using VariableType = ArithmeticVariable<T>;

    using ImplType = InternalArithmeticValue<T>;

    ArithmeticValue();

    explicit ArithmeticValue(UninitializeFlag) { }

    ArithmeticValue(RC<InternalArithmeticValue<T>> impl);

    template<typename U, typename = std::enable_if_t<std::is_arithmetic_v<U>>>
    ArithmeticValue(U other);
    
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

template<typename T, size_t N>
class ArrayImpl
{
    RC<InternalArrayValue<T, N>> impl_;

    void init_as_stack_var();

    template<typename I, typename = std::enable_if_t<std::is_integral_v<I>>>
    PointerImpl<T> get_element_ptr(const ArithmeticValue<I> &index) const;

public:

    using VariableType = ArrayVariable<T, N>;

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
using ArrayValue = ArrayImpl<deval_t<to_cuj_t<T>>, N>;

template<typename T, size_t N>
using Array = typename ArrayValue<T, N>::VariableType;

template<typename T>
class PointerImpl
{
    RC<InternalPointerValue<T>> impl_;

    void init_as_stack_var();

public:

    using VariableType = PointerVariable<T>;

    using ImplType = InternalPointerValue<T>;

    struct CUJPointerTag { };

    using PointedType = T;

    PointerImpl();

    PointerImpl(UninitializeFlag) { }

    PointerImpl(const std::nullptr_t &);

    PointerImpl(RC<InternalPointerValue<T>> impl);

    PointerImpl(const PointerImpl &other);

    PointerImpl &operator=(const PointerImpl &rhs);

    PointerImpl &operator=(const std::nullptr_t &);

    template<typename U, typename = std::enable_if_t<!std::is_same_v<U, void> && std::is_same_v<T, void>>>
    PointerImpl &operator=(const PointerImpl<U> &other);

    Value<T> deref() const;

    Value<T> operator*() const { return this->deref(); }

    operator PointerImpl<void>() const;

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
using PointerValue = PointerImpl<deval_t<to_cuj_t<T>>>;

template<typename T>
using Pointer = typename PointerValue<T>::VariableType;

template<typename T>
class ArithmeticVariable : public ArithmeticValue<T>
{
public:

    using ArithmeticValue<T>::ArithmeticValue;

    ArithmeticVariable(const ArithmeticValue<T> &other)
        : ArithmeticValue<T>(other)
    {
        
    }

    template<typename U>
    ArithmeticVariable &operator=(U &&other) const
    {
        ArithmeticValue<T>::operator=(std::forward<U>(other));
        return *this;
    }

    template<typename U>
    ArithmeticVariable &operator=(U &&other)
    {
        ArithmeticValue<T>::operator=(std::forward<U>(other));
        return *this;
    }
};

template<typename T>
class PointerVariable : public PointerImpl<T>
{
public:

    using PointerImpl<T>::PointerImpl;

    PointerVariable(const PointerImpl<T> &other)
        : PointerImpl<T>(other)
    {
        
    }

    template<typename U>
    PointerVariable &operator=(U &&other) const
    {
        PointerImpl<T>::operator=(std::forward<U>(other));
        return *this;
    }

    template<typename U>
    PointerVariable &operator=(U &&other)
    {
        PointerImpl<T>::operator=(std::forward<U>(other));
        return *this;
    }
};

template<typename T, size_t N>
class ArrayVariable : public ArrayImpl<T, N>
{
public:

    using ArrayImpl<T, N>::ArrayImpl;

    ArrayVariable(const ArrayImpl<T, N> &other)
        : ArrayImpl<T, N>(other)
    {
        
    }

    template<typename U>
    ArrayVariable &operator=(U &&other) const
    {
        ArrayImpl<T, N>::operator=(std::forward<U>(other));
        return *this;
    }

    template<typename U>
    ArrayVariable &operator=(U &&other)
    {
        ArrayImpl<T, N>::operator=(std::forward<U>(other));
        return *this;
    }
};

CUJ_NAMESPACE_END(cuj::ast)
