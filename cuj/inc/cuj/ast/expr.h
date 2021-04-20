#pragma once

#include <memory>

#include <cuj/ir/builder.h>

CUJ_NAMESPACE_BEGIN(cuj::ast)

template<typename T>
class InternalArithmeticValue;

template<typename T>
class InternalArithmeticLeftValue;

template<typename T>
class InternalClassLeftValue;

template<typename T, size_t N>
class InternalArrayValue;

template<typename T>
class InternalPointerValue;

template<typename T>
class ArithmeticValue;

template<typename T, size_t N>
class Array;

template<typename T>
class ClassValue;

template<typename T>
class Pointer;

template<typename C>
class ClassBase;

namespace detail
{
    template<typename T, typename = void>
    struct IsDerivedFromClassBase : std::false_type { };

    template<typename T>
    struct IsDerivedFromClassBase<T, std::void_t<typename T::CUJClassFlag>>
        : std::true_type
    {
        static_assert(std::is_base_of_v<ClassBase<T>, T>);
    };

    template<typename T, typename = void>
    struct IsPointerValue : std::false_type { };

    template<typename T>
    struct IsPointerValue
        <T, std::void_t<typename T::CUJPointerTag>> : std::true_type
    {
        static_assert(std::is_same_v<T, Pointer<typename T::PointedType>>);
    };

    template<typename T, typename = void>
    struct IsIntrinsicValue : std::false_type { };

    template<typename T>
    struct IsIntrinsicValue
        <T, std::void_t<typename T::CUJBuiltinTypeTag>> : std::true_type
    {
        
    };

    template<typename T, typename = void>
    struct IsArrayValue : std::false_type { };

    template<typename T>
    struct IsArrayValue
        <T, std::void_t<typename T::ElementType>> : std::true_type
    {
        static_assert(std::is_same_v<
            T, Array<typename T::ElementType, T::ElementCount>>);
    };

    template<typename T, typename = void>
    struct CUJValueType
    {
        static_assert(
            IsPointerValue<T>  ::value ||
            IsArrayValue<T>    ::value ||
            IsIntrinsicValue<T>::value);

        using Type = T;
    };

    template<typename T>
    struct CUJValueType
        <T, std::void_t<std::enable_if_t<std::is_arithmetic_v<T>>>>
    {
        using Type = ArithmeticValue<T>;
    };

    template<typename T>
    struct CUJValueType
        <T, std::void_t<std::enable_if_t<IsDerivedFromClassBase<T>::value>>>
    {
        using Type = ClassValue<T>;
    };

} // namespace detail

template<typename T>
using Value = typename detail::CUJValueType<T>::Type;

template<typename T>
constexpr bool is_pointer = detail::IsPointerValue<T>::value;

template<typename T>
constexpr bool is_array = detail::IsArrayValue<T>::value;

template<typename T>
constexpr bool is_cuj_class = detail::IsDerivedFromClassBase<T>::value;

template<typename T>
constexpr bool is_intrinsic = detail::IsIntrinsicValue<T>::value;

template<typename T>
class InternalArithmeticValue
{
public:

    static_assert(std::is_arithmetic_v<T>);

    virtual ~InternalArithmeticValue() = default;

    virtual bool is_left() const { return false; }

    virtual RC<InternalArithmeticValue<size_t>> get_address() const;

    virtual ir::BasicValue gen_ir(ir::IRBuilder &builder) const = 0;
};

template<typename T>
class InternalArithmeticLeftValue : public InternalArithmeticValue<T>
{
public:

    RC<InternalArithmeticValue<size_t>> address;

    bool is_left() const override { return true; }

    RC<InternalArithmeticValue<size_t>> get_address() const override;

    ir::BasicValue gen_ir(ir::IRBuilder &builder) const override;
};

template<typename T>
class InternalClassLeftValue
{
public:

    static_assert(std::is_class_v<T>);

    std::unique_ptr<T> obj;

    RC<InternalArithmeticValue<size_t>> address;

    RC<InternalArithmeticValue<size_t>> get_address() const;
};

template<typename T>
class InternalPointerValue
{
public:
    
    RC<InternalArithmeticValue<size_t>> value;

    bool is_left() const { return value->is_left(); }

    RC<InternalArithmeticValue<size_t>> get_address() const;

    template<typename I, typename = std::enable_if_t<std::is_integral_v<I>>>
    RC<InternalPointerValue> offset(RC<InternalArithmeticValue<I>> index);

    ir::BasicValue gen_ir(ir::IRBuilder &builder) const;
};

template<typename T, size_t N>
class InternalArrayValue
{
public:

    RC<InternalPointerValue<T>> data_ptr;
};

template<typename T, typename I>
class InternalPointerValueOffset : public InternalArithmeticValue<size_t>
{
public:

    RC<InternalArithmeticValue<size_t>> pointer;
    RC<InternalArithmeticValue<I>>      index;

    ir::BasicValue gen_ir(ir::IRBuilder &builder) const override;
};

template<typename C>
class InternalMemberPointerValueOffset : public InternalArithmeticValue<size_t>
{
public:

    RC<InternalArithmeticValue<size_t>> class_pointer;
    int                                 member_index;

    ir::BasicValue gen_ir(ir::IRBuilder &builder) const override;
};

template<typename T>
class InternalArithmeticLoad : public InternalArithmeticValue<T>
{
public:

    RC<InternalArithmeticValue<size_t>> pointer;

    ir::BasicValue gen_ir(ir::IRBuilder &builder) const override;
};

template<typename T>
class InternalArithmeticLiterial : public InternalArithmeticValue<T>
{
public:

    T literial;

    ir::BasicValue gen_ir(ir::IRBuilder &builder) const override;
};

class InternalStackAllocationValue : public InternalArithmeticValue<size_t>
{
public:

    int alloc_index;

    ir::BasicValue gen_ir(ir::IRBuilder &builder) const override;
};

template<typename From, typename To>
class InternalCastArithmeticValue : public InternalArithmeticValue<To>
{
public:

    RC<InternalArithmeticValue<From>> from;

    ir::BasicValue gen_ir(ir::IRBuilder &builder) const override;
};

template<typename T, typename L, typename R>
class InternalBinaryOperator : public InternalArithmeticValue<T>
{
public:

    ir::BinaryOp::Type             type;
    RC<InternalArithmeticValue<L>> lhs;
    RC<InternalArithmeticValue<R>> rhs;

    ir::BasicValue gen_ir(ir::IRBuilder &builder) const override;
};

template<typename T, typename I>
class InternalUnaryOperator : public InternalArithmeticValue<T>
{
public:

    ir::UnaryOp::Type              type;
    RC<InternalArithmeticValue<I>> input;

    ir::BasicValue gen_ir(ir::IRBuilder &builder) const override;
};

template<typename T>
class ArithmeticValue
{
    RC<InternalArithmeticValue<T>> impl_;

    void init_as_stack_var();

public:

    using ArithmeticType = T;

    ArithmeticValue();

    template<typename Arg>
    ArithmeticValue(const Arg &arg);

    ArithmeticValue(const ArithmeticValue &rhs);

    ArithmeticValue(ArithmeticValue &&rhs) noexcept;

    template<typename U>
    ArithmeticValue &operator=(const U &rhs);

    ArithmeticValue &operator=(const ArithmeticValue &rhs);

    Pointer<T> address() const;

    RC<InternalArithmeticValue<T>> get_impl() const;
};

template<typename T>
class ClassValue
{
    RC<InternalClassLeftValue<T>> impl_;

public:
    
    explicit ClassValue(UninitializeFlag) { }

    explicit ClassValue(RC<InternalClassLeftValue<T>> impl);

    ClassValue(const ClassValue &rhs);
    
    ClassValue &operator=(const ClassValue &rhs);

    Pointer<T> address() const;

    RC<InternalClassLeftValue<T>> get_impl() const;

    T *operator->() const;
};

template<typename T, size_t N>
class Array
{
    RC<InternalArrayValue<T, N>> impl_;

public:

    using ElementType = T;

    static constexpr size_t ElementCount = N;

    explicit Array(UninitializeFlag) { }

    explicit Array(RC<InternalArrayValue<T, N>> impl);

    Array(const Array &rhs);

    Array &operator=(const Array &rhs);

    Pointer<Array<T, N>> address() const;

    RC<InternalArrayValue<T, N>> get_impl() const;

    template<typename I, typename = std::enable_if_t<std::is_integral_v<I>>>
    Pointer<T> get_element_ptr(const ArithmeticValue<I> &index) const;

    template<typename I, typename = std::enable_if_t<std::is_integral_v<I>>>
    Value<T> operator[](const ArithmeticValue<I> &index) const;

    template<typename I, typename = std::enable_if_t<std::is_integral_v<I>>>
    Value<T> operator[](I index) const;
};

template<typename T>
class Pointer
{
    static_assert(
        is_array<T>             ||
        is_pointer<T>           ||
        std::is_arithmetic_v<T> ||
        is_cuj_class<T>);

    RC<InternalPointerValue<T>> impl_;

public:

    struct CUJPointerTag { };

    using PointedType = T;

    explicit Pointer(UninitializeFlag) { }

    explicit Pointer(RC<InternalPointerValue<T>> impl);

    Pointer(const Pointer &other);

    Pointer &operator=(const Pointer &rhs);

    Value<T> deref() const;

    Pointer<Pointer<T>> address() const;

    RC<InternalPointerValue<T>> get_impl() const;

    template<typename I, typename = std::enable_if_t<std::is_integral_v<I>>>
    Pointer<T> offset(const ArithmeticValue<I> &index) const;

    template<typename I, typename = std::enable_if_t<std::is_integral_v<I>>>
    Value<T> operator[](const ArithmeticValue<I> &index) const;

    template<typename I, typename = std::enable_if_t<std::is_integral_v<I>>>
    Value<T> operator[](I index) const;
};

template<typename T>
std::enable_if_t<std::is_arithmetic_v<T>, ArithmeticValue<T>>
    create_literial(T val);

template<typename T, typename L, typename R>
RC<InternalArithmeticValue<T>> create_binary_operator(
    ir::BinaryOp::Type             type,
    RC<InternalArithmeticValue<L>> lhs,
    RC<InternalArithmeticValue<R>> rhs);

template<typename T, typename I>
RC<InternalArithmeticValue<T>> create_unary_operator(
    ir::UnaryOp::Type              type,
    RC<InternalArithmeticValue<I>> input);

template<typename T, typename I>
RC<InternalArithmeticValue<size_t>> create_pointer_offset(
    RC<InternalArithmeticValue<size_t>> pointer,
    RC<InternalArithmeticValue<I>>      index);

template<typename C>
RC<InternalArithmeticValue<size_t>> create_member_pointer_offset(
    RC<InternalArithmeticValue<size_t>> pointer,
    int                                 member_index);

CUJ_NAMESPACE_END(cuj::ast)
