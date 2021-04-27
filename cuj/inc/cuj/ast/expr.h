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
class InternalPointerLeftValue;

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

    template<typename T>
    auto CUJValueTypeAux()
    {
        if constexpr(std::is_arithmetic_v<T>)
            return ArithmeticValue<T>(UNINIT);
        else if constexpr(IsDerivedFromClassBase<T>::value)
            return ClassValue<T>(UNINIT);
        else
            return std::declval<T>();
    }

    template<typename T>
    struct RawToCUJType
    {
        using Type = T;
    };

    template<typename T, size_t N>
    struct RawToCUJType<T[N]>
    {
        using Type = Array<typename RawToCUJType<T>::Type, N>;
    };

    template<typename T>
    struct RawToCUJType<T *>
    {
        using Type = Pointer<typename RawToCUJType<T>::Type>;
    };

} // namespace detail

template<typename T>
using RawToCUJType = typename detail::RawToCUJType<T>::Type;

template<typename T>
using Value = decltype(detail::CUJValueTypeAux<RawToCUJType<T>>());

template<typename T>
using Var = decltype(detail::CUJValueTypeAux<RawToCUJType<T>>());

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

    virtual RC<InternalPointerValue<T>> get_address() const;

    virtual ir::BasicValue gen_ir(ir::IRBuilder &builder) const = 0;
};

template<typename T>
class InternalArithmeticLeftValue : public InternalArithmeticValue<T>
{
public:
    
    RC<InternalPointerValue<T>> address;

    bool is_left() const override { return true; }

    RC<InternalPointerValue<T>> get_address() const override;

    ir::BasicValue gen_ir(ir::IRBuilder &builder) const override;
};

template<typename T>
class InternalClassLeftValue
{
public:

    static_assert(std::is_class_v<T>);

    std::unique_ptr<T> obj;

    RC<InternalPointerValue<T>> address;
    
    RC<InternalPointerValue<T>> get_address() const;
};

template<typename T>
class InternalPointerValue
{
public:

    virtual ~InternalPointerValue() = default;

    virtual bool is_left() const { return false; }

    virtual RC<InternalPointerValue<Pointer<T>>> get_address() const;

    virtual ir::BasicValue gen_ir(ir::IRBuilder &builder) const = 0;
};

template<typename T>
class InternalPointerLeftValue : public InternalPointerValue<T>
{
public:

    RC<InternalPointerValue<Pointer<T>>> address;

    bool is_left() const override { return true; }

    RC<InternalPointerValue<Pointer<T>>> get_address() const override;

    ir::BasicValue gen_ir(ir::IRBuilder &builder) const override;
};

template<typename T, size_t N>
class InternalArrayValue
{
public:

    RC<InternalPointerValue<T>> data_ptr;
};

template<typename T>
class InternalArrayAllocAddress :
    public InternalPointerValue<typename T::ElementType>
{
public:

    RC<InternalPointerValue<T>> arr_alloc;

    ir::BasicValue gen_ir(ir::IRBuilder &builder) const override;
};

template<typename T, typename I>
class InternalPointerValueOffset : public InternalPointerValue<T>
{
public:

    RC<InternalPointerValue<T>>    pointer;
    RC<InternalArithmeticValue<I>> index;

    ir::BasicValue gen_ir(ir::IRBuilder &builder) const override;
};

template<typename C, typename M>
class InternalMemberPointerValueOffset : public InternalPointerValue<M>
{
public:

    RC<InternalPointerValue<C>> class_pointer;
    int                         member_index;

    ir::BasicValue gen_ir(ir::IRBuilder &builder) const override;
};

template<typename T>
class InternalArithmeticLoad : public InternalArithmeticValue<T>
{
public:

    RC<InternalPointerValue<T>> pointer;

    ir::BasicValue gen_ir(ir::IRBuilder &builder) const override;
};

template<typename T>
class InternalArithmeticLiterial : public InternalArithmeticValue<T>
{
public:

    T literial;

    ir::BasicValue gen_ir(ir::IRBuilder &builder) const override;
};

template<typename T>
class InternalStackAllocationValue : public InternalPointerValue<T>
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

template<typename R, typename...Args>
class InternalArithmeticFunctionCall : public InternalArithmeticValue<R>
{
public:

    int                        func_index;
    std::tuple<Value<Args>...> args;

    InternalArithmeticFunctionCall(int index, const Value<Args> &...args);

    ir::BasicValue gen_ir(ir::IRBuilder &builder) const override;
};

template<typename R, typename...Args>
class InternalPointerFunctionCall :
    public InternalPointerValue<typename R::PointedType>
{
public:

    static_assert(is_pointer<R>);

    int                        func_index;
    std::tuple<Value<Args>...> args;

    InternalPointerFunctionCall(int index, const Value<Args> &...args);

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

public:

    using ArithmeticType = T;

    explicit ArithmeticValue(UninitializeFlag) { }

    explicit ArithmeticValue(RC<InternalArithmeticValue<T>> impl);

    template<typename I, typename = std::enable_if_t<std::is_arithmetic_v<I>>>
    ArithmeticValue(I val);
    
    ArithmeticValue(const ArithmeticValue &rhs);

    template<typename U>
    ArithmeticValue &operator=(const U &rhs);

    ArithmeticValue &operator=(const ArithmeticValue &rhs);

    Pointer<T> address() const;

    RC<InternalArithmeticValue<T>> get_impl() const;

    void set_impl(RC<InternalArithmeticValue<T>> impl);
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

    void set_impl(RC<InternalClassLeftValue<T>> impl);

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

    void set_impl(RC<InternalArrayValue<T, N>> impl);

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

    void set_impl(RC<InternalPointerValue<T>> impl);

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
RC<InternalPointerValue<T>> create_pointer_offset(
    RC<InternalPointerValue<T>>    pointer,
    RC<InternalArithmeticValue<I>> index);

template<typename C, typename M>
RC<InternalPointerValue<M>> create_member_pointer_offset(
    RC<InternalPointerValue<C>> pointer,
    int                         member_index);

CUJ_NAMESPACE_END(cuj::ast)
