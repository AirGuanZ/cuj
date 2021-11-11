#pragma once

#include <memory>

#include <cuj/ir/builder.h>

template<typename T>
struct CUJClassValueMap { };

#define CUJ_REGISTER_GLOBAL_CLASS(CLASS, PROXY)                                 \
    template<>                                                                  \
    struct CUJClassValueMap<CLASS>                                              \
    {                                                                           \
        using Proxy = PROXY;                                                    \
    }

CUJ_NAMESPACE_BEGIN(cuj::ast)

template<typename T>
class InternalArithmeticValue;

template<typename T>
class InternalArithmeticLeftValue;

template<typename T, size_t N>
class InternalArrayValue;

template<typename T>
class InternalPointerValue;

template<typename T>
class InternalPointerLeftValue;

template<typename T>
class InternalArrayAllocAddress;

template<typename T>
class ArithmeticValue;

template<typename T, size_t N>
class ArrayImpl;

template<typename T>
class PointerImpl;

template<typename T>
class ArithmeticVariable;

template<typename T, size_t N>
class ArrayVariable;

template<typename T>
class PointerVariable;

namespace detail
{
    template<typename T, typename = void>
    struct IsCujClass : std::false_type { };

    template<typename T>
    struct IsCujClass<T, std::void_t<typename T::CUJBase>> : std::true_type { };

    template<typename T, typename = void>
    struct IsPointerValue : std::false_type { };

    template<typename T>
    struct IsPointerValue
        <T, std::void_t<typename T::CUJPointerTag>> : std::true_type
    {
        static_assert(std::is_base_of_v<PointerImpl<typename T::PointedType>, T>);
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
            T, ArrayImpl<typename T::ElementType, T::ElementCount>>);
    };

    template<typename T>
    auto CUJValueTypeAux()
    {
        if constexpr(std::is_arithmetic_v<T>)
            return reinterpret_cast<ArithmeticValue<T>*>(0);
        else
            return reinterpret_cast<T *>(0);
    }

    template<typename T>
    T *_cuj_class_to_proxy_aux(const T *) { return nullptr; }

    template<typename T, typename = void>
    struct RawToCUJTypeClassAux
    {
        using Type = std::remove_pointer_t<decltype(
            _cuj_class_to_proxy_aux(std::declval<const T *>()))>;
    };

    template<typename T>
    struct RawToCUJTypeClassAux<
        T, std::void_t<typename CUJClassValueMap<T>::Proxy>>
    {
        using Type = typename CUJClassValueMap<T>::Proxy;
    };

    template<typename T>
    struct RawToCUJType
    {
        using Type = typename RawToCUJTypeClassAux<T>::Type;
    };

    template<typename T, size_t N>
    struct RawToCUJType<T[N]>
    {
        using Type = ArrayImpl<typename RawToCUJType<T>::Type, N>;
    };

    template<typename T>
    struct RawToCUJType<T *>
    {
        using Type = PointerImpl<typename RawToCUJType<T>::Type>;
    };
    
    template<typename T>
    struct DeValueType
    {
        using Type = T;
    };

    template<typename T>
    struct DeValueType<ArithmeticVariable<T>>
    {
        using Type = T;
    };

    template<typename T>
    struct DeValueType<PointerVariable<T>>
    {
        using Type = PointerImpl<typename DeValueType<T>::Type>;
    };

    template<typename T, size_t N>
    struct DeValueType<ArrayVariable<T, N>>
    {
        using Type = ArrayImpl<typename DeValueType<T>::Type, N>;
    };

    template<typename T>
    struct DeValueType<PointerImpl<T>>
    {
        using Type = PointerImpl<typename DeValueType<T>::Type>;
    };

    template<typename T, size_t N>
    struct DeValueType<ArrayImpl<T, N>>
    {
        using Type = ArrayImpl<typename DeValueType<T>::Type, N>;
    };

    template<typename T>
    struct DeValueType<ArithmeticValue<T>>
    {
        using Type = T;
    };

} // namespace detail

template<typename T>
using to_cuj_t = typename detail::RawToCUJType<T>::Type;

template<typename T>
using deval_t = typename detail::DeValueType<T>::Type;

template<typename T>
using Value = std::remove_pointer_t<
    decltype(detail::CUJValueTypeAux<to_cuj_t<T>>())>;

template<typename T>
using Variable = typename Value<T>::VariableType;

template<typename T>
constexpr bool is_pointer = detail::IsPointerValue<T>::value;

template<typename T>
constexpr bool is_array = detail::IsArrayValue<T>::value;

template<typename T>
constexpr bool is_cuj_class = detail::IsCujClass<T>::value;

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
class InternalPointerValue
{
public:

    virtual ~InternalPointerValue() = default;

    virtual bool is_left() const { return false; }

    virtual RC<InternalPointerValue<PointerImpl<T>>> get_address() const;

    virtual ir::BasicValue gen_ir(ir::IRBuilder &builder) const = 0;
};

template<typename T>
class InternalPointerLeftValue : public InternalPointerValue<T>
{
public:

    RC<InternalPointerValue<PointerImpl<T>>> address;

    bool is_left() const override { return true; }

    RC<InternalPointerValue<PointerImpl<T>>> get_address() const override;

    ir::BasicValue gen_ir(ir::IRBuilder &builder) const override;
};

template<typename T, size_t N>
class InternalArrayValue
{
public:

    RC<InternalArrayAllocAddress<ArrayImpl<T, N>>> data_ptr;
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
class InternalPointerToUInt : public InternalArithmeticValue<size_t>
{
public:

    RC<InternalPointerValue<T>> pointer;

    ir::BasicValue gen_ir(ir::IRBuilder &builder) const override;
};

template<typename T>
class InternalUIntToPointer : public InternalPointerValue<T>
{
public:

    RC<InternalArithmeticValue<size_t>> value;

    ir::BasicValue gen_ir(ir::IRBuilder &builder) const override;
};

template<typename T>
class InternalPointerDiff : public InternalArithmeticValue<int64_t>
{
public:

    RC<InternalPointerValue<T>> lhs;
    RC<InternalPointerValue<T>> rhs;

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

    T literal;

    ir::BasicValue gen_ir(ir::IRBuilder &builder) const override;
};

template<typename T>
class InternalStackAllocationValue : public InternalPointerValue<T>
{
public:

    int alloc_index;

    ir::BasicValue gen_ir(ir::IRBuilder &builder) const override;
};

template<typename T>
class InternalEmptyPointer : public InternalPointerValue<T>
{
public:

    ir::BasicValue gen_ir(ir::IRBuilder &builder) const override;
};

template<typename T>
class InternalConstData : public InternalPointerValue<T>
{
public:

    std::vector<unsigned char> bytes;

    ir::BasicValue gen_ir(ir::IRBuilder &builder) const override;
};

template<typename From, typename To>
class InternalCastArithmeticValue : public InternalArithmeticValue<To>
{
public:

    RC<InternalArithmeticValue<From>> from;

    ir::BasicValue gen_ir(ir::IRBuilder &builder) const override;
};

template<typename From, typename To>
class InternalCastPointerValue : public InternalPointerValue<To>
{
public:

    RC<InternalPointerValue<From>> from;

    ir::BasicValue gen_ir(ir::IRBuilder &builder) const override;
};

template<typename R, typename...Args>
class InternalArithmeticFunctionCall : public InternalArithmeticValue<R>
{
public:

    int                                               func_index;
    std::tuple<RC<typename Value<Args>::ImplType>...> args;

    InternalArithmeticFunctionCall(
        int index, const RC<typename Value<Args>::ImplType> &...args);

    ir::BasicValue gen_ir(ir::IRBuilder &builder) const override;
};

template<typename R, typename...Args>
class InternalPointerFunctionCall :
    public InternalPointerValue<typename R::PointedType>
{
public:

    static_assert(is_pointer<R>);

    int                                               func_index;
    std::tuple<RC<typename Value<Args>::ImplType>...> args;

    InternalPointerFunctionCall(
        int index, const RC<typename Value<Args>::ImplType> &...args);

    ir::BasicValue gen_ir(ir::IRBuilder &builder) const override;
};

template<typename T, typename L, typename R>
class InternalBinaryOperator : public InternalArithmeticValue<T>
{
public:

    static_assert(std::is_arithmetic_v<L> || is_pointer<L>);
    static_assert(std::is_arithmetic_v<R> || is_pointer<R>);

    ir::BinaryOp::Type              type;
    RC<typename Value<L>::ImplType> lhs;
    RC<typename Value<R>::ImplType> rhs;

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
std::enable_if_t<std::is_arithmetic_v<T>, ArithmeticValue<T>>
    create_literal(T val);

template<typename T, typename L, typename R>
RC<InternalArithmeticValue<T>> create_binary_operator(
    ir::BinaryOp::Type              type,
    RC<typename Value<L>::ImplType> lhs,
    RC<typename Value<R>::ImplType> rhs);

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
