#pragma once

#include <vector>

#include <cuj/util/variant.h>

CUJ_NAMESPACE_BEGIN(cuj::ir)

enum class BuiltinType
{
    U8, U16, U32, U64,
    S8, S16, S32, S64,
    F32, F64, Bool
};

namespace detail
{

#define CUJ_BT_AUX(T, V)                                                        \
    template<> struct BuiltinTypeAux<T>                                         \
    { static constexpr BuiltinType Value = BuiltinType::V; }

    template<typename T> struct BuiltinTypeAux;

    CUJ_BT_AUX(uint8_t,  U8);
    CUJ_BT_AUX(uint16_t, U16);
    CUJ_BT_AUX(uint32_t, U32);
    CUJ_BT_AUX(uint64_t, U64);
    
    CUJ_BT_AUX(int8_t,  S8);
    CUJ_BT_AUX(int16_t, S16);
    CUJ_BT_AUX(int32_t, S32);
    CUJ_BT_AUX(int64_t, S64);

    CUJ_BT_AUX(float,  F32);
    CUJ_BT_AUX(double, F64);

    CUJ_BT_AUX(bool, Bool);

#undef CUJ_BT_AUX

} // namespace detail

template<typename T>
constexpr BuiltinType to_builtin_type_value = detail::BuiltinTypeAux<T>::Value;

struct ArrayType;
struct IntrinsicType;
struct PointerType;
struct StructType;

using Type = Variant<
    BuiltinType, ArrayType, IntrinsicType, PointerType, StructType>;

struct ArrayType
{
    int         size;
    const Type *elem_type;
};

struct IntrinsicType
{
    std::string name;
};

struct PointerType
{
    const Type *pointed_type;
};

struct StructType
{
    std::string              name;
    std::vector<const Type*> mem_types;
};

CUJ_NAMESPACE_END(cuj::ir)
