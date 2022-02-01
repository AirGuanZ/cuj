#pragma once

#include <map>
#include <typeindex>
#include <vector>

#include <cuj/utils/variant.h>

CUJ_NAMESPACE_BEGIN(cuj::core)

enum class Builtin
{
    S8,
    S16,
    S32,
    S64,
    U8,
    U16,
    U32,
    U64,
    F32,
    F64,
    Char,
    Bool,
    Void
};

template<typename T> struct ArithmeticToBuiltinType;

template<> struct ArithmeticToBuiltinType<uint8_t>
{ static constexpr Builtin value = Builtin::U8; };

template<> struct ArithmeticToBuiltinType<uint16_t>
{ static constexpr Builtin value = Builtin::U16; };

template<> struct ArithmeticToBuiltinType<uint32_t>
{ static constexpr Builtin value = Builtin::U32; };

template<> struct ArithmeticToBuiltinType<uint64_t>
{ static constexpr Builtin value = Builtin::U64; };

template<> struct ArithmeticToBuiltinType<int8_t>
{ static constexpr Builtin value = Builtin::S8; };

template<> struct ArithmeticToBuiltinType<int16_t>
{ static constexpr Builtin value = Builtin::S16; };

template<> struct ArithmeticToBuiltinType<int32_t>
{ static constexpr Builtin value = Builtin::S32; };

template<> struct ArithmeticToBuiltinType<int64_t>
{ static constexpr Builtin value = Builtin::S64; };

template<> struct ArithmeticToBuiltinType<float>
{ static constexpr Builtin value = Builtin::F32; };

template<> struct ArithmeticToBuiltinType<double>
{ static constexpr Builtin value = Builtin::F64; };

template<> struct ArithmeticToBuiltinType<char>
{ static constexpr Builtin value = Builtin::Char; };

template<> struct ArithmeticToBuiltinType<bool>
{ static constexpr Builtin value = Builtin::Bool; };

template<> struct ArithmeticToBuiltinType<void>
{ static constexpr Builtin value = Builtin::Void; };

template<typename T>
constexpr Builtin arithmetic_to_builtin_v = ArithmeticToBuiltinType<T>::value;

struct Struct;
struct Array;
struct Pointer;

using Type = Variant<
    Builtin,
    Struct,
    Array,
    Pointer>;

struct Struct
{
    std::vector<const Type *> members;
    size_t alignment = 0;

    std::strong_ordering operator<=>(const Struct &rhs) const;

    bool operator==(const Struct &rhs) const;
};

struct Array
{
    const Type *element = nullptr;
    size_t      size    = 0;

    std::strong_ordering operator<=>(const Array &rhs) const;

    bool operator==(const Array &rhs) const;
};

struct Pointer
{
    const Type *pointed = nullptr;

    std::strong_ordering operator<=>(const Pointer &rhs) const;

    bool operator==(const Pointer &rhs) const;
};

struct TypeSet
{
    std::map<std::type_index, RC<Type>>     index_to_type;
    std::map<const Type *, std::type_index> type_to_index;
};

inline bool is_floating_point(Builtin builtin)
{
    return builtin == Builtin::F32 || builtin == Builtin::F64;
}

inline bool is_signed(Builtin builtin)
{
    if constexpr(std::is_signed_v<char>)
    {
        if(builtin == Builtin::Char)
            return true;
    }
    switch(builtin)
    {
        case Builtin::S8:
        case Builtin::S16:
        case Builtin::S32:
        case Builtin::S64:
        case Builtin::F32:
        case Builtin::F64:
            return true;
        default:
            return false;
    }
}

inline size_t get_custom_alignment(const Type *type)
{
    return type->match(
        [](Builtin) {  return size_t(0); },
        [](const Struct &s) { return s.alignment; },
        [](const Array &a) { return get_custom_alignment(a.element); },
        [](const Pointer &) { return size_t(0); });
}

CUJ_NAMESPACE_END(cuj::core)
