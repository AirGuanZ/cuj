#pragma once

#include <cuj/ir/val.h>

CUJ_NAMESPACE_BEGIN(cuj::ir)
    
struct BinaryOp
{
    enum class Type
    {
        Add,
        Sub,
        Mul,
        Div,
        Mod,

        And,
        Or,

        BitwiseAnd,
        BitwiseOr,
        BitwiseXOr,

        LeftShift,
        RightShift,

        Equal,
        NotEqual,
        Less,
        LessEqual,
        Greater,
        GreaterEqual
    };

    Type       type;
    BasicValue lhs;
    BasicValue rhs;
};

struct UnaryOp
{
    enum class Type
    {
        Neg,
        Not,
        BitwiseNot,
    };

    Type       type;
    BasicValue input;
};

struct LoadOp
{
    const Type *type;
    BasicValue  src_ptr;
};

struct CallOp
{
    std::string             name;
    std::vector<BasicValue> args;
    const Type             *ret_type;
};

struct IntrinsicOp
{
    std::string             name;
    std::vector<BasicValue> args;
};

struct CastBuiltinOp
{
    BuiltinType to_type;
    BasicValue  val;
};

struct CastPointerOp
{
    const Type *from_type;
    const Type *to_type;
    BasicValue  from_val;
};

struct ArrayElemAddrOp
{
    const Type *arr_type;
    const Type *elem_type;
    BasicValue  arr_alloc;
};

struct MemberPtrOp
{
    BasicValue  ptr;
    const Type *ptr_type;
    int         member_index;
};

struct PointerOffsetOp
{
    const Type *elem_type;
    BasicValue  ptr;
    BasicValue  index;
};

struct EmptyPointerOp
{
    const Type *ptr_type;
};

struct PointerDiffOp // type: i64
{
    const Type *ptr_type;
    BasicValue  lhs;
    BasicValue  rhs;
};

struct PointerToUIntOp
{
    const Type *ptr_type;
    BasicValue  ptr_val;
};

struct UintToPointerOp
{
    const Type *ptr_type;
    BasicValue  uint_val; // tyoe: u32/u64 according to sizeof(void*)
};

using Value = Variant<
    BasicValue,
    BinaryOp,
    UnaryOp,
    LoadOp,
    CallOp,
    CastBuiltinOp,
    CastPointerOp,
    ArrayElemAddrOp,
    IntrinsicOp,
    MemberPtrOp,
    PointerOffsetOp,
    EmptyPointerOp,
    PointerDiffOp,
    PointerToUIntOp,
    UintToPointerOp>;

CUJ_NAMESPACE_END(cuj::ir)
