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

        And,
        Or,
        XOr,

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

struct CastOp
{
    BuiltinType to_type;
    BasicValue  val;
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

struct PointerToUIntOp
{
    const Type *ptr_type;
    BasicValue  ptr_val;
};

using Value = Variant<
    BasicValue,
    BinaryOp,
    UnaryOp,
    LoadOp,
    CallOp,
    CastOp,
    ArrayElemAddrOp,
    IntrinsicOp,
    MemberPtrOp,
    PointerOffsetOp,
    EmptyPointerOp,
    PointerToUIntOp>;

CUJ_NAMESPACE_END(cuj::ir)
