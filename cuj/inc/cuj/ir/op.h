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
        // ...
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
        // ...
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
    int                     func_index;
    std::vector<BasicValue> args;
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

using Value = Variant<
    BasicValue,
    BinaryOp,
    UnaryOp,
    LoadOp,
    CallOp,
    CastOp,
    IntrinsicOp,
    MemberPtrOp,
    PointerOffsetOp>;

CUJ_NAMESPACE_END(cuj::ir)
