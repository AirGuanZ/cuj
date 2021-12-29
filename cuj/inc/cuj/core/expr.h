#pragma once

#include <cuj/core/intrinsic.h>
#include <cuj/core/type.h>

CUJ_NAMESPACE_BEGIN(cuj::core)

struct Func;

struct FuncArgAddr;
struct LocalAllocAddr;
struct Load;
struct Immediate;
struct NullPtr;
struct ArithmeticCast;
struct PointerOffset;
struct ClassPointerToMemberPointer;
struct DerefClassPointer;
struct DerefArrayPointer;
struct SaveClassIntoLocalAlloc; // return its pointer
struct SaveArrayIntoLocalAlloc;
struct ArrayAddrToFirstElemAddr;
struct Binary;
struct Unary;
struct CallFunc;

using Expr = Variant<
    FuncArgAddr,
    LocalAllocAddr,
    Load,
    Immediate,
    NullPtr,
    ArithmeticCast,
    PointerOffset,
    ClassPointerToMemberPointer,
    DerefClassPointer,
    DerefArrayPointer,
    SaveClassIntoLocalAlloc,
    SaveArrayIntoLocalAlloc,
    ArrayAddrToFirstElemAddr,
    Binary,
    Unary,
    CallFunc>;

struct FuncArgAddr
{
    const Type *addr_type;
    size_t      arg_index;
};

struct LocalAllocAddr
{
    const Type *alloc_type;
    size_t      alloc_index;
};

struct Load
{
    const Type *val_type;
    RC<Expr>    src_addr;
};

struct Immediate
{
    using Value = Variant<
        uint8_t,
        uint16_t,
        uint32_t,
        uint64_t,
        int8_t,
        int16_t,
        int32_t,
        int64_t,
        float,
        double,
        char,
        bool>;

    Value value;
};

struct NullPtr
{
    const Type *ptr_type;
};

struct ArithmeticCast
{
    const Type *dst_type;
    const Type *src_type;
    RC<Expr>    src_val;
};

struct PointerOffset
{
    const Type *ptr_type;
    const Type *offset_type;
    RC<Expr>    ptr_val;
    RC<Expr>    offset_val;
    bool        negative;
};

struct ClassPointerToMemberPointer
{
    const Type *class_ptr_type;
    const Type *member_ptr_type;
    RC<Expr>    class_ptr;
    size_t      member_index;
};

struct DerefClassPointer
{
    const Type *class_ptr_type;
    RC<Expr>    class_ptr;
};

struct DerefArrayPointer
{
    const Type *array_ptr_type;
    RC<Expr>    array_ptr;
};

struct SaveClassIntoLocalAlloc
{
    const Type *class_ptr_type;
    RC<Expr>    class_val;
};

struct SaveArrayIntoLocalAlloc
{
    const Type *array_ptr_type;
    RC<Expr>    array_val;
};

struct ArrayAddrToFirstElemAddr
{
    const Type *array_ptr_type;
    RC<Expr>    array_ptr;
};

struct Binary
{
    enum class Op
    {
        Add,
        Sub,
        Mul,
        Div,
        Mod,

        Equal,
        NotEqual,
        Less,
        LessEqual,
        Greater,
        GreaterEqual,

        LeftShift,
        RightShift,

        BitwiseAnd,
        BitwiseOr,
        BitwiseXOr,
    };

    Op          op;
    RC<Expr>    lhs;
    RC<Expr>    rhs;
    const Type *lhs_type;
    const Type *rhs_type;
};

struct Unary
{
    enum class Op
    {
        Neg,
        Not,
        BitwiseNot,
    };

    Op          op;
    RC<Expr>    val;
    const Type *val_type;
};

struct CallFunc
{
    RC<const Func> contextless_func;
    size_t         contexted_func_index = 0;
    Intrinsic      intrinsic            = Intrinsic::None;

    std::vector<RC<Expr>> args;
};

CUJ_NAMESPACE_END(cuj::core)
