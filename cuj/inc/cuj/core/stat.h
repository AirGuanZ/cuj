#pragma once

#include <cuj/core/expr.h>

CUJ_NAMESPACE_BEGIN(cuj::core)

struct Store;
struct Block;
struct Return;
struct If;
struct Loop;
struct Break;
struct Continue;
struct Switch;
struct CallFuncStat;

using Stat = Variant<
    Store,
    Block,
    Return,
    If,
    Loop,
    Break,
    Continue,
    Switch,
    CallFuncStat>;

struct Store
{
    Expr dst_addr;
    Expr val;
};

struct Block
{
    std::vector<RC<Stat>> stats;
};

struct Return
{
    const Type *return_type;
    Expr        val;
};

struct If
{
    RC<Block> calc_cond;
    Expr      cond;
    RC<Stat>  then_body;
    RC<Stat>  else_body;
};

struct Loop
{
    RC<Block> body;
};

struct Break
{
    
};

struct Continue
{
    
};

struct Switch
{
    struct Branch
    {
        Immediate cond;
        RC<Block> body;
        bool      fallthrough = false;
    };

    Expr                value;
    std::vector<Branch> branches;
    RC<Block>           default_body;
};

struct CallFuncStat
{
    CallFunc call_expr;
};

CUJ_NAMESPACE_END(cuj::core)
