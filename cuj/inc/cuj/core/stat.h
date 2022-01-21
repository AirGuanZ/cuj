#pragma once

#include <cuj/core/expr.h>

CUJ_NAMESPACE_BEGIN(cuj::core)

struct Store;
struct Copy;
struct Block;
struct Return;
struct If;
struct Loop;
struct Break;
struct Continue;
struct Switch;
struct CallFuncStat;
struct MakeScope;
struct ExitScope;
struct InlineAsm;

using Stat = Variant<
    Store,
    Copy,
    Block,
    Return,
    If,
    Loop,
    Break,
    Continue,
    Switch,
    CallFuncStat,
    MakeScope,
    ExitScope,
    InlineAsm>;

struct Store
{
    Expr dst_addr;
    Expr val;
};

struct Copy
{
    Expr dst_addr;
    Expr src_addr;
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

struct MakeScope
{
    RC<Block> body;
};

struct ExitScope
{
    
};

struct InlineAsm
{
    std::string asm_string;
    bool side_effects;
    
    std::vector<Expr> input_values;
    std::vector<Expr> output_addresses;

    std::string input_constraints;
    std::string output_constraints;
    std::string clobber_constraints;
};

CUJ_NAMESPACE_END(cuj::core)
