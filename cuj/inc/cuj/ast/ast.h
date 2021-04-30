#pragma once

#include <cuj/ast/class.h>
#include <cuj/ast/context.h>
#include <cuj/ast/context_scope.h>
#include <cuj/ast/expr.h>
#include <cuj/ast/func.h>
#include <cuj/ast/func_context.h>
#include <cuj/ast/opr.h>
#include <cuj/ast/stat.h>
#include <cuj/ast/stat_builder.h>

#include <cuj/ast/detail/class.inl>
#include <cuj/ast/detail/context.inl>
#include <cuj/ast/detail/expr.inl>
#include <cuj/ast/detail/func.inl>
#include <cuj/ast/detail/func_context.inl>
#include <cuj/ast/detail/stat.inl>
#include <cuj/ast/detail/stat_builder.inl>

CUJ_NAMESPACE_BEGIN(cuj)

using ast::Context;
using ast::ScopedContext;

using ast::Function;
using ast::FunctionContext;

using ast::ArithmeticValue;
using ast::Array;
using ast::ClassBase;
using ast::ClassValue;
using ast::Pointer;
using ast::Value;

using ast::to_callable;

using ast::get_current_context;
using ast::get_current_function;

using ast::push_context;
using ast::pop_context;

CUJ_NAMESPACE_END(cuj)

#define $int    ::cuj::ast::Value<int>
#define $uint   ::cuj::ast::Value<unsigned>
#define $float  ::cuj::ast::Value<float>
#define $double ::cuj::ast::Value<double>
#define $bool   ::cuj::ast::Value<bool>

#define $i8     ::cuj::ast::Value<int8_t>
#define $i16    ::cuj::ast::Value<int16_t>
#define $i32    ::cuj::ast::Value<int32_t>
#define $i64    ::cuj::ast::Value<int64_t>
#define $u8     ::cuj::ast::Value<uint8_t>
#define $u16    ::cuj::ast::Value<uint16_t>
#define $u32    ::cuj::ast::Value<uint32_t>
#define $u64    ::cuj::ast::Value<uint64_t>
#define $f32    ::cuj::ast::Value<float>
#define $f64    ::cuj::ast::Value<double>

#define $var(TYPE, NAME, ...)                                                   \
    ::cuj::ast::Value<::cuj::ast::RawToCUJType<TYPE>> NAME =                    \
        ::cuj::ast::get_current_function()                                      \
            ->create_stack_var<::cuj::ast::RawToCUJType<TYPE>>(__VA_ARGS__)

#define $arg(TYPE, NAME)                                                        \
    ::cuj::ast::Value<::cuj::ast::RawToCUJType<TYPE>> NAME =                    \
        ::cuj::ast::get_current_function()                                      \
            ->create_arg<::cuj::ast::RawToCUJType<TYPE>>()

#define $mem(TYPE, NAME, ...)                                                   \
    ::cuj::ast::Value<::cuj::ast::RawToCUJType<TYPE>> NAME =                    \
        new_member<::cuj::ast::RawToCUJType<TYPE>>(__VA_ARGS__)

#define $if(COND)   ::cuj::ast::IfBuilder()+(COND)+[&]
#define $elif(COND) +(COND)+[&]
#define $else       -[&]

#define $while(COND) ::cuj::ast::WhileBuilder(COND)+[&]

#define $break                                                                  \
    ::cuj::ast::get_current_function()->append_statement(                       \
        ::cuj::newRC<::cuj::ast::Break>())
#define $continue                                                               \
    ::cuj::ast::get_current_function()->append_statement(                       \
        ::cuj::newRC<::cuj::ast::Continue>())

#define $return(...)                                                            \
    do {                                                                        \
        ::cuj::ast::ReturnBuilder t(__VA_ARGS__);                               \
    } while(false)
