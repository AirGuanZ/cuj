#pragma once

#include <cuj/ast/class.h>
#include <cuj/ast/context.h>
#include <cuj/ast/expr.h>
#include <cuj/ast/func.h>
#include <cuj/ast/func_context.h>
#include <cuj/ast/opr.h>
#include <cuj/ast/stat.h>
#include <cuj/ast/stat_builder.h>

#include <cuj/util/macro_overloading.h>

#include <cuj/ast/detail/class.inl>
#include <cuj/ast/detail/context.inl>
#include <cuj/ast/detail/expr.inl>
#include <cuj/ast/detail/func.inl>
#include <cuj/ast/detail/func_context.inl>
#include <cuj/ast/detail/stat.inl>
#include <cuj/ast/detail/stat_builder.inl>

#define $var(TYPE, NAME, ...)                                                   \
    ::cuj::ast::Value<::cuj::ast::RawToCUJType<TYPE>> NAME =                    \
        ::cuj::ast::get_current_function()                                      \
            ->create_stack_var<::cuj::ast::RawToCUJType<TYPE>>(##__VA_ARGS__)

#define $arg(TYPE, NAME)                                                        \
    ::cuj::ast::Value<::cuj::ast::RawToCUJType<TYPE>> NAME =                    \
        ::cuj::ast::get_current_function()                                      \
            ->create_arg<::cuj::ast::RawToCUJType<TYPE>>()

#define $mem(TYPE, NAME, ...)                                                   \
    ::cuj::ast::Value<::cuj::ast::RawToCUJType<TYPE>> NAME =                    \
        new_member<::cuj::ast::RawToCUJType<TYPE>>(##__VA_ARGS__)

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

#define $return_void(DUMMY) ::cuj::ast::ReturnBuilder()
#define $return_value(DUMMY, ...) \
    do {\
        ::cuj::ast::ReturnBuilder(##__VA_ARGS__); \
    } while(false)

#define $return(...) \
    ::cuj::ast::ReturnBuilder ret_builder##__LINE__(##__VA_ARGS__)
