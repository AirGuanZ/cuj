#pragma once

#include <cuj/ast/context.h>
#include <cuj/ast/stat_builder.h>

//#define $var ::cuj::ast::Value

#define $define_var(TYPE, NAME, ...)                                            \
    ::cuj::ast::Value<TYPE> NAME =                                              \
    ::cuj::ast::get_current_function()->create_stack_var<TYPE>(##__VA_ARGS__)

#define $define_arg(TYPE, NAME)                                                 \
    ::cuj::ast::Value<TYPE> NAME =                                              \
    ::cuj::ast::get_current_function()->create_arg<TYPE>()

#define $member(TYPE, NAME, ...)                                                \
    ::cuj::ast::Value<TYPE> NAME = new_member<TYPE>(##__VA_ARGS__)

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
