#pragma once

#include <cuj/ast/context.h>
#include <cuj/ast/stat_builder.h>

#define $var(TYPE, NAME, ...) \
    auto NAME = ::cuj::ast::get_current_function()->create_stack_var<TYPE>(##__VA_ARGS__)

#define $member(TYPE, NAME, ...) \
    auto NAME = new_member<TYPE>(##__VA_ARGS__)

#define $if(COND)   ::cuj::ast::IfBuilder()+(COND)+[&]
#define $elif(COND) +(COND)+[&]
#define $else       -[&]

#define $while(COND) ::cuj::ast::WhileBuilder(COND)+[&]
