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

using signed_char_t = Value<signed char>;
using short_t       = Value<short>;
using int_t         = Value<int>;
using long_t        = Value<long>;
using longlong_t    = Value<long long>;

using unsigned_char_t     = Value<unsigned char>;
using unsigned_short_t    = Value<unsigned short>;
using unsigned_int_t      = Value<unsigned int>;
using unsigned_long_t     = Value<unsigned long>;
using unsigned_longlong_t = Value<unsigned long long>;

using i8  = Value<int8_t>;
using i16 = Value<int16_t>;
using i32 = Value<int32_t>;
using i64 = Value<int64_t>;

using u8  = Value<uint8_t>;
using u16 = Value<uint16_t>;
using u32 = Value<uint32_t>;
using u64 = Value<uint64_t>;

using f32 = Value<float>;
using f64 = Value<double>;

using boolean = Value<bool>;

CUJ_NAMESPACE_END(cuj)

#define CUJ_DEFINE_CLASS(NAME)                                                  \
    using CUJClassBase = ::cuj::ast::ClassBase<NAME>;                           \
    using ClassAddress = typename CUJClassBase::ClassAddress;                   \
    using CUJClassFlag = typename CUJClassBase::CUJClassFlag;

#define $arg(TYPE, NAME)                                                        \
    ::cuj::ast::Value<::cuj::ast::RawToCUJType<TYPE>> NAME =                    \
        ::cuj::ast::get_current_function()                                      \
            ->create_arg<::cuj::ast::RawToCUJType<TYPE>>()

#define $mem(TYPE, NAME, ...)                                                   \
    ::cuj::ast::Value<::cuj::ast::RawToCUJType<TYPE>> NAME =                    \
        this->CUJClassBase::                                                    \
            template new_member<::cuj::ast::RawToCUJType<TYPE>>(__VA_ARGS__)

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
