#pragma once

#include <cuj/dsl/function.h>
#include <cuj/dsl/return.h>

CUJ_NAMESPACE_BEGIN(cuj::dsl)

inline void _add_return_statement()
{
    auto func = FunctionContext::get_func_context();
    auto type = func->get_type_context()->get_type<CujVoid>();
    func->append_statement(core::Return{
        .return_type = type,
        .val         = {}
    });
}

template<typename T> requires is_cuj_var_v<T> || is_cuj_ref_v<T>
void _add_return_statement(const T &val)
{
    auto func = FunctionContext::get_func_context();
    auto type_ctx = func->get_type_context();
    auto &func_ret_type = func->get_return();
    if(func_ret_type.is_reference)
    {
        if constexpr(!is_cuj_ref_v<T>)
            throw CujException("returning reference of local variable");
        else
        {
            using RT = remove_reference_t<T>;
            if(!type_ctx->get_type<RT>() != func_ret_type.type)
                throw CujException("return type doesn't match");

            using PRT = ptr<RT>;
            func->append_statement(core::Return{
                .return_type = type_ctx->get_type<PRT>(),
                .val         = val.address()._load()
            });
        }
    }
    else
    {
        using RT = remove_reference_t<T>;
        if(type_ctx->get_type<RT>() != func_ret_type.type)
            throw CujException("return type doesn't match");

        if constexpr(is_cuj_class_v<RT>)
        {
            auto class_type = type_ctx->get_type<RT>();
            auto class_ptr_type = type_ctx->get_type<ptr<RT>>();
            core::Return ret_stat = {
                .return_type = class_type,
                .val         = core::DerefClassPointer{
                    .class_ptr_type = class_ptr_type,
                    .class_ptr      = newRC<core::Expr>(val.address()._load())
                }
            };
            func->append_statement(std::move(ret_stat));
        }
        else if constexpr(is_cuj_array_v<RT>)
        {
            auto arr_type = type_ctx->get_type<RT>();
            auto arr_ptr_type = type_ctx->get_type<ptr<RT>>();
            core::Return ret_stat = {
                .return_type = arr_type,
                .val         = core::DerefArrayPointer{
                    .array_ptr_type = arr_ptr_type,
                    .array_ptr      = newRC<core::Expr>(val.address()._load())
                }
            };
            func->append_statement(std::move(ret_stat));
        }
        else
        {
            core::Return ret_stat = {
                .return_type = type_ctx->get_type<RT>(),
                .val         = val._load()
            };
            func->append_statement(std::move(ret_stat));
        }
    }
}

CUJ_NAMESPACE_END(cuj::dsl)
