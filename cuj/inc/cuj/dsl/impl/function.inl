#pragma once

#include <cassert>

#include <functional>

#include <cuj/dsl/function.h>
#include <cuj/dsl/pointer.h>
#include <cuj/dsl/pointer_temp_var.h>
#include <cuj/utils/foreach_type.h>
#include <cuj/utils/scope_guard.h>

CUJ_NAMESPACE_BEGIN(cuj::dsl)

namespace function_detail
{

    template<typename T>
    struct ArgToVar;

    template<typename T> requires is_cuj_var_v<T>
    struct ArgToVar<T>
    {
        using Type = T;
    };

    template<typename T> requires is_cuj_ref_v<T>
    struct ArgToVar<T>
    {
        using Type = ptr<remove_reference_t<T>>;
    };

    template<typename T>
    using arg_to_var_t = typename ArgToVar<T>::Type;

    template<typename Arg, typename Pointer>
    Arg deref_arg_pointer(const Pointer &ptr)
    {
        if constexpr(is_cuj_ref_v<Arg>)
        {
            return Arg::_from_ptr(*ptr);
        }
        else
        {
            return Arg(*ptr);
        }
    }

    template<typename ArgTuple, typename ArgPointerTuple, int...Is>
    ArgTuple deref_arg_pointers(
        const ArgPointerTuple &ptrs, std::integer_sequence<int, Is...>)
    {
        return ArgTuple{
            deref_arg_pointer<
                std::tuple_element_t<Is, ArgTuple>,
                std::tuple_element_t<Is, ArgPointerTuple>>(
                    std::get<Is>(ptrs))...
        };
    }

} // namespace function_detail

template<typename Ret, typename ... Args>
void Function<Ret(Args ...)>::initialize()
{
    auto mod = Module::get_current_module();
    context_ = newRC<FunctionContext>(!mod);
    if(mod)
        context_->set_module(mod);

    TypeContext *type_ctx = context_->get_type_context();

    const core::Type *ret_type;
    if constexpr(is_cuj_ref_v<Ret>)
        ret_type = type_ctx->get_type<remove_reference_t<Ret>>();
    else
        ret_type = type_ctx->get_type<Ret>();
    context_->set_return(ret_type, is_cuj_ref_v<Ret>);

    auto add_arg = [&]<typename Arg>
    {
        const core::Type *arg_type;
        if constexpr(is_cuj_ref_v<Arg>)
            arg_type = type_ctx->get_type<remove_reference_t<Arg>>();
        else
            arg_type = type_ctx->get_type<Arg>();
        context_->add_argument(arg_type, is_cuj_ref_v<Arg>);
    };
    ((add_arg.template operator()<Args>()), ...);
}

template<typename Ret, typename...Args>
template<typename F>
void Function<Ret(Args...)>::define_impl(F &&body_func)
{
    using namespace function_detail;

    static_assert(
        std::is_same_v<
            std::function<Ret(Args...)>,
            decltype(std::function{ std::forward<F>(body_func) })> ||
        std::is_same_v<
            std::function<void(Args...)>,
            decltype(std::function{ std::forward<F>(body_func) })> ||
        std::is_same_v<
            std::function<add_reference_t<Ret>(Args...)>,
            decltype(std::function{ std::forward<F>(body_func) })> ||
        std::is_same_v<
            std::function<var<Ret>(Args...)>,
            decltype(std::function{ std::forward<F>(body_func) })>);

    constexpr bool is_F_ret_void = std::is_same_v<
        std::function<void(Args...)>,
        decltype(std::function{ std::forward<F>(body_func) })>;

    FunctionContext &func_ctx = *context_;
    FunctionContext::push_func_context(&func_ctx);
    CUJ_SCOPE_EXIT{ FunctionContext::pop_func_context(); };

    PointerTempVarContext ptr_temp_ctx;
    PointerTempVarContext::push_context(&ptr_temp_ctx);
    CUJ_SCOPE_EXIT{ PointerTempVarContext::pop_context(); };

    CUJ_SCOPE_SUCCESS{ func_ctx.mark_as_non_declaration(); };

    auto type_ctx = func_ctx.get_type_context();

    std::tuple<ptr<arg_to_var_t<Args>>...> arg_pointers;
    foreach_type_indexed<Args...>([&]<typename Arg, int Idx>
    {
        using Var = arg_to_var_t<Arg>;
        core::FuncArgAddr addr = {
            .addr_type = type_ctx->get_type<ptr<Var>>(),
            .arg_index = Idx
        };
        std::get<Idx>(arg_pointers) = ptr<Var>::_from_expr(addr);
    });

    std::tuple<Args...> args = deref_arg_pointers<std::tuple<Args...>>(
        arg_pointers, std::make_integer_sequence<int, sizeof...(Args)>());

    if constexpr(is_F_ret_void)
    {
        std::apply(std::forward<F>(body_func), args);
    }
    else
    {
        auto ret_tmp = std::apply(std::forward<F>(body_func), args);
        remove_var_wrapper_t<decltype(ret_tmp)> ret = ret_tmp;
        static_assert(
            std::is_same_v<decltype(ret), Ret> ||
            std::is_same_v<decltype(ret), add_reference_t<Ret>>);

        if constexpr(is_cuj_ref_v<Ret>)
        {
            core::Return ret_stat = {
                .return_type = type_ctx->get_type<arg_to_var_t<Ret>>(),
                .val         = ret.address()._load()
            };
            func_ctx.append_statement(std::move(ret_stat));
        }
        else if constexpr(is_cuj_class_v<Ret>)
        {
            auto class_type = type_ctx->get_type<Ret>();
            auto class_ptr_type = type_ctx->get_type<ptr<Ret>>();
            core::Return ret_stat = {
                .return_type = class_type,
                .val         = core::DerefClassPointer{
                    .class_ptr_type = class_ptr_type,
                    .class_ptr      = newRC<core::Expr>(ret.address()._load())
                }
            };
            func_ctx.append_statement(std::move(ret_stat));
        }
        else if constexpr(is_cuj_array_v<Ret>)
        {
            auto arr_type = type_ctx->get_type<Ret>();
            auto arr_ptr_type = type_ctx->get_type<ptr<Ret>>();
            core::Return ret_stat = {
                .return_type = arr_type,
                .val         = core::DerefArrayPointer{
                    .array_ptr_type = arr_ptr_type,
                    .array_ptr      = newRC<core::Expr>(ret.address()._load())
                }
            };
            func_ctx.append_statement(std::move(ret_stat));
        }
        else
        {
            core::Return ret_stat = {
                .return_type = type_ctx->get_type<Ret>(),
                .val         = ret._load()
            };
            func_ctx.append_statement(std::move(ret_stat));
        }
    }
}

template<typename Ret, typename...Args>
Function<Ret(Args...)>::Function()
{
    initialize();
}

template<typename Ret, typename...Args>
template<typename F> requires (!function_detail::is_function_v<F>)
Function<Ret(Args...)>::Function(F &&body_func)
{
    initialize();
    define_impl(std::forward<F>(body_func));
}

template<typename Ret, typename...Args>
void Function<Ret(Args...)>::set_name(std::string name)
{
    assert(context_);
    context_->set_name(std::move(name));
}

template<typename Ret, typename...Args>
void Function<Ret(Args...)>::set_type(core::Func::FuncType type)
{
    assert((std::is_same_v<Ret, CujVoid>) || type != core::Func::Kernel);
    context_->set_type(type);
}

template<typename Ret, typename...Args>
template<typename F>
void Function<Ret(Args...)>::define(F &&body_func)
{
    if(!context_->get_module())
    {
        throw CujException(
            "context-less function cannot be declared and defined separately");
    }
    define_impl(std::forward<F>(body_func));
}

template<typename Ret, typename...Args>
Ret Function<Ret(Args...)>::operator()(Args...args)
{
    auto func_ctx = FunctionContext::get_func_context();
    auto type_ctx = func_ctx->get_type_context();

    if(context_->is_contexted() && !func_ctx->is_contexted())
    {
        throw CujException(
            "cannot call contexted function within a context-less function");
    }

    core::CallFunc call;
    if(context_->is_contexted())
        call.contexted_func_index = context_->get_index_in_module();
    else
        call.contextless_func = context_->get_core_func();

    std::tuple<Args*...> native_args_pointers = { &args... };
    foreach_type_indexed<Args...>([&]<typename Arg, int Idx>
    {
        auto &arg = *std::get<Idx>(native_args_pointers);

        if constexpr(is_cuj_ref_v<Arg>)
        {
            call.args.push_back(newRC<core::Expr>(arg.address()._load()));
        }
        else if constexpr(is_cuj_class_v<Arg>)
        {
            core::DerefClassPointer deref_class = {
                .class_ptr_type = type_ctx->get_type<ptr<Arg>>(),
                .class_ptr      = newRC<core::Expr>(arg.address()._load())
            };
            call.args.push_back(newRC<core::Expr>(std::move(deref_class)));
        }
        else if constexpr(is_cuj_array_v<Arg>)
        {
            core::DerefArrayPointer deref_array = {
                .array_ptr_type = type_ctx->get_type<ptr<Arg>>(),
                .array_ptr      = newRC<core::Expr>(arg.address()._load())
            };
            call.args.push_back(newRC<core::Expr>(std::move(deref_array)));
        }
        else
        {
            call.args.push_back(newRC<core::Expr>(arg._load()));
        }
    });

    if constexpr(std::is_same_v<Ret, CujVoid>)
    {
        func_ctx->append_statement(core::CallFuncStat{
            .call_expr = std::move(call)
        });
        return CujVoid{};
    }
    else if constexpr(is_cuj_ref_v<Ret>)
    {
        auto p = ptr<remove_reference_t<Ret>>::_from_expr(std::move(call));
        return Ret::_from_ptr(p);
    }
    else if constexpr(is_cuj_class_v<Ret>)
    {
        core::SaveClassIntoLocalAlloc alloc = {
            .class_ptr_type = type_ctx->get_type<ptr<Ret>>(),
            .class_val      = newRC<core::Expr>(std::move(call))
        };
        return *ptr<Ret>::_from_expr(std::move(alloc));
    }
    else if constexpr(is_cuj_array_v<Ret>)
    {
        core::SaveArrayIntoLocalAlloc alloc = {
            .array_ptr_type = type_ctx->get_type<ptr<Ret>>(),
            .array_val      = newRC<core::Expr>(std::move(call))
        };
        return *ptr<Ret>::_from_expr(std::move(alloc));
    }
    else
    {
        return Ret::_from_expr(std::move(call));
    }
}

template<typename Ret, typename ... Args>
Module *Function<Ret(Args ...)>::get_module() const
{
    return context_->get_module();
}

template<typename Ret, typename ... Args>
RC<const FunctionContext> Function<Ret(Args ...)>::_get_context() const
{
    return context_;
}

CUJ_NAMESPACE_END(cuj::dsl)
