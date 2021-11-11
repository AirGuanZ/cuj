#pragma once

#include <cuj/ast/context.h>
#include <cuj/ast/func.h>
#include <cuj/ast/func_context.h>

CUJ_NAMESPACE_BEGIN(cuj::ast)

namespace detail
{

    template<size_t I, typename ToArgs, typename FromArgs>
    auto convert_func_arg_type(const FromArgs &from_args)
    {
        using From = deval_t<rm_cvref_t<std::remove_pointer_t<
                        std::tuple_element_t<I, FromArgs>>>>;
        using To = std::tuple_element_t<I, ToArgs>;

        auto &from = *std::get<I>(from_args);

        if constexpr(std::is_same_v<From, To>)
            return from.get_impl();
        else if constexpr(std::is_arithmetic_v<To> && std::is_arithmetic_v<From>)
            return cast<To, From>(from).get_impl();
        else
            throw CUJException("invalid function arg type conversion");
    }

    template<typename ResultType, typename ToArgs, typename FromArgs, size_t...Is>
    auto create_call_obj(
        int index, const FromArgs &from_args, std::index_sequence<Is...>)
    {
        return newRC<ResultType>(
            index, convert_func_arg_type<Is, ToArgs>(from_args)...);
    }

    template<typename ResultType, typename RetClass,
             typename ToArgs, typename FromArgs, size_t...Is>
    auto create_call_class_obj(
        int index, const PointerImpl<RetClass> &ret_ptr,
        const FromArgs &from_args, std::index_sequence<Is...>)
    {
        return newRC<ResultType>(
            index, ret_ptr, convert_func_arg_type<Is, ToArgs>(from_args)...);
    }

} // namespace detail

template<typename Ret, typename...Args>
template<typename...CallArgs>
typename detail::FuncRetType<Ret>::Type
    FunctionImpl<Ret, Args...>::operator()(const CallArgs &...args) const
{
    auto context = get_current_context();
    auto func = context->get_current_function();
    auto args_tuple = std::make_tuple(&args...);

    using ToTuple = std::tuple<Args...>;

    if constexpr(std::is_same_v<Ret, void>)
    {
        using CallStatType = CallVoid<Args...>;
        auto stat = detail::create_call_obj<CallStatType, ToTuple>(
            index_, args_tuple, std::make_index_sequence<sizeof...(Args)>());
        func->append_statement(std::move(stat));
        return;
    }
    else if constexpr(std::is_arithmetic_v<Ret>)
    {
        using InternalCallType = InternalArithmeticFunctionCall<Ret, Args...>;
        auto call = detail::create_call_obj<InternalCallType, ToTuple>(
            index_, args_tuple, std::make_index_sequence<sizeof...(Args)>());
        
        Value<Ret> var = func->create_stack_var<Ret>();
        var = Value<Ret>(std::move(call));
        return var;
    }
    else if constexpr(is_cuj_class<Ret>)
    {
        using CallStatType = CallClass<Ret, Args...>;
        Value<Ret> var(func->create_stack_var<Ret>());
        auto call = detail::create_call_class_obj<CallStatType, Ret, ToTuple>(
            index_, var.address(), args_tuple,
            std::make_index_sequence<sizeof...(Args)>());
        func->append_statement(std::move(call));
        return var;
    }
    else if constexpr(is_array<Ret>)
    {
        using CallStatType = CallArray<Ret, Args...>;
        Value<Ret> var = func->create_stack_var<Ret>();
        auto call = detail::create_call_class_obj<CallStatType, Ret, ToTuple>(
            index_, var.address(), args_tuple,
            std::make_index_sequence<sizeof...(Args)>());
        func->append_statement(std::move(call));
        return var;
    }
    else
    {
        using InternalCallType = InternalPointerFunctionCall<Ret, Args...>;
        auto call = detail::create_call_obj<InternalCallType, ToTuple>(
            index_, args_tuple, std::make_index_sequence<sizeof...(Args)>());

        Value<Ret> var = func->create_stack_var<Ret>();
        var = Value<Ret>(std::move(call));
        return var;
    }
}

template<typename Ret, typename ... Args>
std::string FunctionImpl<Ret, Args...>::get_name() const
{
    return get_current_context()->get_function_name(index_);
}

template<typename Ret, typename ... Args>
int FunctionImpl<Ret, Args...>::get_index() const
{
    return index_;
}

template<typename Ret, typename...Args>
FunctionImpl<Ret, Args...>::FunctionImpl(int func_index)
    : index_(func_index)
{
    if(!check_return_type())
        throw CUJException("unmatched function return type");
    if(!check_param_types())
        throw CUJException("unmatched function argument type(s)");
}

template<typename Ret, typename...Args>
bool FunctionImpl<Ret, Args...>::check_return_type() const
{
    auto ctx = get_current_context();
    
    return ctx->get_function_context(index_).match(
        [&](const Box<FunctionContext> &c)
    {
        return ctx->get_type<Ret>() == c->get_return_type();
    },
        [&](const RC<ir::ImportedHostFunction> &c)
    {
        return ctx->get_type<Ret>() == c->ret_type;
    });
}

template<typename Ret, typename...Args>
bool FunctionImpl<Ret, Args...>::check_param_types() const
{
    auto ctx = get_current_context();
    
    const int arg_cnt = ctx->get_function_context(index_).match(
        [&](const Box<FunctionContext> &c)
    {
        return c->get_arg_count();
    },
        [&](const RC<ir::ImportedHostFunction> &c)
    {
        return static_cast<int>(c->arg_types.size());
    });
    if(sizeof...(Args) != arg_cnt)
        return false;

    return check_param_types_aux(std::make_index_sequence<sizeof...(Args)>());
}

template<typename Ret, typename ... Args>
template<size_t... Is>
bool FunctionImpl<Ret, Args...>::check_param_types_aux(std::index_sequence<Is...>) const
{
    using ArgTypes = std::tuple<Args...>;
    auto ctx = get_current_context();

    return ctx->get_function_context(index_).match(
        [&](const Box<FunctionContext> &c)
    {
        return ((c->get_arg_type(static_cast<int>(Is)) ==
                ctx->get_type<std::tuple_element_t<Is, ArgTypes>>()) && ...);
    },
        [&](const RC<ir::ImportedHostFunction> &c)
    {
        return ((c->arg_types[static_cast<int>(Is)] ==
                ctx->get_type<std::tuple_element_t<Is, ArgTypes>>()) && ...);
    });
}

template<typename ForcedCFunctionType, typename Ret, typename ... Args>
template<typename ... CallArgs>
typename detail::FuncRetType<typename Function<ForcedCFunctionType, Ret(Args ...)>::ReturnType>::Type
    Function<ForcedCFunctionType, Ret(Args ...)>::operator()(const CallArgs &... args) const
{
    return (*impl_)(detail::MakeArgValue<CallArgs>::process(args)...);
}

template<typename ForcedCFunctionType, typename Ret, typename ... Args>
std::string Function<ForcedCFunctionType, Ret(Args ...)>::get_name() const
{
    return impl_->get_name();
}

template<typename ForcedCFunctionType, typename Ret, typename ... Args>
int Function<ForcedCFunctionType, Ret(Args ...)>::get_index() const
{
    return impl_->get_index();
}

template<typename ForcedCFunctionType, typename Ret, typename ... Args>
template<typename Callable>
void Function<ForcedCFunctionType, Ret(Args ...)>::define(Callable &&callable)
{
    to_callable<Ret>(get_name(), std::forward<Callable>(callable));
}

template<typename ForcedCFunctionType, typename Ret, typename...Args>
void Function<ForcedCFunctionType, Ret(Args...)>::get_arg_types(std::vector<const ir::Type*> &output)
{
    auto ctx = get_current_context();
    (output.push_back(ctx->get_type<to_cuj_t<Args>>()), ...);
}

template<typename ForcedCFunctionType, typename Ret, typename ... Args>
Function<ForcedCFunctionType, Ret(Args ...)>::Function(int func_index)
{
    impl_ = std::make_unique<Impl>(func_index);
}

CUJ_NAMESPACE_END(cuj::ast)
