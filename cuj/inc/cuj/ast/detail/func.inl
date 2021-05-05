#pragma once

#include <cuj/ast/context.h>
#include <cuj/ast/func.h>
#include <cuj/ast/func_context.h>

CUJ_NAMESPACE_BEGIN(cuj::ast)

namespace detail
{

    template<typename From, typename To>
    RC<typename Value<To>::ImplType> convert_func_arg_type(
        const Value<From> &from)
    {
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
            index, convert_func_arg_type<
                        typename DeValueType<
                            rm_cvref_t<std::remove_pointer_t<
                                std::tuple_element_t<Is, FromArgs>>>>::Type,
                        std::tuple_element_t<Is, ToArgs>>(
                                *std::get<Is>(from_args))...);
    }

    template<typename ResultType, typename RetClass,
             typename ToArgs, typename FromArgs, size_t...Is>
    auto create_call_class_obj(
        int index, const PointerImpl<RetClass> &ret_ptr,
        const FromArgs &from_args, std::index_sequence<Is...>)
    {
        return newRC<ResultType>(
            index, ret_ptr,
            convert_func_arg_type<
                typename DeValueType<
                    rm_cvref_t<std::remove_pointer_t<
                        std::tuple_element_t<Is, FromArgs>>>>::Type,
                std::tuple_element_t<Is, ToArgs>>(
                    *std::get<Is>(from_args))...);
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
const std::string &FunctionImpl<Ret, Args...>::get_name() const
{
    return get_current_context()->get_function_context(index_)->get_name();
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
    auto func = ctx->get_function_context(index_);
    return ctx->get_type<Ret>() == func->get_return_type();
}

template<typename Ret, typename...Args>
bool FunctionImpl<Ret, Args...>::check_param_types() const
{
    auto ctx = get_current_context();
    auto func = ctx->get_function_context(index_);
    if(sizeof...(Args) != func->get_arg_count())
        return false;
    return check_param_types_aux(std::make_index_sequence<sizeof...(Args)>());
}

template<typename Ret, typename ... Args>
template<size_t... Is>
bool FunctionImpl<Ret, Args...>::check_param_types_aux(std::index_sequence<Is...>) const
{
    using ArgTypes = std::tuple<Args...>;

    auto ctx = get_current_context();
    auto func = ctx->get_function_context(index_);

    return
        ((func->get_arg_type(static_cast<int>(Is)) ==
          ctx->get_type<std::tuple_element_t<Is, ArgTypes>>()) && ...);
}

template<typename Ret, typename ... Args>
template<typename ... CallArgs>
typename detail::FuncRetType<typename Function<Ret(Args ...)>::ReturnType>::Type
    Function<Ret(Args ...)>::operator()(const CallArgs &... args) const
{
    return FunctionImpl<ReturnType, RawToCUJType<Args>...>::operator()(
        detail::MakeArgValue<CallArgs>::process(args)...);
}

template<typename Ret, typename...Args>
void Function<Ret(Args...)>::get_arg_types(std::vector<const ir::Type*> &output)
{
    auto ctx = get_current_context();
    (output.push_back(ctx->get_type<RawToCUJType<Args>>()), ...);
}

CUJ_NAMESPACE_END(cuj::ast)
