#pragma once

#include <cassert>
#include <functional>
#include <map>
#include <stack>

#include <cuj/core/func.h>
#include <cuj/dsl/type_context.h>
#include <cuj/dsl/variable.h>
#include <cuj/utils/uncopyable.h>

CUJ_NAMESPACE_BEGIN(cuj::gen)

class FunctionOptimizer;

CUJ_NAMESPACE_END(cuj::gen)

CUJ_NAMESPACE_BEGIN(cuj::dsl)

template<typename T>
class Function;

class Module;

namespace function_detail
{

    template<typename T>
    struct IsFunction : std::false_type { };

    template<typename T>
    struct IsFunction<Function<T>> : std::true_type { };

    template<typename T>
    constexpr bool is_function_v = IsFunction<std::remove_reference_t<T>>::value;

    template<typename Func>
    struct AutoFunctionTrait
    {
        using T = AutoFunctionTrait<
            decltype(std::function{ std::declval<Func>() })>;
        using R = typename T::R;
        using FRaw = typename T::FRaw;
        using F = typename T::F;
    };

    template<typename Ret, typename...Args>
    struct AutoFunctionTrait<std::function<Ret(Args...)>>
    {
        using R = std::conditional_t<
            std::is_same_v<Ret, void>,
            CujVoid,
            remove_var_wrapper_t<remove_reference_t<Ret>>>;
        using FRaw = R(Args...);
        using F = Function<R(Args...)>;
    };

    template<typename T>
    struct AutoFunctionTrait<Function<T>>
    {
        using FRaw = T;
    };

    template<typename CustomRet, typename Func>
    struct AutoFunctionTraitWithCustomRet
    {
        using T = AutoFunctionTraitWithCustomRet<
            CustomRet, decltype(std::function{ std::declval<Func>() }) > ;
        using R = typename T::R;
        using FRaw = typename T::FRaw;
        using F = typename T::F;
    };

    template<typename CustomRet, typename Ret, typename...Args>
    struct AutoFunctionTraitWithCustomRet<CustomRet, std::function<Ret(Args...)>>
    {
        using R = CustomRet;
        using FRaw = R(Args...);
        using F = Function<R(Args...)>;
    };

} // namespace function_detail

class FunctionContext :
    public Uncopyable, public std::enable_shared_from_this<FunctionContext>
{
    struct Uninit { };

    explicit FunctionContext(Uninit) { }

public:

    static void push_func_context(FunctionContext *context);

    static void pop_func_context();

    static FunctionContext *get_func_context();

    FunctionContext(bool self_contained_typeset);

    void set_module(Module *mod);

    void set_name(std::string name);

    void set_type(core::Func::FuncType type);

    void add_argument(const core::Type *type, bool is_reference);

    void set_return(const core::Type *type, bool is_reference);

    void append_statement(RC<core::Stat> stat);

    void append_statement(core::Stat stat);

    void push_block(RC<core::Block> block);

    void pop_block();

    TypeContext *get_type_context();

    Module *get_module() const;

    bool is_contexted() const;

    size_t get_index_in_module() const;

    size_t alloc_local_var(const core::Type *type);

    RC<FunctionContext> clone_with_module(Module *mod);

    RC<const core::Func> get_core_func() const;

private:

    RC<core::Func>  func_;
    RC<TypeContext> type_context_;

    size_t                       index_in_module_;
    Module                      *module_;
    std::stack<RC<core::Block>>  blocks_;
};

template<typename Ret, typename...Args>
class Function<Ret(Args...)>
{
    static_assert(is_cuj_var_v<Ret> || is_cuj_ref_v<Ret>);

    static_assert(
        (((is_cuj_var_v<Args> && !std::is_same_v<Args, CujVoid>) ||
            is_cuj_ref_v<Args>) && ...));

    RC<FunctionContext> context_;

    void initialize();

    template<typename F>
    void define_impl(F &&body_func);

public:

    Function();

    template<typename F> requires !function_detail::is_function_v<F>
    Function(F &&body_func);

    void set_name(std::string name);

    void set_type(core::Func::FuncType type);

    template<typename F>
    void define(F &&body_func);

    Ret operator()(Args...args);

    RC<const FunctionContext> _get_context() const;
};

template<typename F> requires !function_detail::is_function_v<F>
Function(F)->Function<
    typename function_detail::AutoFunctionTrait<std::remove_reference_t<F>>::FRaw>;

// create regular function

template<typename F> requires !(is_cuj_ref_v<F> || is_cuj_var_v<F>)
auto function(F &&body_func)
{
    using Func = typename function_detail::AutoFunctionTrait<F>::F;
    Func ret(std::forward<F>(body_func));
    return ret;
}

template<typename R, typename F> requires is_cuj_ref_v<R> || is_cuj_var_v<R>
auto function(F &&body_func)
{
    using Func = typename function_detail::AutoFunctionTraitWithCustomRet<R, F>::F;
    Func ret(std::forward<F>(body_func));
    return ret;
}

template<typename F>
auto function(std::string name, F &&body_func)
{
    auto ret = function(std::forward<F>(body_func));
    ret.set_name(std::move(name));
    return ret;
}

template<typename R, typename F> requires is_cuj_ref_v<R> || is_cuj_var_v<R>
auto function(std::string name, F &&body_func)
{
    auto ret = function<R>(std::forward<F>(body_func));
    ret.set_name(std::move(name));
    return ret;
}

template<typename F>
auto declare(std::string name = {})
{
    using Func = typename function_detail::AutoFunctionTrait<F>::F;
    Func ret;
    if(!name.empty())
        ret.set_name(std::move(name));
    return ret;
}

// create kernel function

template<typename F>
auto kernel(F &&body_func)
{
    using Func = typename function_detail::AutoFunctionTrait<F>::F;
    Func ret(std::forward<F>(body_func));
    ret.set_type(core::Func::Kernel);
    return ret;
}

template<typename F>
auto kernel(std::string name, F &&body_func)
{
    auto ret = kernel(std::forward<F>(body_func));
    ret.set_name(std::move(name));
    return ret;
}

CUJ_NAMESPACE_END(cuj::dsl)
