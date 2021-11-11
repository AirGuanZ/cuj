#pragma once

#include <map>
#include <string_view>
#include <typeindex>

#include <cuj/ast/func.h>
#include <cuj/ast/func_context.h>
#include <cuj/ast/func_trait.h>
#include <cuj/gen/native.h>
#include <cuj/ir/type.h>
#include <cuj/util/scope_guard.h>
#include <cuj/util/uncopyable.h>

CUJ_NAMESPACE_BEGIN(cuj::ast)

class Context : public Uncopyable
{
public:

    // global info retrieve

    template<typename T>
    const ir::Type *get_type();

    FunctionContext *get_current_function();
    
    const Variant<Box<FunctionContext>, RC<ir::ImportedHostFunction>> &
        get_function_context(int func_index);

    std::string get_function_name(int func_index);

    // function definition

    template<typename FuncType>
    Function<void, detail::deval_func_t<FuncType>>
        declare_function(std::string name);

    template<typename FuncType>
    Function<void, detail::deval_func_t<FuncType>>
        declare_function();

    template<typename Ret, typename Callable>
    Function<void, func_t<to_cuj_t<Ret>, Callable>> add_function(
        std::string name, Callable &&callable);

    template<typename Ret, typename Callable>
    Function<void, func_t<to_cuj_t<Ret>, Callable>> add_function(
        std::string name, ir::Function::Type type, Callable &&callable);

    template<typename Ret, typename Callable>
    Function<void, func_t<to_cuj_t<Ret>, Callable>> add_function(
        Callable &&callable);

    template<typename Ret, typename Callable>
    Function<void, func_t<to_cuj_t<Ret>, Callable>> add_function(
        ir::Function::Type type, Callable &&callable);

    template<typename FuncType>
    Function<FuncType, FuncType> import_raw_host_function(
        std::string name, uint64_t func_ptr, RC<UntypedOwner> ctx_data = {});

    template<typename FuncType>
    Function<FuncType, FuncType> import_raw_host_function(
        uint64_t func_ptr, RC<UntypedOwner> ctx_data = {});

    template<typename Ret, typename...Args>
    Function<Ret(Args...), func_trait_detail::to_cuj_func_t<Ret, Args...>>
        import_host_functor(std::function<Ret(Args...)> func);

    template<typename Ret, typename...Args>
    Function<Ret(Args...), func_trait_detail::to_cuj_func_t<Ret, Args...>>
        import_host_functor(std::string name, std::function<Ret(Args...)> func);

    template<typename Ret, typename...Args>
    Function<Ret(Args...), func_trait_detail::to_cuj_func_t<Ret, Args...>>
        import_host_function(Ret(*func_ptr)(Args...));

    template<typename Ret, typename...Args>
    Function<Ret(Args...), func_trait_detail::to_cuj_func_t<Ret, Args...>>
        import_host_function(std::string name, Ret(*func_ptr)(Args...));

    template<typename T, typename = std::enable_if_t<!std::is_pointer_v<T>>>
    auto import_host_function(T func)
    {
        return this->import_host_functor(std::function{ std::move(func) });
    }

    template<typename T, typename = std::enable_if_t<!std::is_pointer_v<T>>>
    auto import_host_function(std::string name, T func)
    {
        return this->import_host_functor(
            std::move(name), std::function{ std::move(func) });
    }

    template<typename FuncType>
    Function<void, detail::deval_func_t<FuncType>> begin_function(
        ir::Function::Type type = ir::Function::Type::Default);

    template<typename FuncType>
    Function<void, detail::deval_func_t<FuncType>> begin_function(
        std::string        name,
        ir::Function::Type type = ir::Function::Type::Default);

    void end_function();

    // codegen

    ir::Program gen_ir() const;

    std::string gen_ir_string() const;

    std::string gen_llvm_string(
        gen::LLVMIRGenerator::Target target) const;
    
    gen::NativeJIT gen_native_jit(
        gen::OptLevel opt = gen::OptLevel::Default, bool fast_math = false) const;

    std::string gen_c(bool cuda) const;
    
#if CUJ_ENABLE_CUDA
    
    std::string gen_ptx(
        gen::OptLevel opt = gen::OptLevel::Default, bool fast_math = false) const;

#endif

private:

    template<typename FuncType>
    Function<void, detail::deval_func_t<FuncType>>
        get_function(std::string_view name) const;

    template<typename FuncType>
    Function<void, detail::deval_func_t<FuncType>>
        get_function(int index) const;

    template<typename Ret, typename Callable, typename...Args, size_t...Is>
    Function<void, func_t<to_cuj_t<Ret>, Callable>> add_function_impl(
        std::string        name,
        ir::Function::Type type,
        Callable         &&callable,
        std::tuple<Args...>*,
        std::index_sequence<Is...>);

    void gen_ir_impl(ir::IRBuilder &builder) const;

    static std::map<std::type_index, RC<ir::Type>> &all_types();

    std::map<std::type_index, RC<ir::Type>> used_types_;

    using ContextFunc = Variant<Box<FunctionContext>, RC<ir::ImportedHostFunction>>;
    
    std::map<int, ContextFunc>   funcs_;
    std::stack<FunctionContext*> func_stack_;

    int                                     struct_count_ = 0;
    std::map<std::string, int, std::less<>> func_name_to_index_;
};

inline void push_context(Context *context);

inline void pop_context();

inline Context *get_current_context();

inline FunctionContext *get_current_function();

template<typename FuncType>
auto declare()
{
    return get_current_context()->declare_function<FuncType>();
}

template<typename FuncType>
auto declare(std::string name)
{
    return get_current_context()->declare_function<FuncType>(std::move(name));
}

template<typename Ret, typename Callable>
auto to_callable(Callable &&callable)
{
    return get_current_context()
        ->add_function<Ret>(std::forward<Callable>(callable));
}

template<typename Ret, typename Callable>
auto to_callable(std::string name, Callable &&callable)
{
    return get_current_context()
        ->add_function<Ret>(std::move(name), std::forward<Callable>(callable));
}

template<typename Ret, typename Callable>
auto to_callable(ir::Function::Type type, Callable &&callable)
{
    return get_current_context()
        ->add_function<Ret>(type, std::forward<Callable>(callable));
}

template<typename Ret, typename Callable>
auto to_callable(std::string name, ir::Function::Type type, Callable &&callable)
{
    return get_current_context()->add_function<Ret>(
        std::move(name), type, std::forward<Callable>(callable));
}

template<typename T>
auto import_function(T func)
{
    return get_current_context()->import_host_function(std::move(func));
}

template<typename T>
auto import_function(std::string name, T func)
{
    return get_current_context()->import_host_function(
            std::move(name), std::move(func));
}

template<typename Callable>
auto to_kernel(Callable &&callable)
{
    return to_callable<void>(
        ir::Function::Type::Kernel, std::forward<Callable>(callable));
}

template<typename Callable>
auto to_kernel(std::string name, Callable &&callable)
{
    return to_callable<void>(
        std::move(name),
        ir::Function::Type::Kernel,
        std::forward<Callable>(callable));
}

#define CUJ_SCOPED_CONTEXT(CTX_PTR)                                             \
    ::cuj::ast::push_context(CTX_PTR);                                          \
    CUJ_SCOPE_GUARD({ ::cuj::ast::pop_context(); })

CUJ_NAMESPACE_END(cuj::ast)
