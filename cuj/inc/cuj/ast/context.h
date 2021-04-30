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

    FunctionContext *get_function_context(int func_index);

    template<typename FuncType>
    Function<FuncType> get_function(std::string_view name) const;

    template<typename FuncType>
    Function<FuncType> get_function(int index) const;

    // function definition

    template<typename Ret, typename Callable>
    Function<FunctionType<RawToCUJType<Ret>, Callable>> add_function(
        std::string name, Callable &&callable);

    template<typename Ret, typename Callable>
    Function<FunctionType<RawToCUJType<Ret>, Callable>> add_function(
        std::string name, ir::Function::Type type, Callable &&callable);

    template<typename Ret, typename Callable>
    Function<FunctionType<RawToCUJType<Ret>, Callable>> add_function(
        Callable &&callable);

    template<typename Ret, typename Callable>
    Function<FunctionType<RawToCUJType<Ret>, Callable>> add_function(
        ir::Function::Type type, Callable &&callable);

    template<typename FuncType>
    Function<FuncType> begin_function(
        std::string        name,
        ir::Function::Type type = ir::Function::Type::Default);

    void end_function();

    // codegen

    ir::Program gen_ir() const;

    std::string gen_ir_string() const;
    
    gen::NativeJIT gen_native_jit(
        gen::NativeJIT::OptLevel opt = gen::NativeJIT::OptLevel::Default) const;
    
#if CUJ_ENABLE_CUDA
    
    std::string gen_ptx() const;

#endif

private:

    template<typename Ret, typename Callable, typename...Args, size_t...Is>
    Function<FunctionType<RawToCUJType<Ret>, Callable>> add_function_impl(
        std::string        name,
        ir::Function::Type type,
        Callable         &&callable,
        std::tuple<Args...>*,
        std::index_sequence<Is...>);

    void gen_ir_impl(ir::IRBuilder &builder) const;

    std::map<std::type_index, RC<ir::Type>> types_;
    int                                     struct_name_index_ = 0;

    std::vector<Box<FunctionContext>> funcs_;
    std::stack<FunctionContext*>      func_stack_;

    std::map<std::string, int, std::less<>> func_name_to_index_;
};

inline void push_context(Context *context);

inline void pop_context();

inline Context *get_current_context();

inline FunctionContext *get_current_function();

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

#define CUJ_SCOPED_CONTEXT(CTX_PTR)                                             \
    ::cuj::ast::push_context(CTX_PTR);                                          \
    CUJ_SCOPE_GUARD({ ::cuj::ast::pop_context(); })

CUJ_NAMESPACE_END(cuj::ast)
