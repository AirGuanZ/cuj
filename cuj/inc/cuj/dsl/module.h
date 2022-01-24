#pragma once

#include <set>
#include <string>

#include <cuj/core/prog.h>
#include <cuj/dsl/function.h>
#include <cuj/dsl/global_var.h>

CUJ_NAMESPACE_BEGIN(cuj::dsl)

class Module : public Uncopyable
{
public:

    static void set_current_module(Module *mod);

    static Module *get_current_module();

    Module();

    template<typename F>
    void register_function(const Function<F> &func);

    template<typename T>
    GlobalVariable<T> allocate_global_memory(std::string symbol_name = {});

    template<typename T>
    GlobalVariable<T> allocate_constant_memory(std::string symbol_name = {});

    RC<FunctionContext> _get_function(size_t index);

    TypeContext *_get_type_context();

    size_t _add_function(RC<FunctionContext> context);

    core::Prog _generate_prog() const;

private:

    std::vector<RC<FunctionContext>> functions_;
    std::set<RC<FunctionContext>>    registered_contextless_functions_;
    RC<TypeContext>                  type_context_;

    std::set<RC<core::GlobalVar>> global_vars_;
    int                           auto_global_memory_index_;
};

template<typename T>
GlobalVariable<T> allocate_global_memory(std::string symbol_name = {});

template<typename T>
GlobalVariable<T> allocate_constant_memory(std::string symbol_name = {});

class ScopedModule : public Module
{
public:

    ScopedModule();

    ~ScopedModule();
};

template<typename F>
void Module::register_function(const Function<F> &func)
{
    auto func_ctx = func._get_context();
    if(!func_ctx->is_contexted())
    {
        registered_contextless_functions_.insert(func_ctx);
        return;
    }
    if(func_ctx->get_module() != this)
        throw CujException("cannot add function from one module into another");
}

template<typename T>
GlobalVariable<T> Module::allocate_global_memory(std::string symbol_name)
{
    if(symbol_name.empty())
    {
        symbol_name = "__cuj_global_memory_"
            + std::to_string(auto_global_memory_index_++);
    }

    auto var = newRC<core::GlobalVar>();
    var->symbol_name = std::move(symbol_name);
    var->memory_type = MemoryType::Regular;
    var->type = type_context_->get_type<T>();
    global_vars_.insert(var);

    return GlobalVariable<T>(std::move(var));
}

template<typename T>
GlobalVariable<T> Module::allocate_constant_memory(std::string symbol_name)
{
    if(symbol_name.empty())
    {
        symbol_name = "__cuj_global_memory_"
            + std::to_string(auto_global_memory_index_++);
    }

    auto var = newRC<core::GlobalVar>();
    var->symbol_name = std::move(symbol_name);
    var->memory_type = MemoryType::Constant;
    var->type = type_context_->get_type<T>();
    global_vars_.insert(var);

    return GlobalVariable<T>(std::move(var));
}

template<typename T>
GlobalVariable<T> allocate_global_memory(std::string symbol_name)
{
    auto mod = Module::get_current_module();
    if(!mod)
        throw CujException("global memory must be allocared from a module");
    return mod->allocate_global_memory<T>(std::move(symbol_name));
}

template<typename T>
GlobalVariable<T> allocate_constant_memory(std::string symbol_name)
{
    auto mod = Module::get_current_module();
    if(!mod)
        throw CujException("global memory must be allocared from a module");
    return mod->allocate_constant_memory<T>(std::move(symbol_name));
}

CUJ_NAMESPACE_END(cuj::dsl)
