#pragma once

#include <set>

#include <cuj/core/prog.h>
#include <cuj/dsl/function.h>

CUJ_NAMESPACE_BEGIN(cuj::dsl)

class Module : public Uncopyable
{
public:

    static void set_current_module(Module *mod);

    static Module *get_current_module();

    Module();

    template<typename F>
    void register_function(const Function<F> &func);

    RC<FunctionContext> _get_function(size_t index);

    TypeContext *_get_type_context();

    size_t _add_function(RC<FunctionContext> context);

    core::Prog _generate_prog() const;

private:

    std::vector<RC<FunctionContext>>    functions_;
    std::set<RC<const FunctionContext>> registered_contextless_functions_;
    RC<TypeContext>                     type_context_;
};

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

CUJ_NAMESPACE_END(cuj::dsl)
