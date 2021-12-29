#include <stack>

#include <cuj/core/visit.h>
#include <cuj/dsl/dsl.h>

CUJ_NAMESPACE_BEGIN(cuj::dsl)

namespace
{

    Module *&current_module()
    {
        static thread_local Module *current_module = nullptr;
        return current_module;
    }

} // namespace anonymous

void Module::set_current_module(Module *mod)
{
    current_module() = mod;
}

Module *Module::get_current_module()
{
    return current_module();
}

Module::Module()
{
    type_context_ = newRC<TypeContext>(newRC<core::TypeSet>());
}

RC<FunctionContext> Module::_get_function(size_t index)
{
    return functions_[index];
}

TypeContext *Module::_get_type_context()
{
    return type_context_.get();
}

size_t Module::_add_function(RC<FunctionContext> context)
{
    const size_t ret = functions_.size();
    functions_.push_back(std::move(context));
    return ret;
}

core::Prog Module::_generate_prog() const
{
    core::Prog ret;
    ret.global_type_set = type_context_->get_type_set();

    std::set<RC<const core::Func>> all_contextless_functions;

    std::stack<RC<const core::Func>> unprocessed_funcs;
    for(auto &f : registered_contextless_functions_)
        unprocessed_funcs.push(f->get_core_func());

    core::Visitor visitor;
    visitor.on_call_func = [&](const core::CallFunc &call_func)
    {
        if(call_func.contextless_func)
            unprocessed_funcs.push(call_func.contextless_func);
    };

    for(auto &f : functions_)
        visitor.visit(*f->get_core_func()->root_block);

    while(!unprocessed_funcs.empty())
    {
        auto func = unprocessed_funcs.top();
        unprocessed_funcs.pop();

        auto it = all_contextless_functions.find(func);
        if(it == all_contextless_functions.end())
        {
            visitor.visit(*func->root_block);
            all_contextless_functions.insert(func);
        }
    }

    for(auto &f : functions_)
        ret.funcs.push_back(f->get_core_func());
    for(auto &f : all_contextless_functions)
        ret.funcs.push_back(f);

    return ret;
}

ScopedModule::ScopedModule()
    : Module()
{
    set_current_module(this);
}

ScopedModule::~ScopedModule()
{
    assert(get_current_module() == this);
    set_current_module(nullptr);
}

CUJ_NAMESPACE_END(cuj::dsl)
