#include <cassert>
#include <string>

#include <cuj/dsl/dsl.h>

CUJ_NAMESPACE_BEGIN(cuj::dsl)

namespace
{

    auto &get_func_ctcs_per_thread()
    {
        static thread_local std::stack<FunctionContext *> func_ctxs_per_thread;
        return func_ctxs_per_thread;
    }

} // namespace anonymous

void FunctionContext::push_func_context(FunctionContext *context)
{
    get_func_ctcs_per_thread().push(context);
}

void FunctionContext::pop_func_context()
{
    assert(!get_func_ctcs_per_thread().empty());
    get_func_ctcs_per_thread().pop();
}

FunctionContext *FunctionContext::get_func_context()
{
    assert(!get_func_ctcs_per_thread().empty());
    return get_func_ctcs_per_thread().top();
}

FunctionContext::FunctionContext(bool self_contained_typeset)
{
    static std::atomic<uint64_t> auto_func_name_index = 0;

    index_in_module_ = 0;
    module_ = nullptr;

    func_ = newRC<core::Func>();
    func_->name =
        "__cuj_auto_function_name_" + std::to_string(auto_func_name_index++);
    func_->root_block = newRC<core::Block>();

    if(self_contained_typeset)
    {
        func_->type_set = newRC<core::TypeSet>();
        type_context_ = newRC<TypeContext>(func_->type_set);
    }

    blocks_.push(func_->root_block);
}

void FunctionContext::set_module(Module *mod)
{
    module_ = mod;
    index_in_module_ = mod->_add_function(shared_from_this());
}

void FunctionContext::set_name(std::string name)
{
    func_->name = std::move(name);
}

void FunctionContext::set_type(core::Func::FuncType type)
{
    func_->type = type;
}

void FunctionContext::append_statement(RC<core::Stat> stat)
{
    assert(!blocks_.empty());
    blocks_.top()->stats.push_back(std::move(stat));
}

void FunctionContext::add_argument(const core::Type *type, bool is_reference)
{
    func_->argument_types.push_back({ type, is_reference });
}

void FunctionContext::set_return(const core::Type *type, bool is_reference)
{
    func_->return_type = { type, is_reference };
}

void FunctionContext::append_statement(core::Stat stat)
{
    append_statement(newRC<core::Stat>(std::move(stat)));
}

void FunctionContext::push_block(RC<core::Block> block)
{
    blocks_.push(std::move(block));
}

void FunctionContext::pop_block()
{
    assert(blocks_.size() >= 2);
    blocks_.pop();
}

TypeContext *FunctionContext::get_type_context()
{
    return module_ ? module_->_get_type_context() : type_context_.get();
}

Module *FunctionContext::get_module() const
{
    return module_;
}

bool FunctionContext::is_contexted() const
{
    return module_ != nullptr;
}

size_t FunctionContext::get_index_in_module() const
{
    return index_in_module_;
}

size_t FunctionContext::alloc_local_var(const core::Type *type)
{
    const size_t index = func_->local_alloc_types.size();
    func_->local_alloc_types.push_back(type);
    return index;
}

RC<FunctionContext> FunctionContext::clone_with_module(Module *mod)
{
    assert(!module_);
    auto ret = RC<FunctionContext>(new FunctionContext(Uninit{}));
    ret->func_            = func_;
    ret->type_context_    = type_context_;
    ret->index_in_module_ = index_in_module_;
    ret->module_          = mod;
    ret->blocks_          = blocks_;
    return ret;
}

RC<const core::Func> FunctionContext::get_core_func() const
{
    return func_;
}

CUJ_NAMESPACE_END(cuj::dsl)
