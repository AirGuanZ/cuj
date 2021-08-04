#include <cuj/ir/builder.h>

CUJ_NAMESPACE_BEGIN(cuj::ir)

void IRBuilder::set_assertion(bool enabled)
{
    enable_assertion_ = enabled;
}

bool IRBuilder::is_assertion_enabled() const
{
    return enable_assertion_;
}

const Program &IRBuilder::get_prog() const
{
    CUJ_INTERNAL_ASSERT(!cur_func_);
    return prog_;
}

void IRBuilder::add_type(std::type_index type_index, RC<Type> type)
{
    prog_.types.insert({ type_index, std::move(type) });
}

void IRBuilder::begin_function(
    std::string name, Function::Type type, const Type *ret_type)
{
    CUJ_INTERNAL_ASSERT(!cur_func_);
    CUJ_INTERNAL_ASSERT(blocks_.empty());

    cur_func_ = newRC<Function>();
    cur_func_->name = std::move(name);
    cur_func_->type = type;
    cur_func_->body = newRC<Block>();

    cur_func_->ret_type = ret_type;

    next_temp_value_ = 0;

    blocks_.push(cur_func_->body);
}

void IRBuilder::end_function()
{
    CUJ_INTERNAL_ASSERT(cur_func_);
    CUJ_INTERNAL_ASSERT(blocks_.size() == 1);
    prog_.funcs.push_back(cur_func_);
    cur_func_ = {};
    blocks_   = {};
}

void IRBuilder::add_function_arg(int alloc_index)
{
    CUJ_INTERNAL_ASSERT(cur_func_);
    cur_func_->args.push_back({ alloc_index });
}

void IRBuilder::add_host_imported_function(RC<ImportedHostFunction> func)
{
    prog_.funcs.push_back(std::move(func));
}

void IRBuilder::add_alloc(int alloc_index, const Type *type)
{
    CUJ_INTERNAL_ASSERT(cur_func_);
    CUJ_INTERNAL_ASSERT(!cur_func_->index_to_allocs.count(alloc_index));
    auto alloc = newRC<Allocation>();
    alloc->type = type;
    cur_func_->index_to_allocs.insert({ alloc_index, std::move(alloc) });
}

RC<Allocation> IRBuilder::get_alloc(int alloc_index) const
{
    CUJ_INTERNAL_ASSERT(cur_func_);
    const auto it = cur_func_->index_to_allocs.find(alloc_index);
    CUJ_INTERNAL_ASSERT(it != cur_func_->index_to_allocs.end());
    return it->second;
}

void IRBuilder::push_block(RC<Block> block)
{
    CUJ_INTERNAL_ASSERT(cur_func_);
    blocks_.push(std::move(block));
}

void IRBuilder::pop_block()
{
    CUJ_INTERNAL_ASSERT(blocks_.size() >= 2);
    blocks_.pop();
}

void IRBuilder::append_statement(RC<Statement> stat)
{
    CUJ_INTERNAL_ASSERT(cur_func_);
    CUJ_INTERNAL_ASSERT(!blocks_.empty());
    blocks_.top()->stats.push_back(std::move(stat));
}

void IRBuilder::append_assign(BasicTempValue lhs, Value rhs)
{
    auto stat = newRC<Statement>(Assign{ lhs, rhs });
    append_statement(std::move(stat));
}

BasicTempValue IRBuilder::gen_temp_value(const Type *type)
{
    return BasicTempValue{ type, next_temp_value_++ };
}

CUJ_NAMESPACE_END(cuj::ir)
