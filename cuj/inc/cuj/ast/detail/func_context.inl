#pragma once

#include <cuj/ast/func_context.h>

CUJ_NAMESPACE_BEGIN(cuj::ast)

template<typename T, typename...Args>
Value<T> FunctionContext::alloc_stack_var(bool is_arg, Args &&...args)
{
    auto alloc_type = get_current_context()->get_type<T>();

    RC<InternalStackAllocationValue<T>> address = alloc_on_stack<T>(alloc_type);
    if(is_arg)
    {
        if(arg_indices_.size() > arg_types_.size())
            throw CUJException("too many function argments");
        if(alloc_type != arg_types_[arg_indices_.size()])
            throw CUJException("unmatched function argument type");
        arg_indices_.push_back(address->alloc_index);
    }

    static_assert(!is_array<T>   || sizeof...(args) == 0);
    static_assert(!is_pointer<T> || sizeof...(args) <= 1);
    static_assert(!std::is_arithmetic_v<T> || sizeof...(args) <= 1);

    if constexpr(is_array<T>)
    {
        auto cast_addr = newRC<InternalArrayAllocAddress<T>>();
        cast_addr->arr_alloc = address;

        auto impl = newRC<InternalArrayValue<
            typename T::ElementType, T::ElementCount>>();
        impl->data_ptr = cast_addr;
        return Value<T>(std::move(impl));
    }
    else if constexpr(is_pointer<T>)
    {
        auto impl = newRC<InternalPointerLeftValue<typename T::PointedType>>();
        impl->address = std::move(address);
        
        auto ret = Value<T>(std::move(impl));
        if constexpr(sizeof...(args) == 1)
            ret = (std::forward<Args>(args), ...);

        return ret;
    }
    else if constexpr(std::is_arithmetic_v<T>)
    {
        auto impl = newRC<InternalArithmeticLeftValue<T>>();
        impl->address = std::move(address);

        auto ret = Value<T>(std::move(impl));
        if constexpr(sizeof...(args) == 1)
            ret = (std::forward<Args>(args), ...);

        return std::move(ret);
    }
    else if constexpr(is_intrinsic<T>)
    {
        return T(address, std::forward<Args>(args)...);
    }
    else
    {
        auto impl = newRC<InternalClassLeftValue<T>>();
        impl->address = address;
        impl->obj     = newBox<T>(std::move(address), std::forward<Args>(args)...);
        return Value<T>(std::move(impl));
    }
}

inline FunctionContext::FunctionContext(
    std::string                   name,
    ir::Function::Type            type,
    const ir::Type               *ret_type,
    std::vector<const ir::Type *> arg_types)
{
    blocks_.push(newRC<Block>());

    set_name(std::move(name));
    set_type(type);

    arg_types_ = std::move(arg_types);
    ret_type_  = ret_type;
}

inline void FunctionContext::set_name(std::string name)
{
    name_ = std::move(name);
}

inline void FunctionContext::set_type(ir::Function::Type type)
{
    type_ = type;
}

inline void FunctionContext::append_statement(RC<Statement> stat)
{
    blocks_.top()->append(std::move(stat));
}

inline void FunctionContext::push_block(RC<Block> block)
{
    blocks_.push(std::move(block));
}

inline void FunctionContext::pop_block()
{
    CUJ_ASSERT(blocks_.size() >= 2);
    blocks_.pop();
}

inline const std::string &FunctionContext::get_name() const
{
    return name_;
}

inline int FunctionContext::get_arg_count() const
{
    return static_cast<int>(arg_types_.size());
}

inline const ir::Type *FunctionContext::get_arg_type(int index) const
{
    return arg_types_[index];
}

inline const ir::Type *FunctionContext::get_return_type() const
{
    return ret_type_;
}

template<typename T, typename...Args>
Value<T> FunctionContext::create_stack_var(Args &&...args)
{
    return alloc_stack_var<T>(false, std::forward<Args>(args)...);
}

template<typename T>
RC<InternalStackAllocationValue<T>> FunctionContext::alloc_on_stack(
    const ir::Type *type)
{
    const int alloc_index = static_cast<int>(stack_allocs_.size());
    stack_allocs_.push_back({ type, alloc_index });
    
    auto address = newRC<InternalStackAllocationValue<T>>();
    address->alloc_index = alloc_index;

    return address;
}

template<typename T>
Value<T> FunctionContext::create_arg()
{
    return alloc_stack_var<T>(true);
}

inline void FunctionContext::gen_ir(ir::IRBuilder &builder) const
{
    CUJ_ASSERT(blocks_.size() == 1);

    if(arg_indices_.size() != arg_types_.size())
    {
        throw CUJException(
            "function argument(s) is(are) declared but not defined");
    }

    builder.begin_function(name_, type_, ret_type_);
    CUJ_SCOPE_GUARD({ builder.end_function(); });

    for(auto &sa : stack_allocs_)
        builder.add_alloc(sa.alloc_index, sa.type);

    for(auto arg_idx : arg_indices_)
    {
        auto &sa = stack_allocs_[arg_idx];
        builder.add_function_arg(sa.alloc_index);
    }

    blocks_.top()->gen_ir(builder);
}

CUJ_NAMESPACE_END(cuj::ast)
