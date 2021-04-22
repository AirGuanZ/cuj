#pragma once

#include <cuj/ast/func.h>

CUJ_NAMESPACE_BEGIN(cuj::ast)

template<typename T, typename...Args>
Value<T> Function::alloc_stack_var(bool is_arg, Args &&...args)
{
    auto alloc_type = get_current_context()->get_type<T>();

    RC<InternalStackAllocationValue<T>> address = alloc_on_stack<T>(alloc_type);
    if(is_arg)
        arg_indices_.push_back(address->alloc_index);

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

inline Function::Function(std::string name, ir::Function::Type type)
{
    blocks_.push(newRC<Block>());

    set_name(std::move(name));
    set_type(type);
}

inline void Function::set_name(std::string name)
{
    name_ = std::move(name);
}

inline void Function::set_type(ir::Function::Type type)
{
    type_ = type;
}

inline void Function::append_statement(RC<Statement> stat)
{
    blocks_.top()->append(std::move(stat));
}

inline void Function::push_block(RC<Block> block)
{
    blocks_.push(std::move(block));
}

inline void Function::pop_block()
{
    CUJ_ASSERT(blocks_.size() >= 2);
    blocks_.pop();
}

template<typename T, typename...Args>
Value<T> Function::create_stack_var(Args &&...args)
{
    return alloc_stack_var<T>(false, std::forward<Args>(args)...);
}

template<typename T>
RC<InternalStackAllocationValue<T>> Function::alloc_on_stack(
    const ir::Type *type)
{
    const int alloc_index = static_cast<int>(stack_allocs_.size());
    stack_allocs_.push_back({ type, alloc_index });
    
    auto address = newRC<InternalStackAllocationValue<T>>();
    address->alloc_index = alloc_index;

    return address;
}

template<typename T>
Value<T> Function::create_arg()
{
    return alloc_stack_var<T>(true);
}

inline void Function::gen_ir(ir::IRBuilder &builder) const
{
    CUJ_ASSERT(blocks_.size() == 1);

    builder.begin_function(name_, type_);
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
