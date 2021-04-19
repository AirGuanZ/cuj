#pragma once

#include <cuj/ast/context.h>

CUJ_NAMESPACE_BEGIN(cuj::ast)

namespace detail
{

    inline std::stack<Context*> &get_context_stack()
    {
        static std::stack<Context*> ret;
        return ret;
    }

} // namespace detail

template<typename T>
const ir::Type *Context::get_type()
{
    static_assert(
        is_array<T> ||std::is_arithmetic_v<T> || is_pointer<T> || is_cuj_class<T>);

    const auto type_idx = std::type_index(typeid(T));
    if(auto it = types_.find(type_idx); it != types_.end())
        return it->second.get();
        
    if constexpr(std::is_arithmetic_v<T>)
    {
        auto &type = types_[type_idx];
        type = newRC<ir::Type>(ir::to_builtin_type_value<T>);
        return type.get();
    }
    else if constexpr(is_array<T>)
    {
        auto &type = types_[type_idx];
        type = newRC<ir::Type>(ir::ArrayType{ 0, nullptr });

        auto elem_type = this->get_type<typename T::ElementType>();
        type->as<ir::ArrayType>() = { T::ElementCount, elem_type };

        return type.get();
    }
    else if constexpr(is_pointer<T>)
    {
        auto &type = types_[type_idx];
        type = newRC<ir::Type>(ir::PointerType{ nullptr });

        auto pointed_type = this->get_type<typename T::PointedType>();
        type->as<ir::PointerType>().pointed_type = pointed_type;
        
        return type.get();
    }
    else
    {
        auto &type = types_[type_idx];
        type = newRC<ir::Type>(ir::StructType{});

        StructTypeRecorder type_recorder(&type->as<ir::StructType>());
        (void)T(&type_recorder);
        
        return type.get();
    }
}

inline void Context::begin_function(std::string name, ir::Function::Type type)
{
    CUJ_ASSERT(!current_func_);
    current_func_ = newBox<Function>(std::move(name), type);
}

inline void Context::end_function()
{
    CUJ_ASSERT(current_func_);
    completed_funcs_.push_back(std::move(current_func_));
    CUJ_ASSERT(!current_func_);
}

inline void Context::gen_ir(ir::IRBuilder &builder) const
{
    CUJ_ASSERT(!current_func_);

    for(auto &p : types_)
        builder.add_type(p.first, p.second);

    for(auto &f : completed_funcs_)
        f->gen_ir(builder);
}

inline Function *Context::get_current_function()
{
    CUJ_ASSERT(current_func_);
    return current_func_.get();
}

inline void push_context(Context *context)
{
    CUJ_ASSERT(context);
    detail::get_context_stack().push(context);
}

inline void pop_context()
{
    CUJ_ASSERT(!detail::get_context_stack().empty());
    detail::get_context_stack().pop();
}

inline Context *get_current_context()
{
    CUJ_ASSERT(!detail::get_context_stack().empty());
    return detail::get_context_stack().top();
}

inline Function *get_current_function()
{
    return get_current_context()->get_current_function();
}

CUJ_NAMESPACE_END(cuj::ast)
