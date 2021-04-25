#pragma once

#include <cuj/ast/context.h>

#include <cuj/gen/printer.h>

#if CUJ_ENABLE_CUDA && CUJ_ENABLE_LLVM
#include <cuj/gen/ptx.h>
#endif

CUJ_NAMESPACE_BEGIN(cuj::ast)

namespace detail
{

    inline std::stack<Context*> &get_context_stack()
    {
        static std::stack<Context*> ret;
        return ret;
    }

    template<typename T, typename = void>
    struct HasStructName : std::false_type { };

    template<typename T>
    struct HasStructName<T, std::void_t<decltype(T::struct_name())>>
        : std::true_type{ };

} // namespace detail

template<typename T>
const ir::Type *Context::get_type()
{
    static_assert(
        std::is_same_v<T, void> ||
        is_array<T>             ||
        std::is_arithmetic_v<T> ||
        is_pointer<T>           ||
        is_cuj_class<T>);

    const auto type_idx = std::type_index(typeid(T));
    if(auto it = types_.find(type_idx); it != types_.end())
        return it->second.get();
    
    if constexpr(std::is_same_v<T, void> || std::is_arithmetic_v<T>)
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
    else if constexpr(is_intrinsic<T>)
    {
        auto &type = types_[type_idx];
        type = newRC<ir::Type>(T::get_type());
        return type.get();
    }
    else
    {
        auto &type = types_[type_idx];
        type = newRC<ir::Type>(ir::StructType{});
        auto &s_type = type->as<ir::StructType>();

        StructTypeRecorder type_recorder(&s_type);
        T obj(&type_recorder);

        if constexpr(detail::HasStructName<T>::value)
        {
            s_type.name = T::struct_name();
        }
        else
        {
            s_type.name = "CUJStruct" + std::to_string(struct_name_index_++);
        }

        return type.get();
    }
}

template<typename FuncType>
Function<FuncType> Context::get_function(std::string_view name) const
{
    const auto it = func_name_to_index_.find(name);
    if(it == func_name_to_index_.end())
        throw CUJException("unknown function name: " + std::string(name));
    return get_function<FuncType>(it->second);
}

template<typename FuncType>
Function<FuncType> Context::get_function(int index) const
{
    CUJ_ASSERT(0 <= index && index < static_cast<int>(funcs_.size()));
    return Function<FuncType>(index);
}

template<typename Ret, typename Callable>
Function<FunctionType<RawToCUJType<Ret>, Callable>> Context::add_function(
    std::string name, Callable &&callable)
{
    return add_function<Ret>(
        std::move(name),
        ir::Function::Type::Default,
        std::forward<Callable>(callable));
}

template<typename Ret, typename Callable>
Function<FunctionType<RawToCUJType<Ret>, Callable>> Context::add_function(
    std::string name, ir::Function::Type type, Callable &&callable)
{
    return add_function_impl<Ret>(
        std::move(name), type,
        std::forward<Callable>(callable),
        reinterpret_cast<FunctionArgs<RawToCUJType<Ret>, Callable>*>(nullptr),
        std::make_index_sequence<
            std::tuple_size_v<FunctionArgs<RawToCUJType<Ret>, Callable>>>());
}

template<typename Ret, typename Callable>
Function<FunctionType<RawToCUJType<Ret>, Callable>> Context::add_function(
    Callable &&callable)
{
    const std::string name =
        "_cuj_auto_named_func_" + std::to_string(funcs_.size());
    return this->add_function<Ret>(
        std::move(name), std::forward<Callable>(callable));
}

template<typename Ret, typename Callable>
Function<FunctionType<RawToCUJType<Ret>, Callable>> Context::add_function(
    ir::Function::Type type, Callable &&callable)
{
    const std::string name =
        "_cuj_auto_named_func_" + std::to_string(funcs_.size());
    return this->add_function<Ret>(
        std::move(name), type, std::forward<Callable>(callable));
}

template<typename FuncType>
Function<FuncType> Context::begin_function(
    std::string name, ir::Function::Type type)
{
    if(func_name_to_index_.count(name))
        throw CUJException("repeated function name");

    auto ret_type = get_type<typename Function<FuncType>::ReturnType>();

    std::vector<const ir::Type*> arg_types;
    Function<FuncType>::get_arg_types(arg_types);

    funcs_.push_back(
        newBox<FunctionContext>(std::move(name), type, ret_type, arg_types));
    func_stack_.push(funcs_.back().get());

    const int index = static_cast<int>(funcs_.size() - 1);
    func_name_to_index_[name] = index;

    return Function<FuncType>(index);
}

inline void Context::end_function()
{
    func_stack_.pop();
}

inline ir::Program Context::gen_ir() const
{
    ir::IRBuilder builder;
    gen_ir_impl(builder);
    return builder.get_prog();
}

inline std::string Context::gen_ir_string() const
{
    gen::IRPrinter printer;
    printer.print(gen_ir());
    return printer.get_string();
}

#if CUJ_ENABLE_LLVM

inline gen::NativeJIT Context::gen_native_jit() const
{
    gen::NativeJIT jit;
    jit.generate(gen_ir());
    return jit;
}

#endif // #if CUJ_ENABLE_LLVM

#if CUJ_ENABLE_CUDA && CUJ_ENABLE_LLVM

inline std::string Context::gen_ptx32() const
{
    gen::PTXGenerator generator;
    generator.set_target(gen::PTXGenerator::Target::PTX32);
    generator.generate(gen_ir());
    return generator.get_result();
}

inline std::string Context::gen_ptx64() const
{
    gen::PTXGenerator generator;
    generator.set_target(gen::PTXGenerator::Target::PTX64);
    generator.generate(gen_ir());
    return generator.get_result();
}

#endif // #if CUJ_ENABLE_CUDA && CUJ_ENABLE_LLVM

template<typename Ret, typename Callable, typename...Args, size_t...Is>
Function<FunctionType<RawToCUJType<Ret>, Callable>> Context::add_function_impl(
    std::string        name,
    ir::Function::Type type,
    Callable         &&callable,
    std::tuple<Args...>*,
    std::index_sequence<Is...>)
{
    using ArgsTuple = std::tuple<Args...>;

    auto ret = begin_function<FunctionType<RawToCUJType<Ret>, Callable>>(
        std::move(name), type);
    CUJ_SCOPE_GUARD({ end_function(); });
    
    std::apply(
        std::forward<Callable>(callable),
        std::tuple{ get_current_function()->create_arg<
            std::tuple_element_t<Is, ArgsTuple>>()... });
    
    return ret;
}

inline void Context::gen_ir_impl(ir::IRBuilder &builder) const
{
    CUJ_ASSERT(func_stack_.empty());

    for(auto &p : types_)
        builder.add_type(p.first, p.second);

    for(auto &f : funcs_)
        f->gen_ir(builder);
}

inline FunctionContext *Context::get_current_function()
{
    CUJ_ASSERT(!func_stack_.empty());
    return func_stack_.top();
}

inline FunctionContext *Context::get_function_context(int func_index)
{
    CUJ_ASSERT(0 <= func_index);
    CUJ_ASSERT(func_index < static_cast<int>(funcs_.size()));
    return funcs_[func_index].get();
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

inline FunctionContext *get_current_function()
{
    return get_current_context()->get_current_function();
}

CUJ_NAMESPACE_END(cuj::ast)
