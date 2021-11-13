#pragma once

#include <cuj/ast/context.h>

#include <cuj/gen/c.h>
#include <cuj/gen/llvm.h>
#include <cuj/gen/printer.h>

#if CUJ_ENABLE_CUDA
#include <cuj/gen/nvrtc.h>
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

    class StructMemberRecorder
    {
    public:

        Context                       *context;
        std::vector<const ir::Type *> &members;

        template<typename MemberProxyBase>
        void operator()() const
        {
            members.push_back(
                context->get_type<typename MemberProxyBase::Member>());
        }
    };

} // namespace detail

template<typename T_>
const ir::Type *Context::get_type()
{
    using T = deval_t<to_cuj_t<T_>>;

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
        s_type.name = "CUJStruct" + std::to_string(struct_count_++);

        detail::StructMemberRecorder recorder{ this, s_type.mem_types };
        T::foreach_member(recorder);

        return type.get();
    }
}

template<typename FuncType>
Function<void, detail::deval_func_t<FuncType>>
    Context::declare_function(std::string name)
{
    if(auto it = func_name_to_index_.find(name); it != func_name_to_index_.end())
    {
        auto &ctx = funcs_.at(it->second);
        if(!ctx.is<Box<FunctionContext>>())
            throw CUJException("imported function cannot be redeclared");
        return Function<void, detail::deval_func_t<FuncType>>(it->second);
    }

    auto ret_type =
        get_type<typename Function<void, detail::deval_func_t<FuncType>>::ReturnType>();

    std::vector<const ir::Type *> arg_types;
    Function<void, detail::deval_func_t<FuncType>>::get_arg_types(arg_types);

    const int index = static_cast<int>(funcs_.size());
    func_name_to_index_[name] = index;

    funcs_.insert(
        {
            index,
            ContextFunc
            {
                newBox<FunctionContext>(
                    std::move(name),
                    ir::Function::Type::Default,
                    ret_type, arg_types)
            }
        });

    return Function<void, detail::deval_func_t<FuncType>>(index);
}

template<typename FuncType>
Function<void, detail::deval_func_t<FuncType>> Context::declare_function()
{
    std::string name =
        "_cuj_auto_named_func_" + std::to_string(funcs_.size());
    return declare_function<FuncType>(std::move(name));
}

template<typename Ret, typename Callable>
Function<void, func_t<to_cuj_t<Ret>, Callable>> Context::add_function(
    std::string name, Callable &&callable)
{
    return add_function<Ret>(
        std::move(name),
        ir::Function::Type::Default,
        std::forward<Callable>(callable));
}

template<typename Ret, typename Callable>
Function<void, func_t<to_cuj_t<Ret>, Callable>> Context::add_function(
    std::string name, ir::Function::Type type, Callable &&callable)
{
    return add_function_impl<Ret>(
        std::move(name), type,
        std::forward<Callable>(callable),
        reinterpret_cast<func_args_t<to_cuj_t<Ret>, Callable>*>(0),
        std::make_index_sequence<
            std::tuple_size_v<func_args_t<to_cuj_t<Ret>, Callable>>>());
}

template<typename Ret, typename Callable>
Function<void, func_t<to_cuj_t<Ret>, Callable>> Context::add_function(
    Callable &&callable)
{
    std::string name =
        "_cuj_auto_named_func_" + std::to_string(funcs_.size());
    return this->add_function<Ret>(
        std::move(name), std::forward<Callable>(callable));
}

template<typename Ret, typename Callable>
Function<void, func_t<to_cuj_t<Ret>, Callable>> Context::add_function(
    ir::Function::Type type, Callable &&callable)
{
    std::string name =
        "_cuj_auto_named_func_" + std::to_string(funcs_.size());
    return this->add_function<Ret>(
        std::move(name), type, std::forward<Callable>(callable));
}

template<typename FuncType>
Function<FuncType, FuncType> Context::import_raw_host_function(
    std::string name, uint64_t func_ptr, RC<UntypedOwner> ctx_data)
{
    if(func_name_to_index_.count(name))
        throw CUJException("repeated function name");

    auto ret_type = get_type<typename Function<FuncType, FuncType>::ReturnType>();

    std::vector<const ir::Type *> arg_types;
    Function<FuncType, FuncType>::get_arg_types(arg_types);

    const int index = static_cast<int>(funcs_.size());
    func_name_to_index_[name] = index;

    auto &func = *funcs_.insert(
        { index, ContextFunc{ newRC<ir::ImportedHostFunction>() } })
            .first->second.as<RC<ir::ImportedHostFunction>>();
    
    func.context_data = ctx_data;
    func.address      = func_ptr;
    func.is_external  = true;
    func.symbol_name  = std::move(name);
    func.arg_types    = std::move(arg_types);
    func.ret_type     = ret_type;

    return Function<FuncType, FuncType>(index);
}

template<typename FuncType>
Function<FuncType, FuncType> Context::import_raw_host_function(
    uint64_t func_ptr, RC<UntypedOwner> ctx_data)
{
    std::string name =
        "_cuj_auto_named_func_" + std::to_string(funcs_.size());
    return this->import_raw_host_function<FuncType>(
        std::move(name), func_ptr, ctx_data);
}

template<typename Ret, typename ... Args>
Function<Ret(Args ...), func_trait_detail::to_cuj_func_t<Ret, Args...>>
    Context::import_host_functor(std::function<Ret(Args ...)> func)
{
    using namespace func_trait_detail;

    RC<UntypedOwner> owner = newRC<UntypedOwner>(std::move(func));
    auto imported_func = [](void* raw_func, Args...args)
    {
        auto pfunc = reinterpret_cast<std::function<Ret(Args...)>*>(raw_func);
        return (*pfunc)(args...);
    };

    Ret(*p_imported_func)(void*, Args...);
    p_imported_func = imported_func;

    static_assert((std::is_same_v<rm_cvref_t<Args>, Args> && ...));
    using FuncType = to_cuj_arg_t<Ret>(to_cuj_arg_t<Args>...);
    auto f = import_raw_host_function<FuncType>(
        reinterpret_cast<uint64_t>(p_imported_func), owner);

    return Function<Ret(Args...), to_cuj_func_t<Ret, Args...>>(f.get_index());
}

template<typename Ret, typename ... Args>
Function<Ret(Args ...), func_trait_detail::to_cuj_func_t<Ret, Args...>>
    Context::import_host_functor(std::string name, std::function<Ret(Args ...)> func)
{
    using namespace func_trait_detail;

    RC<UntypedOwner> owner = newRC<UntypedOwner>(std::move(func));
    auto imported_func = [](uint64_t raw_func, Args...args)
    {
        auto pfunc = reinterpret_cast<std::function<Ret(Args...)>*>(raw_func);
        return (*pfunc)(args...);
    };

    Ret(*p_imported_func)(uint64_t, Args...);
    p_imported_func = imported_func;

    static_assert((std::is_same_v<rm_cvref_t<Args>, Args> && ...));
    using FuncType = to_cuj_arg_t<Ret>(to_cuj_arg_t<Args>...);
    auto f = import_raw_host_function<FuncType>(
        std::move(name), reinterpret_cast<uint64_t>(p_imported_func), owner);

    return Function<Ret(Args...), to_cuj_func_t<Ret, Args...>>(f.get_index());
}

template<typename Ret, typename ... Args>
Function<Ret(Args...), func_trait_detail::to_cuj_func_t<Ret, Args...>>
    Context::import_host_function(Ret (*func_ptr)(Args ...))
{
    using namespace func_trait_detail;
    static_assert((std::is_same_v<rm_cvref_t<Args>, Args> && ...));
    using FuncType = to_cuj_arg_t<Ret>(to_cuj_arg_t<Args>...);
    auto f = import_raw_host_function<FuncType>(reinterpret_cast<uint64_t>(func_ptr));
    return Function<Ret(Args...), to_cuj_func_t<Ret, Args...>>(f.get_index());
}

template<typename Ret, typename ... Args>
Function<Ret(Args...), func_trait_detail::to_cuj_func_t<Ret, Args...>>
    Context::import_host_function(std::string name, Ret (*func_ptr)(Args ...))
{
    using namespace func_trait_detail;
    static_assert((std::is_same_v<rm_cvref_t<Args>, Args> && ...));
    using FuncType = to_cuj_arg_t<Ret>(to_cuj_arg_t<Args>...);
    auto f = import_raw_host_function<FuncType>(
                std::move(name), reinterpret_cast<uint64_t>(func_ptr));
    return Function<Ret(Args...), to_cuj_func_t<Ret, Args...>>(f.get_index());
}

template<typename FuncType>
Function<void, detail::deval_func_t<FuncType>>
    Context::begin_function(ir::Function::Type type)
{
    const std::string name =
        "_cuj_auto_named_func_" + std::to_string(funcs_.size());
    return begin_function<detail::deval_func_t<FuncType>>(
        std::move(name), type);
}

template<typename FuncType>
Function<void, detail::deval_func_t<FuncType>> Context::begin_function(
    std::string name, ir::Function::Type type)
{
    if(auto it = func_name_to_index_.find(name); it != func_name_to_index_.end())
    {
        auto &ctx = funcs_.at(it->second);
        auto func_ctx = ctx.as_if<Box<FunctionContext>>();
        if(!func_ctx)
            throw CUJException("imported function cannot be redefined");
        if(func_ctx->get()->get_type() != type)
        {
            throw CUJException(
                "function.definition.type != function.declaration.type");
        }
        func_stack_.push(func_ctx->get());
        return Function<void, detail::deval_func_t<FuncType>>(it->second);
    }

    auto ret_type =
        get_type<typename Function<
            void, detail::deval_func_t<FuncType>>::ReturnType>();

    std::vector<const ir::Type*> arg_types;
    Function<void, detail::deval_func_t<FuncType>>::get_arg_types(arg_types);

    const int index = static_cast<int>(funcs_.size());
    func_name_to_index_[name] = index;

    auto &ctx = funcs_.insert(
        {
            index,
            ContextFunc
            {
                newBox<FunctionContext>(
                    std::move(name), type, ret_type, arg_types)
            }
        }).first->second;

    func_stack_.push(ctx.as<Box<FunctionContext>>().get());

    return Function<void, detail::deval_func_t<FuncType>>(index);
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

inline std::string Context::gen_llvm_string(
    gen::LLVMIRGenerator::Target target) const
{
    gen::LLVMIRGenerator llvm_gen;
    llvm_gen.set_target(target);
    llvm_gen.generate(gen_ir());
    return llvm_gen.get_string();
}

inline gen::NativeJIT Context::gen_native_jit(const gen::Options &options) const
{
    gen::NativeJIT jit;
    jit.generate(gen_ir(), options);
    return jit;
}

inline std::string Context::gen_c(bool cuda) const
{
    gen::CGenerator gen;
    if(cuda)
        gen.set_cuda();
    gen.print(gen_ir());
    return gen.get_string();
}

#if CUJ_ENABLE_CUDA

inline std::string Context::gen_ptx(const gen::Options &options) const
{
    gen::PTXGenerator generator;
    generator.generate(gen_ir(), options);
    return generator.get_result();
}

inline std::string Context::gen_ptx_nvrtc(const gen::Options &options) const
{
    gen::NVRTC generator;
    generator.generate(gen_ir(), options);
    return generator.get_ptx();
}

#endif // #if CUJ_ENABLE_CUDA

template<typename FuncType>
Function<void, detail::deval_func_t<FuncType>>
    Context::get_function(std::string_view name) const
{
    const auto it = func_name_to_index_.find(name);
    if(it == func_name_to_index_.end())
        throw CUJException("unknown function name: " + std::string(name));
    return get_function<detail::deval_func_t<FuncType>>(it->second);
}

template<typename FuncType>
Function<void, detail::deval_func_t<FuncType>>
    Context::get_function(int index) const
{
    CUJ_INTERNAL_ASSERT(0 <= index && index < static_cast<int>(funcs_.size()));
    return Function<void, detail::deval_func_t<FuncType>>(index);
}

template<typename Ret, typename Callable, typename...Args, size_t...Is>
Function<void, func_t<to_cuj_t<Ret>, Callable>> Context::add_function_impl(
    std::string        name,
    ir::Function::Type type,
    Callable         &&callable,
    std::tuple<Args...>*,
    std::index_sequence<Is...>)
{
    using ArgsTuple = std::tuple<Args...>;

    auto ret = begin_function<func_t<to_cuj_t<Ret>, Callable>>(
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
    CUJ_INTERNAL_ASSERT(func_stack_.empty());

    for(auto &f : funcs_)
    {
        f.second.match(
            [&](const Box<FunctionContext> &func)
        {
            func->gen_ir(builder);
        },
            [&](const RC<ir::ImportedHostFunction> &func)
        {
            builder.add_host_imported_function(func);
        });
    }

    for(auto &p : types_)
        builder.add_type(p.second);
}

inline FunctionContext *Context::get_current_function()
{
    CUJ_INTERNAL_ASSERT(!func_stack_.empty());
    return func_stack_.top();
}

inline const Variant<Box<FunctionContext>, RC<ir::ImportedHostFunction>> &
    Context::get_function_context(int func_index)
{
    CUJ_INTERNAL_ASSERT(0 <= func_index);
    CUJ_INTERNAL_ASSERT(func_index < static_cast<int>(funcs_.size()));
    return funcs_[func_index];
}

inline std::string Context::get_function_name(int func_index)
{
    auto &ctx = get_function_context(func_index);
    return ctx.match(
        [](const Box<FunctionContext> &c)
    {
        return c->get_name();
    },
        [](const RC<ir::ImportedHostFunction> &c)
    {
        return c->symbol_name;
    });
}

inline void push_context(Context *context)
{
    CUJ_INTERNAL_ASSERT(context);
    detail::get_context_stack().push(context);
}

inline void pop_context()
{
    CUJ_INTERNAL_ASSERT(!detail::get_context_stack().empty());
    detail::get_context_stack().pop();
}

inline Context *get_current_context()
{
    CUJ_INTERNAL_ASSERT(!detail::get_context_stack().empty());
    return detail::get_context_stack().top();
}

inline FunctionContext *get_current_function()
{
    return get_current_context()->get_current_function();
}

CUJ_NAMESPACE_END(cuj::ast)
