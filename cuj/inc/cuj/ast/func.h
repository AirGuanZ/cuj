#pragma once

#include <cuj/ast/expr.h>

CUJ_NAMESPACE_BEGIN(cuj::ast)

namespace detail
{
    
    template<typename T>
    struct FuncRetType
    {
        using Type = Value<T>;
    };

    template<>
    struct FuncRetType<void>
    {
        using Type = void;
    };

    template<typename T>
    struct MakeArgValue
    {
        static_assert(std::is_arithmetic_v<T> || is_cuj_class<T>);

        static auto process(const T &val)
        {
            if constexpr(is_cuj_class<T>)
                return val;
            else
                return create_literial(val);
        }
    };

    template<typename T>
    struct MakeArgValue<ArithmeticVariable<T>>
    {
        static auto process(const ArithmeticVariable<T> &val)
        {
            return MakeArgValue<ArithmeticValue<T>>::process(val);
        }
    };

    template<typename T>
    struct MakeArgValue<PointerVariable<T>>
    {
        static auto process(const PointerVariable<T> &val)
        {
            return MakeArgValue<PointerImpl<T>>::process(val);
        }
    };

    template<typename T, size_t N>
    struct MakeArgValue<ArrayVariable<T, N>>
    {
        static auto process(const ArrayVariable<T, N> &val)
        {
            return MakeArgValue<ArrayImpl<T, N>>::process(val);
        }
    };

    template<typename T, size_t N>
    struct MakeArgValue<ArrayImpl<T, N>>
    {
        static auto process(const ArrayImpl<T, N> &val)
        {
            return val;
        }
    };

    template<typename T>
    struct MakeArgValue<ArithmeticValue<T>>
    {
        static auto process(const ArithmeticValue<T> &val)
        {
            return val;
        }
    };

    template<typename T>
    struct MakeArgValue<PointerImpl<T>>
    {
        static auto process(const PointerImpl<T> &val)
        {
            return val;
        }
    };

} // namespace detail

class Context;
class FunctionContext;

template<typename Ret, typename...Args>
class FunctionImpl
{
public:

    explicit FunctionImpl(int func_index);

    template<typename...CallArgs>
    typename detail::FuncRetType<Ret>::Type
        operator()(const CallArgs &...args) const;

    int get_index() const;

    std::string get_name() const;

private:
    
    bool check_return_type() const;
    
    bool check_param_types() const;

    template<size_t...Is>
    bool check_param_types_aux(std::index_sequence<Is...>) const;

    int index_;
};

template<typename ForcedCFunction, typename T>
class Function;

namespace detail
{

    template<typename T>
    auto CFunctionArgTypeAux()
    {
        if constexpr(is_cuj_class<T> || is_array<T> || is_pointer<T>)
            return reinterpret_cast<void**>(0);
        else
            return reinterpret_cast<T*>(0);
    }

    template<typename T>
    using c_func_arg_t = rm_cvref_t<std::remove_pointer_t<decltype(
        CFunctionArgTypeAux<T>())>>;

    template<typename T>
    struct DeValFuncAux
    {
        
    };

    template<typename R, typename...Args>
    struct DeValFuncAux<R(Args...)>
    {
        using Type = deval_t<to_cuj_t<R>>(deval_t<to_cuj_t<Args>>...);
    };

    template<typename Func>
    using deval_func_t = typename DeValFuncAux<Func>::Type;

} // namespace detail

template<typename ForcedCFunctionType, typename Ret, typename...Args>
class Function<ForcedCFunctionType, Ret(Args...)>
{
public:

    using ReturnType = deval_t<to_cuj_t<Ret>>;

    using CFunctionType = std::conditional_t<
        std::is_function_v<ForcedCFunctionType>,
        ForcedCFunctionType,
        std::conditional_t<
            is_cuj_class<ReturnType> || is_array<ReturnType>,
            void(void *, detail::c_func_arg_t<to_cuj_t<Args>>...),
            to_cuj_t<ReturnType>(
                detail::c_func_arg_t<to_cuj_t<Args>>...)>>;

    using CFunctionPointer = CFunctionType*;

    template<typename...CallArgs>
    typename detail::FuncRetType<ReturnType>::Type
        operator()(const CallArgs &...args) const;

    std::string get_name() const;

    int get_index() const;

    template<typename Callable>
    void define(Callable &&callable);

private:

    using Impl = FunctionImpl<
        deval_t<to_cuj_t<Ret>>,
        to_cuj_t<Args>...>;

    std::unique_ptr<Impl> impl_;

    friend class Context;

    static void get_arg_types(std::vector<const ir::Type *> &output);

    explicit Function(int func_index);
};

CUJ_NAMESPACE_END(cuj::ast)
