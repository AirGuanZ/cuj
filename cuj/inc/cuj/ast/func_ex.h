#pragma once

#include <functional>

#include <cuj/ast/expr.h>

CUJ_NAMESPACE_BEGIN(cuj::ast)

namespace func_ex_detail
{

    template<typename T>
    auto underlying_func_decl_type_aux()
    {
        if constexpr(is_pointer<T> || is_array<T> || is_cuj_class<T>)
            return static_cast<T *>(nullptr);
        else
            return static_cast<typename T::ArithmeticType *>(nullptr);
    }

    template<typename T>
    using UnderlyingFuncDeclType = std::remove_pointer_t<
        decltype(underlying_func_decl_type_aux<T>())>;

    template<typename T>
    struct FuncTrait
    {
        using Ret = typename FuncTrait<rm_cvref_t<decltype(
            std::function{ std::declval<rm_cvref_t<T>>() })>>::Ret;

        using ArgTuple = typename FuncTrait<rm_cvref_t<decltype(
            std::function{ std::declval<rm_cvref_t<T>>() })>>::ArgTuple;
    };

    template<typename R, typename...Args>
    struct FuncTrait<std::function<R(Args...)>>
    {
        using Ret = UnderlyingFuncDeclType<rm_cvref_t<R>>;

        using ArgTuple = std::tuple<UnderlyingFuncDeclType<rm_cvref_t<Args>>...>;
    };

} // namespace func_ex_detail

template<typename Signature>
class FunctionEx
{
public:

    // let R(A...) = Signature
    // transform R(A...) to:
    //      if R == void || is_arithmetic<R> || is_pointer<R>
    //          R -> R
    //      else
    //          R -> void
    //          add arg R*
    //      for each A:
    //          if is_arithmetic<A> || is_pointer<A>
    //              A -> A
    //          else
    //              A -> A*
    // call:
    //      if R == void || is_arithmetic<R> || is_pointer<R>
    //          return
    //      else
    //          R r
    //          call with r.addr()
    //          return r
    //      for each A:
    //          


};

CUJ_NAMESPACE_END(cuj::ast)
