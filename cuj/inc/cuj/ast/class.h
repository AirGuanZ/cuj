#pragma once

#include <cuj/ast/expr.h>
#include <cuj/util/macro_foreach.h>

CUJ_NAMESPACE_BEGIN(cuj::ast)

namespace class_detail
{

    template<typename T>
    struct MemVarTypeAux { };

    template<typename C, typename M>
    struct MemVarTypeAux<M C::*>
    {
        using Type = M;
    };

    template<typename T>
    using mem_var_t = typename MemVarTypeAux<T>::Type;

    struct Empty {};

    template<typename T>
    auto address_to_member_impl(RC<InternalPointerValue<T>> address)
    {
        if constexpr(is_array<T>)
        {
            auto alloc_addr = newRC<InternalArrayAllocAddress<T>>();
            alloc_addr->arr_alloc = address;

            auto impl = newRC<InternalArrayValue<
                typename T::ElementType, T::ElementCount>>();
            impl->data_ptr = alloc_addr;

            return impl;
        }
        else if constexpr(is_pointer<T>)
        {
            auto impl_value = newRC<InternalArithmeticLeftValue<size_t>>();
            impl_value->address = std::move(address);

            auto impl = newRC<InternalPointerValue<typename T::PointedType>>();
            impl->value = std::move(impl_value);

            return impl;
        }
        else if constexpr(std::is_arithmetic_v<T>)
        {
            auto impl = newRC<InternalArithmeticLeftValue<T>>();
            impl->address = std::move(address);
            return impl;
        }
        else
        {
            return T(std::move(address));
        }
    }

#define CUJ_MAKE_PROXY_BASE(PREFIX, CLASS_NAME, MEMBER_NAME)                    \
    struct PREFIX##MEMBER_NAME                                                  \
    {                                                                           \
        using MemberPtr = decltype(&CLASS_NAME::MEMBER_NAME);                   \
        using Member = ::cuj::ast::deval_t<::cuj::ast::Value<                   \
            ::cuj::ast::class_detail::mem_var_t<MemberPtr>>>;                   \
        using MemberProxy = ::cuj::ast::Value<::cuj::ast::to_cuj_t<Member>>;    \
        explicit PREFIX##MEMBER_NAME(                                           \
            ::cuj::RC<::cuj::ast::InternalPointerValue<Member>> address)        \
            : MEMBER_NAME(::cuj::ast::class_detail::address_to_member_impl(     \
                std::move(address))) { }                                        \
        MemberProxy MEMBER_NAME;                                                \
    };

#define CUJ_INHERIT(A, B) A##B,

#define CUJ_INITIALIZE_PROXY_BASE(MEMBER_INDEX, PREFIX, MEMBER_NAME)            \
    PREFIX##MEMBER_NAME(::cuj::ast::create_member_pointer_offset<               \
            Proxy, PREFIX##MEMBER_NAME::Member>(address, MEMBER_INDEX - 1)),

#define CUJ_CALL_MEMBER_PROCESSOR(PREFIX, MEMBER_NAME) \
    f.template operator()<PREFIX##MEMBER_NAME>();

#define CUJ_MAKE_PROXY_BEGIN(CLASS_NAME, PROXY_NAME, ...)                       \
    CUJ_MACRO_FOREACH_3(                                                        \
        CUJ_MAKE_PROXY_BASE,                                                    \
        CUJProxyBase##PROXY_NAME, CLASS_NAME,                                   \
        __VA_ARGS__)                                                            \
    struct PROXY_NAME : CUJ_MACRO_FOREACH_2(                                    \
        CUJ_INHERIT, CUJProxyBase##PROXY_NAME, __VA_ARGS__)                     \
        ::cuj::ast::class_detail::Empty                                         \
    {                                                                           \
        struct CUJClassFlag { };                                                \
        using Proxy = PROXY_NAME;                                               \
        using Address = ::cuj::RC<::cuj::ast::InternalPointerValue<PROXY_NAME>>;\
        using ImplType = ::cuj::ast::InternalPointerValue<PROXY_NAME>;          \
        Address _cuj_address_;                                                  \
        PROXY_NAME(Address address)                                             \
            : CUJ_MACRO_FOREACH_INDEXED_2(                                      \
                CUJ_INITIALIZE_PROXY_BASE,                                      \
                CUJProxyBase##PROXY_NAME,                                       \
                __VA_ARGS__) Empty{}                                            \
        {                                                                       \
            _cuj_address_ = std::move(address);                                 \
        }                                                                       \
        auto address() const                                                    \
            { return ::cuj::ast::PointerImpl<Proxy>(_cuj_address_); }           \
        auto get_impl() const { return address().get_impl(); }                  \
        template<typename F>                                                    \
        static void foreach_member(const F &f)                                  \
        {                                                                       \
            CUJ_MACRO_FOREACH_2(                                                \
                CUJ_CALL_MEMBER_PROCESSOR,                                      \
                CUJProxyBase##PROXY_NAME,                                       \
                __VA_ARGS__)                                                    \
        }                                                                       \

#define CUJ_MAKE_PROXY_END };

#define CUJ_PROXY_CONSTRUCTOR(PROXY_NAME, ...)                                  \
    PROXY_NAME(__VA_ARGS__) : PROXY_NAME(::cuj::ast::get_current_function()     \
                ->alloc_on_stack<PROXY_NAME>(::cuj::ast::get_current_context()  \
                    ->get_type<PROXY_NAME>()))

} // namespace class_detail

CUJ_NAMESPACE_END(cuj::ast)
