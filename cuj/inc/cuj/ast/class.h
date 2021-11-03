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

#define CUJ_MAKE_REFLECTION_MEMBER_INFO(INDEX, CLASS, MEMBER)                   \
    template<>                                                                  \
    struct MemberInfo<INDEX - 1>                                                \
    {                                                                           \
        using MemberPtr = decltype(&CLASS::MEMBER);                             \
        using Member = ::cuj::ast::deval_t<::cuj::ast::Value<                   \
            ::cuj::ast::class_detail::mem_var_t<MemberPtr>>>;                   \
        using MemberProxy = ::cuj::ast::Value<::cuj::ast::to_cuj_t<Member>>;    \
        template<typename F>                                                    \
        static void process(const F &f)                                         \
        {                                                                       \
            f.template operator()<MemberInfo<INDEX - 1>>();                     \
        }                                                                       \
    };

#define CUJ_MAKE_PROXY_BASE_MEMBER(INDEX, CLASS, MEMBER)                        \
    Reflection::MemberInfo<INDEX - 1>::MemberProxy MEMBER =                     \
        ::cuj::ast::class_detail::address_to_member_impl(                       \
            ::cuj::ast::create_member_pointer_offset<                           \
                Proxy, Reflection::MemberInfo<INDEX - 1>::Member>(              \
                    cuj_address_, INDEX - 1));

#define CUJ_ASSIGN_MEMBER_FROM_OTHER(MEMBER) this->MEMBER = other.MEMBER;

#define CUJ_CLASS(CLASS, ...) \
    CUJ_PROXY_CLASS(CUJProxy##CLASS, CLASS, __VA_ARGS__)

#define CUJ_PROXY_CLASS(PROXY, CLASS, ...)                                      \
    struct PROXY;                                                               \
    struct CUJReflection##PROXY                                                 \
    {                                                                           \
        static constexpr int MemberCount =                                      \
            CUJ_MACRO_OVERLOADING_COUNT_ARGS(__VA_ARGS__);                      \
        template<int N> struct MemberInfo;                                      \
        CUJ_MACRO_FOREACH_INDEXED_2(                                            \
            CUJ_MAKE_REFLECTION_MEMBER_INFO, CLASS, __VA_ARGS__)                \
    };                                                                          \
    inline PROXY *_cuj_class_to_proxy_aux(const CLASS *) { return nullptr; }    \
    class CUJBase##PROXY                                                        \
    {                                                                           \
    public:                                                                     \
        using CUJBase = CUJBase##PROXY;                                         \
        using Proxy = PROXY;                                                    \
        using Reflection = CUJReflection##PROXY;                                \
        using Address = ::cuj::RC<::cuj::ast::InternalPointerValue<PROXY>>;     \
        using VariableType = PROXY;                                             \
        using ImplType = ::cuj::ast::InternalPointerValue<PROXY>;               \
        struct CUJClassFlag { };                                                \
        CUJBase##PROXY() : CUJBase##PROXY(                                      \
                ::cuj::ast::get_current_function()->alloc_on_stack<PROXY>(      \
                    ::cuj::ast::get_current_context()->get_type<PROXY>())) { }  \
        CUJBase##PROXY(Address address)                                         \
            : cuj_address_(std::move(address)) { }                              \
        CUJBase##PROXY(const CUJBase##PROXY &other)                             \
            : CUJBase##PROXY()                                                  \
        {                                                                       \
            CUJ_MACRO_FOREACH_1(CUJ_ASSIGN_MEMBER_FROM_OTHER, __VA_ARGS__)      \
        }                                                                       \
        CUJBase##PROXY &operator=(const CUJBase##PROXY &other)                  \
        {                                                                       \
            CUJ_MACRO_FOREACH_1(CUJ_ASSIGN_MEMBER_FROM_OTHER, __VA_ARGS__)      \
            return *this;                                                       \
        }                                                                       \
        CUJBase##PROXY(CUJBase##PROXY &&) noexcept = delete;                    \
        CUJBase##PROXY &operator=(CUJBase##PROXY &&) noexcept = delete;         \
        auto address() const                                                    \
            { return ::cuj::ast::PointerImpl<PROXY>(cuj_address_); }            \
        auto get_impl() const { return address().get_impl(); }                  \
        template<typename F>                                                    \
        static void foreach_member(const F &f)                                  \
        {                                                                       \
            ::cuj::ast::class_detail::foreach_member<PROXY>(f);                 \
        }                                                                       \
        Address cuj_address_;                                                   \
        CUJ_MACRO_FOREACH_INDEXED_2(                                            \
            CUJ_MAKE_PROXY_BASE_MEMBER, CLASS, __VA_ARGS__)                     \
    };                                                                          \
    struct PROXY : public CUJBase##PROXY

    template<typename T, typename F, int...Is>
    void foreach_member_aux(const F &f, std::integer_sequence<int, Is...>)
    {
        ((T::Reflection::template MemberInfo<Is>::process(f)), ...);
    }

    template<typename T, typename F>
    void foreach_member(const F &f)
    {
        foreach_member_aux<T>(
            f, std::make_integer_sequence<int, T::Reflection::MemberCount>());
    }

} // namespace class_detail

CUJ_NAMESPACE_END(cuj::ast)
