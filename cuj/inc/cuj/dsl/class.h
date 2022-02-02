#pragma once

#include <cuj/dsl/function.h>
#include <cuj/dsl/variable_forward.h>
#include <cuj/utils/macro_foreach.h>

CUJ_NAMESPACE_BEGIN(cuj::dsl)

namespace class_detail
{

    struct ClassInternalConstructorTag { };

} // namespace class_detail

template<typename T> requires is_cuj_class_v<T>
class ref<T> : public T
{
    using typename T::CujClassTag;

    explicit ref(const ptr<T> &addr)
        : T(class_detail::ClassInternalConstructorTag{}, addr)
    {
        static_assert(is_cuj_class_v<T>);
    }

public:

    ref(const T &class_object)
        : ref(class_object.address())
    {
        static_assert(is_cuj_class_v<T>);
    }

    ref(const ref &other)
        : ref(other.address())
    {
        
    }

    static ref _from_ptr(const ptr<T> &ptr)
    {
        static_assert(is_cuj_class_v<T>);
        return ref(ptr);
    }
};

namespace class_detail
{
    template<int N>
    struct Int : Int<N - 1> { };

    template<>
    struct Int<0> { };

    template<int N>
    struct Sizer { char _data[N]; };

    static_assert(sizeof(Sizer<17>) == 17);

    template<typename T>
    struct MemberVariableTypeAux { };

    template<typename C, typename M>
    struct MemberVariableTypeAux<M C::*>
    {
        using Type = M;
    };

    template<typename T>
    using pointer_to_member_t = typename MemberVariableTypeAux<T>::Type;

    template<typename T>
    using pointer_to_variable_t = cxx_to_cuj_t<pointer_to_member_t<T>>;

    template<typename T>
    using pointer_to_reference_t = add_reference_t<pointer_to_variable_t<T>>;

    template<typename C, typename M>
    ptr<M> class_pointer_to_member_ptr(
        const ptr<C> &class_ptr, size_t member_index);

    template<typename C>
    ptr<C> alloc_local_var();

#define CUJ_MAKE_REFLECTION_MEMBER_INFO(INDEX, CLASS, MEMBER)                   \
    template<>                                                                  \
    struct MemberInfo<INDEX - 1>                                                \
    {                                                                           \
        using MemberPtr = decltype(&CLASS::MEMBER);                             \
        using Variable =                                                        \
            ::cuj::dsl::class_detail::pointer_to_variable_t<MemberPtr>;         \
        using Reference =                                                       \
            ::cuj::dsl::class_detail::pointer_to_reference_t<MemberPtr>;        \
        template<typename F>                                                    \
        static void process(const F &f)                                         \
        {                                                                       \
            f.template operator()<MemberInfo<INDEX - 1>::Variable>();           \
        }                                                                       \
    };

#define CUJ_MAKE_PROXY_BASE_MEMBER(INDEX, CLASS, MEMBER)                        \
    Reflection::MemberInfo<INDEX - 1>::Reference MEMBER =                       \
        Reflection::MemberInfo<INDEX - 1>::Reference::_from_ptr(                \
            ::cuj::dsl::class_detail::class_pointer_to_member_ptr<              \
                ::cuj::dsl::cxx_to_cuj_t<CLASS>,                                \
                Reflection::MemberInfo<INDEX - 1>::Variable>(                   \
                    cuj_class_object_address_, INDEX - 1));

#define CUJ_ASSIGN_MEMBER_FROM_OTHER(MEMBER) this->MEMBER = other.MEMBER;

#define CUJ_CLASS(CLASS, ...)                                                   \
    CUJ_PROXY_CLASS(CujProxy##CLASS, CLASS, __VA_ARGS__)
    
#define CUJ_CLASS_EX(CLASS, ...)                                                \
    CUJ_PROXY_CLASS_EX(CujProxy##CLASS, CLASS, __VA_ARGS__)

#define CUJ_PROXY_CLASS(PROXY, CLASS, ...)                                      \
    CUJ_PROXY_CLASS_EX(PROXY, CLASS, __VA_ARGS__)                               \
    { CUJ_BASE_CONSTRUCTORS }

    template<typename Reflection, int...Is>
    constexpr bool are_all_members_trivially_copyable(
        std::integer_sequence<int, Is...>)
    {
        return ((is_trivially_copyable_v<
            typename Reflection::template MemberInfo<Is>::Variable>) && ...);
    }

    template<typename Reflection>
    constexpr bool are_all_members_trivially_copyable()
    {
        return are_all_members_trivially_copyable<Reflection>(
            std::make_integer_sequence<int, Reflection::MemberCount>());
    }

#define CUJ_PROXY_CLASS_EX(PROXY, CLASS, ...)                                   \
    struct PROXY;                                                               \
    inline PROXY *_cxx_class_to_cuj_class(const CLASS *) { return nullptr; }    \
    namespace CujReflectionNamespace##PROXY                                     \
    {                                                                           \
        template<int N>                                                         \
        struct MemberInfo;                                                      \
        CUJ_MACRO_FOREACH_INDEXED_2(                                            \
            CUJ_MAKE_REFLECTION_MEMBER_INFO, CLASS, __VA_ARGS__)                \
    }                                                                           \
    struct CujReflection##PROXY                                                 \
    {                                                                           \
        static constexpr int MemberCount =                                      \
            CUJ_MACRO_OVERLOADING_COUNT_ARGS(__VA_ARGS__);                      \
        template<int N>                                                         \
        using MemberInfo = CujReflectionNamespace##PROXY::MemberInfo<N>;        \
    };                                                                          \
    struct CujBase##PROXY                                                       \
    {                                                                           \
        using CXXClass = CLASS;                                                 \
        using CujBase = CujBase##PROXY;                                         \
        using Reflection = CujReflection##PROXY;                                \
        static constexpr size_t CujClassAlignment = 0;                          \
        static constexpr bool all_members_trivially_copyable = ::cuj::dsl       \
            ::class_detail::are_all_members_trivially_copyable<Reflection>();   \
        struct CujClassTag { };                                                 \
        CujBase##PROXY() : CujBase##PROXY(                                      \
            ::cuj::dsl::class_detail::ClassInternalConstructorTag{},            \
            ::cuj::dsl::class_detail::alloc_local_var<PROXY>()) { }             \
        CujBase##PROXY(                                                         \
            ::cuj::dsl::class_detail::ClassInternalConstructorTag,              \
            const ::cuj::dsl::ptr<PROXY> &ptr)                                  \
                : cuj_class_object_address_(ptr) { }                            \
        CujBase##PROXY(const CujBase##PROXY &other)                             \
            : CujBase##PROXY()                                                  \
        {                                                                       \
            *this = other;                                                      \
        }                                                                       \
        CujBase##PROXY &operator=(const CujBase##PROXY &other)                  \
        {                                                                       \
            if constexpr(!all_members_trivially_copyable)                       \
            {                                                                   \
                CUJ_MACRO_FOREACH_1(CUJ_ASSIGN_MEMBER_FROM_OTHER, __VA_ARGS__)  \
            }                                                                   \
            else                                                                \
            {                                                                   \
                ::cuj::dsl::FunctionContext::get_func_context()                 \
                    ->append_statement(::cuj::core::Copy{                       \
                        .dst_addr = cuj_class_object_address_._load(),          \
                        .src_addr = other.cuj_class_object_address_._load()     \
                    });                                                         \
            }                                                                   \
            return *this;                                                       \
        }                                                                       \
        CujBase##PROXY(CujBase##PROXY &&) noexcept = delete;                    \
        CujBase##PROXY &operator=(CujBase##PROXY &&) noexcept = delete;         \
        auto address() const { return cuj_class_object_address_; }              \
        template<typename F>                                                    \
        static void foreach_member(const F &f)                                  \
        {                                                                       \
            ::cuj::dsl::class_detail::foreach_member<PROXY>(f);                 \
        }                                                                       \
        ::cuj::dsl::ptr<PROXY> cuj_class_object_address_;                       \
        CUJ_MACRO_FOREACH_INDEXED_2(                                            \
            CUJ_MAKE_PROXY_BASE_MEMBER, CLASS, __VA_ARGS__)                     \
    };                                                                          \
    struct PROXY : CujBase##PROXY

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

#define CUJ_BASE_CONSTRUCTORS using CujBase::CujBase;
#define CUJ_NONE_TRIVIALLY_COPYABLE struct NoneTriviallyCopyableTag { };
#define CUJ_CLASS_ALIGNMENT(Align) static constexpr size_t CujClassAlignment = Align;

} // namespace class_detail

#define CUJ_CLASS_MAX_MEMBER_COUNT 128

#define CUJ_CLASS_BEGIN(CLASS_NAME)                                             \
    class CLASS_NAME;                                                           \
    class CujBase##CLASS_NAME                                                   \
    {                                                                           \
    public:                                                                     \
        static constexpr size_t CujClassAlignment = 0;                          \
        ::cuj::dsl::ptr<CLASS_NAME> cuj_class_object_address_;                  \
        CujBase##CLASS_NAME(                                                    \
            ::cuj::dsl::class_detail::ClassInternalConstructorTag,              \
            const ::cuj::dsl::ptr<CLASS_NAME> &addr)                            \
            : cuj_class_object_address_(addr) { }                               \
        CujBase##CLASS_NAME()                                                   \
            : CujBase##CLASS_NAME(                                              \
                ::cuj::dsl::class_detail::ClassInternalConstructorTag{},        \
                ::cuj::dsl::class_detail::alloc_local_var<CLASS_NAME>()) { }    \
        CujBase##CLASS_NAME(const CujBase##CLASS_NAME &) = delete;              \
        CujBase##CLASS_NAME &operator=(const CujBase##CLASS_NAME &) = delete;   \
        auto address() const { return cuj_class_object_address_; }              \
    };                                                                          \
    class CLASS_NAME : public CujBase##CLASS_NAME                               \
    {                                                                           \
    public:                                                                     \
        struct CujClassTag { };                                                 \
        using CujBase##CLASS_NAME::CujBase##CLASS_NAME;                         \
        using CujReflectionSelf = CLASS_NAME;                                   \
        static ::cuj::dsl::class_detail::Sizer<1>                               \
            _cuj_refl_mem_var_counter(...);                                     \
        template<int N>                                                         \
        struct MemberInfo                                                       \
        {                                                                       \
            static constexpr int exists = 0;                                    \
            template<typename F>                                                \
            static void process(const F &f) { }                                 \
        };                                                                      \
        CLASS_NAME(const CLASS_NAME &other)                                     \
        {                                                                       \
            *this = other;                                                      \
        }                                                                       \
        CLASS_NAME &operator=(const CLASS_NAME &other)                          \
        {                                                                       \
            ::cuj::dsl::FunctionContext::get_func_context()                     \
                    ->append_statement(::cuj::core::Copy{                       \
                        .dst_addr = cuj_class_object_address_._load(),          \
                        .src_addr = other.cuj_class_object_address_._load()     \
                    });                                                         \
            return *this;                                                       \
        }                                                                       \
        template<typename F>                                                    \
        static void foreach_member(const F &f)                                  \
        {                                                                       \
            ::cuj::dsl::class_detail::foreach_member_2<CujReflectionSelf>(f);   \
        }                                                                       \

#define CUJ_MEMBER_VARIABLE(TYPE, NAME)                                         \
    static constexpr int cuj_member_counter##NAME =                             \
        sizeof(_cuj_refl_mem_var_counter(                                       \
            (::cuj::dsl::class_detail::Int<                                     \
                CUJ_CLASS_MAX_MEMBER_COUNT>*)nullptr)) - 1;                     \
    static ::cuj::dsl::class_detail::Sizer<cuj_member_counter##NAME + 2>        \
        _cuj_refl_mem_var_counter(                                              \
            ::cuj::dsl::class_detail::Int<                                      \
                cuj_member_counter##NAME + 2>*);                                \
    template<>                                                                  \
    struct MemberInfo<cuj_member_counter##NAME>                                 \
    {                                                                           \
        static constexpr bool exists = true;                                    \
        using Variable = TYPE;                                                  \
        using Reference = ::cuj::dsl::add_reference_t<Variable>;                \
        template<typename F>                                                    \
        static void process(const F &f)                                         \
        {                                                                       \
            f.template operator()<Variable>();                                  \
        }                                                                       \
    };                                                                          \
    ::cuj::dsl::add_reference_t<MemberInfo<                                     \
        cuj_member_counter##NAME>::Variable> NAME =                             \
            MemberInfo<cuj_member_counter##NAME>::Reference::_from_ptr(         \
                ::cuj::dsl::class_detail::class_pointer_to_member_ptr<          \
                    CujReflectionSelf,                                          \
                    MemberInfo<cuj_member_counter##NAME>::Variable>(            \
                        cuj_class_object_address_, cuj_member_counter##NAME));

#define CUJ_CLASS_END };

namespace class_detail
{

    template<typename T, typename F, int...Is>
    void foreach_member_2_aux(const F &f, std::integer_sequence<int, Is...>)
    {
        (T::template MemberInfo<Is>::template process<F>(f), ...);
    }

    template<typename T, typename F>
    void foreach_member_2(const F &f)
    {
        foreach_member_2_aux<T>(
            f, std::make_integer_sequence<int, CUJ_CLASS_MAX_MEMBER_COUNT>());
    }

} // namespace class_detail

CUJ_NAMESPACE_END(cuj::dsl)
