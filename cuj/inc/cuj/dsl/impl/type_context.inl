#pragma once

#include <cuj/dsl/type_context.h>
#include <cuj/dsl/variable_forward.h>

CUJ_NAMESPACE_BEGIN(cuj::dsl)

inline TypeContext::TypeContext(RC<core::TypeSet> type_set)
    : type_set_(std::move(type_set))
{

}

template<typename T> requires is_cuj_var_v<T>
const TypeContext::Type *TypeContext::get_type()
{
    const auto idx = std::type_index(typeid(T));
    auto it = type_set_->index_to_type.find(idx);
    if(it != type_set_->index_to_type.end())
        return it->second.get();

    it = type_set_->index_to_type.insert({ idx, newRC<core::Type>() }).first;
    type_set_->type_to_index.insert({ it->second.get(), idx });

    if constexpr(is_cuj_arithmetic_v<T>)
        *it->second = core::arithmetic_to_builtin_v<typename T::RawType>;

    if constexpr(std::is_same_v<T, CujVoid>)
        *it->second = core::Builtin::Void;

    if constexpr(is_cuj_pointer_v<T>)
    {
        auto pointed = get_type<typename T::PointedType>();
        *it->second = core::Pointer{ pointed };
    }

    if constexpr(is_cuj_array_v<T>)
    {
        auto element = get_type<typename T::ElementType>();
        *it->second = core::Array{ element, T::ElementCount };
    }

    if constexpr(is_cuj_class_v<T>)
    {
        std::vector<const core::Type *> members;
        T::foreach_member([&]<typename M>
        {
            members.push_back(get_type<M>());
        });
        const size_t alignment = T::CujClassAlignment;
        *it->second = core::Struct{ std::move(members), alignment };
    }

    assert(it != type_set_->index_to_type.end());
    return it->second.get();
}

inline std::type_index TypeContext::get_type_index(const core::Type *type) const
{
    return type_set_->type_to_index.at(type);
}

CUJ_NAMESPACE_END(cuj::dsl)
