#pragma once

#include <typeindex>

#include <cuj/core/type.h>
#include <cuj/dsl/variable_forward.h>

CUJ_NAMESPACE_BEGIN(cuj::dsl)

class TypeContext
{
public:

    using Type = core::Type;

    explicit TypeContext(RC<core::TypeSet> type_set);

    TypeContext(const TypeContext &) = default;

    TypeContext(TypeContext &&) noexcept = default;

    TypeContext &operator=(const TypeContext &) = default;

    TypeContext &operator=(TypeContext &&) noexcept = default;

    template<typename T> requires is_cuj_var_v<T>
    const Type *get_type();

    std::type_index get_type_index(const core::Type *type) const;

    auto &get_all_types() const { return type_set_->index_to_type; }

    auto get_type_set() const { return type_set_; }

private:

    RC<core::TypeSet> type_set_;
};

CUJ_NAMESPACE_END(cuj::dsl)
