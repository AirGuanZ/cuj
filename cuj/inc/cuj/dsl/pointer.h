#pragma once

#include <cstddef>

#include <cuj/core/expr.h>
#include <cuj/dsl/variable_forward.h>

CUJ_NAMESPACE_BEGIN(cuj::dsl)

template<typename T>
class ptr
{
    size_t alloc_index_;

    static const core::Type *type();

public:

    using PointedType = T;

    ptr();

    ptr(std::nullptr_t);

    ptr(const ref<ptr<T>> &ref);

    ptr(const ptr &other);

    ptr(ptr &&other) noexcept;

    ptr &operator=(const ptr &other);

    template<typename U> requires std::is_integral_v<U>
    ptr operator+(const num<U> &rhs) const;

    template<typename U> requires std::is_integral_v<U>
    ptr operator-(const num<U> &rhs) const;

    template<typename U> requires std::is_integral_v<U>
    add_reference_t<PointedType> operator[](const num<U> &rhs) const;

    template<typename U> requires std::is_integral_v<U>
    ptr operator+(U rhs) const;

    template<typename U> requires std::is_integral_v<U>
    ptr operator-(U rhs) const;

    template<typename U> requires std::is_integral_v<U>
    add_reference_t<PointedType> operator[](U rhs) const;

    template<typename U> requires std::is_integral_v<U>
    ptr operator+(const ref<num<U>> &rhs) const;

    template<typename U> requires std::is_integral_v<U>
    ptr operator-(const ref<num<U>> &rhs) const;

    template<typename U> requires std::is_integral_v<U>
    add_reference_t<PointedType> operator[](const ref<num<U>> &rhs) const;

    ptr<ptr> address() const;

    add_reference_t<T> deref() const;

    add_reference_t<T> operator*() const;

    add_reference_t<T> *operator->() const;

    static ptr _from_expr(core::Expr expr);

    core::LocalAllocAddr _addr() const;

    core::Load _load() const;
};

template<typename T> requires std::is_same_v<T, std::nullptr_t>
ptr(T)->ptr<CujVoid>;

template<typename U, typename T> requires std::is_integral_v<U>
ptr<T> operator+(const num<U> &lhs, const ptr<T> &rhs);

template<typename U, typename T> requires std::is_integral_v<U>
ptr<T> operator+(U lhs, const ptr<T> &rhs);

template<typename U, typename T> requires std::is_integral_v<U>
ptr<T> operator+(const ref<num<U>> &lhs, const ptr<T> &rhs);

template<typename T>
ptr<cxx<T>> import_pointer(T *pointer);

template<typename T>
ptr<cxx<T>> import_pointer(const T *pointer);

CUJ_NAMESPACE_END(cuj::dsl)
