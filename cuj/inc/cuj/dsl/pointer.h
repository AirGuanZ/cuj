#pragma once

#include <cstddef>

#include <cuj/core/expr.h>
#include <cuj/dsl/variable_forward.h>

CUJ_NAMESPACE_BEGIN(cuj::dsl)

template<typename T>
class Pointer
{
    size_t alloc_index_;

    static const core::Type *type();

public:

    using PointedType = T;

    Pointer();

    Pointer(std::nullptr_t);

    Pointer(const ref<Pointer<T>> &ref);

    Pointer(const Pointer &other);

    Pointer(Pointer &&other) noexcept;

    Pointer &operator=(const Pointer &other);

    template<typename U> requires std::is_integral_v<U>
    Pointer operator+(const Arithmetic<U> &rhs) const;

    template<typename U> requires std::is_integral_v<U>
    Pointer operator-(const Arithmetic<U> &rhs) const;

    template<typename U> requires std::is_integral_v<U>
    add_reference_t<PointedType> operator[](const Arithmetic<U> &rhs) const;

    template<typename U> requires std::is_integral_v<U>
    Pointer operator+(U rhs) const;

    template<typename U> requires std::is_integral_v<U>
    Pointer operator-(U rhs) const;

    template<typename U> requires std::is_integral_v<U>
    add_reference_t<PointedType> operator[](U rhs) const;

    template<typename U> requires std::is_integral_v<U>
    Pointer operator+(const ref<Arithmetic<U>> &rhs) const;

    template<typename U> requires std::is_integral_v<U>
    Pointer operator-(const ref<Arithmetic<U>> &rhs) const;

    template<typename U> requires std::is_integral_v<U>
    add_reference_t<PointedType> operator[](const ref<Arithmetic<U>> &rhs) const;

    Pointer<Pointer> address() const;

    add_reference_t<T> deref() const;

    add_reference_t<T> operator*() const;

    add_reference_t<T> *operator->() const;

    static Pointer _from_expr(core::Expr expr);

    core::LocalAllocAddr _addr() const;

    core::Load _load() const;
};

template<typename T> requires std::is_same_v<T, std::nullptr_t>
Pointer(T)->Pointer<CujVoid>;

template<typename U, typename T> requires std::is_integral_v<U>
Pointer<T> operator+(const Arithmetic<U> &lhs, const Pointer<T> &rhs);

template<typename U, typename T> requires std::is_integral_v<U>
Pointer<T> operator+(U lhs, const Pointer<T> &rhs);

template<typename U, typename T> requires std::is_integral_v<U>
Pointer<T> operator+(const ref<Arithmetic<U>> &lhs, const Pointer<T> &rhs);

CUJ_NAMESPACE_END(cuj::dsl)
