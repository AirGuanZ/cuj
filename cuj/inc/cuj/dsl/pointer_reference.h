#pragma once

#include <cuj/dsl/pointer.h>
#include <cuj/dsl/variable_forward.h>

CUJ_NAMESPACE_BEGIN(cuj::dsl)

template<typename T>
class ref<Pointer<T>>
{
    Pointer<Pointer<T>> addr_;

    ref() = default;

public:

    using PointedType = T;

    ref(const Pointer<T> &ptr);

    ref(const ref &ref);

    ref(ref &&other) noexcept;

    ref &operator=(const ref &other);

    ref &operator=(const Pointer<T> &other);

    template<typename U> requires std::is_integral_v<U>
    Pointer<T> operator+(const Arithmetic<U> &rhs) const;

    template<typename U> requires std::is_integral_v<U>
    Pointer<T> operator-(const Arithmetic<U> &rhs) const;

    template<typename U> requires std::is_integral_v<U>
    add_reference_t<T> operator[](const Arithmetic<U> &rhs) const;

    template<typename U> requires std::is_integral_v<U>
    Pointer<T> operator+(U rhs) const;

    template<typename U> requires std::is_integral_v<U>
    Pointer<T> operator-(U rhs) const;

    template<typename U> requires std::is_integral_v<U>
    add_reference_t<T> operator[](U rhs) const;
    
    template<typename U> requires std::is_integral_v<U>
    Pointer<T> operator+(const ref<Arithmetic<U>> &rhs) const;

    template<typename U> requires std::is_integral_v<U>
    Pointer<T> operator-(const ref<Arithmetic<U>> &rhs) const;

    template<typename U> requires std::is_integral_v<U>
    add_reference_t<T> operator[](const ref<Arithmetic<U>> &rhs) const;

    Pointer<Pointer<T>> address() const;

    add_reference_t<T> deref() const;

    add_reference_t<T> operator*() const;

    add_reference_t<T> *operator->() const;

    core::Load _load() const;

    static ref _from_ptr(const Pointer<Pointer<T>> &ptr);
};

template<typename U, typename T> requires std::is_integral_v<U>
Pointer<T> operator+(const Arithmetic<U> &lhs, const ref<Pointer<T>> &rhs);

template<typename U, typename T> requires std::is_integral_v<U>
Pointer<T> operator+(const ref<Arithmetic<U>> &lhs, const ref<Pointer<T>> &rhs);

CUJ_NAMESPACE_END(cuj::dsl)
