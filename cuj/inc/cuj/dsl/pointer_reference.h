#pragma once

#include <cuj/dsl/pointer.h>
#include <cuj/dsl/variable_forward.h>

CUJ_NAMESPACE_BEGIN(cuj::dsl)

template<typename T>
class ref<ptr<T>>
{
    ptr<ptr<T>> addr_;

    ref() = default;

public:

    using PointedType = T;

    ref(const ptr<T> &ptr);

    ref(const ref &ref);

    ref(ref &&other) noexcept;

    ref &operator=(const ref &other);

    ref &operator=(const ptr<T> &other);

    template<typename U> requires std::is_integral_v<U>
    ptr<T> operator+(const num<U> &rhs) const;

    template<typename U> requires std::is_integral_v<U>
    ptr<T> operator-(const num<U> &rhs) const;

    template<typename U> requires std::is_integral_v<U>
    add_reference_t<T> operator[](const num<U> &rhs) const;

    template<typename U> requires std::is_integral_v<U>
    ptr<T> operator+(U rhs) const;

    template<typename U> requires std::is_integral_v<U>
    ptr<T> operator-(U rhs) const;

    template<typename U> requires std::is_integral_v<U>
    add_reference_t<T> operator[](U rhs) const;
    
    template<typename U> requires std::is_integral_v<U>
    ptr<T> operator+(const ref<num<U>> &rhs) const;

    template<typename U> requires std::is_integral_v<U>
    ptr<T> operator-(const ref<num<U>> &rhs) const;

    template<typename U> requires std::is_integral_v<U>
    add_reference_t<T> operator[](const ref<num<U>> &rhs) const;

    ptr<ptr<T>> address() const;

    add_reference_t<T> deref() const;

    add_reference_t<T> operator*() const;

    add_reference_t<T> *operator->() const;

    core::Load _load() const;

    static ref _from_ptr(const ptr<ptr<T>> &ptr);
};

template<typename U, typename T> requires std::is_integral_v<U>
ptr<T> operator+(const num<U> &lhs, const ref<ptr<T>> &rhs);

template<typename U, typename T> requires std::is_integral_v<U>
ptr<T> operator+(const ref<num<U>> &lhs, const ref<ptr<T>> &rhs);

template<typename T>
num<bool> operator==(const ref<ptr<T>> &lhs, const ref<ptr<T>> &rhs);

template<typename T>
num<bool> operator!=(const ref<ptr<T>> &lhs, const ref<ptr<T>> &rhs);

CUJ_NAMESPACE_END(cuj::dsl)
