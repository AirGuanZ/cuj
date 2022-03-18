#pragma once

#include <cuj/dsl/pointer.h>
#include <cuj/dsl/variable_forward.h>

CUJ_NAMESPACE_BEGIN(cuj::dsl)

template<typename T, size_t N>
class ref<arr<T, N>>
{
    ptr<arr<T, N>> addr_;

    ref() = default;

public:

    using ElementType = T;

    static constexpr size_t ElementCount = N;

    ref(const arr<T, N> &var);

    ref(const ref &other);

    ref(ref &&other) noexcept;

    ref &operator=(const ref &other);

    ref &operator=(const arr<T, N> &other);

    constexpr size_t size() const { return N; }
    
    template<typename U> requires std::is_integral_v<U>
    add_reference_t<T> operator[](const num<U> &idx) const;

    template<typename U> requires std::is_integral_v<U>
    add_reference_t<T> operator[](const ref<num<U>> &idx) const;

    template<typename U> requires std::is_integral_v<U>
    add_reference_t<T> operator[](U idx) const;

    ptr<arr<T, N>> address() const;

    core::ArrayAddrToFirstElemAddr _first_elem_addr() const;

    static ref _from_ptr(const ptr<arr<T, N>> &ptr);
};

CUJ_NAMESPACE_END(cuj::dsl)
