#pragma once

#include <cuj/dsl/pointer.h>
#include <cuj/dsl/variable_forward.h>

CUJ_NAMESPACE_BEGIN(cuj::dsl)

template<typename T, size_t N>
class ref<Array<T, N>>
{
    Pointer<Array<T, N>> addr_;

    ref() = default;

public:

    using ElementType = T;

    static constexpr size_t ElementCount = N;

    ref(const Array<T, N> &var);

    ref(const ref &other);

    ref(ref &&other) noexcept;

    ref &operator=(const ref &other);

    ref &operator=(const Array<T, N> &other);
    
    template<typename U> requires std::is_integral_v<U>
    add_reference_t<T> operator[](const Arithmetic<U> &idx) const;

    template<typename U> requires std::is_integral_v<U>
    add_reference_t<T> operator[](U idx) const;

    Pointer<Array<T, N>> address() const;

    core::ArrayAddrToFirstElemAddr _first_elem_addr() const;

    static ref _from_ptr(const Pointer<Array<T, N>> &ptr);
};

CUJ_NAMESPACE_END(cuj::dsl)
