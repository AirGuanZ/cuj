#pragma once

#include <cuj/dsl/array_reference.h>
#include <cuj/dsl/function.h>

CUJ_NAMESPACE_BEGIN(cuj::dsl)

template<typename T, size_t N>
ref<arr<T, N>>::ref(const arr<T, N> &var)
    : addr_(var.address())
{
    
}

template<typename T, size_t N>
ref<arr<T, N>>::ref(const ref &other)
    : addr_(other.addr_)
{
    
}

template<typename T, size_t N>
ref<arr<T, N>>::ref(ref &&other) noexcept
    : addr_(std::move(other.addr_))
{
    
}

template<typename T, size_t N>
ref<arr<T, N>> &ref<arr<T, N>>::operator=(const ref &other)
{
    for(size_t i = 0; i < N; ++i)
        this->operator[](i) = other[i];
    return *this;
}

template<typename T, size_t N>
ref<arr<T, N>> &ref<arr<T, N>>::operator=(const arr<T, N> &other)
{
    for(size_t i = 0; i < N; ++i)
        this->operator[](i) = other[i];
    return *this;
}

template<typename T, size_t N>
template<typename U> requires std::is_integral_v<U>
add_reference_t<T> ref<arr<T, N>>::operator[](const num<U> &idx) const
{
    auto first_elem_ptr = ptr<T>::_from_expr(_first_elem_addr());
    return *(first_elem_ptr + idx);
}

template<typename T, size_t N>
template<typename U> requires std::is_integral_v<U>
add_reference_t<T> ref<arr<T, N>>::operator[](U idx) const
{
    return this->operator[](num(idx));
}

template<typename T, size_t N>
ptr<arr<T, N>> ref<arr<T, N>>::address() const
{
    return addr_;
}

template<typename T, size_t N>
core::ArrayAddrToFirstElemAddr ref<arr<T, N>>::_first_elem_addr() const
{
    auto func_ctx = FunctionContext::get_func_context();
    auto type_ctx = func_ctx->get_type_context();
    auto arr_ptr_type = type_ctx->get_type<ptr<arr<T, N>>>();
    return core::ArrayAddrToFirstElemAddr{
        .array_ptr_type = arr_ptr_type,
        .array_ptr      = newRC<core::Expr>(addr_._load())
    };
}

template<typename T, size_t N>
ref<arr<T, N>> ref<arr<T, N>>::_from_ptr(const ptr<arr<T, N>> &ptr)
{
    ref ret;
    ret.addr_ = ptr;
    return ret;
}

CUJ_NAMESPACE_END(cuj::dsl)
