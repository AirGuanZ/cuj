#pragma once

#include <cuj/dsl/array.h>
#include <cuj/dsl/pointer.h>
#include <cuj/dsl/function.h>

CUJ_NAMESPACE_BEGIN(cuj::dsl)

template<typename T, size_t N>
const core::Type *arr<T, N>::type()
{
    return FunctionContext::get_func_context()
        ->get_type_context()->get_type<arr>();
}

template<typename T, size_t N>
arr<T, N>::arr()
{
    static_assert(is_cuj_var_v<T> && !std::is_same_v<T, CujVoid>);
    auto func_ctx = FunctionContext::get_func_context();
    alloc_index_ = func_ctx->alloc_local_var(type());
}

template<typename T, size_t N>
arr<T, N>::arr(const ref<arr<T, N>> &ref)
    : arr()
{
    for(size_t i = 0; i < N; ++i)
        this->operator[](i) = ref[i];
}

template<typename T, size_t N>
arr<T, N>::arr(const arr &other)
    : arr()
{
    *this = other;
}

template<typename T, size_t N>
arr<T, N>::arr(arr &&other) noexcept
    : alloc_index_(other.alloc_index_)
{
    
}

template<typename T, size_t N>
arr<T, N> &arr<T, N>::operator=(const arr &other)
{
    for(size_t i = 0; i < N; ++i)
        this->operator[](i) = other[i];
    return *this;
}

template<typename T, size_t N>
template<typename U> requires std::is_integral_v<U>
add_reference_t<T> arr<T, N>::operator[](const num<U> &idx) const
{
    auto first_elem_ptr = ptr<T>::_from_expr(_first_elem_addr());
    return *(first_elem_ptr + idx);
}

template<typename T, size_t N>
template<typename U> requires std::is_integral_v<U>
add_reference_t<T> arr<T, N>::operator[](U idx) const
{
    return this->operator[](num(idx));
}

template<typename T, size_t N>
ptr<arr<T, N>> arr<T, N>::address() const
{
    return ptr<arr>::_from_expr(core::LocalAllocAddr{
        .alloc_type  = type(),
        .alloc_index = alloc_index_
    });
}

template<typename T, size_t N>
core::ArrayAddrToFirstElemAddr arr<T, N>::_first_elem_addr() const
{
    auto func_ctx = FunctionContext::get_func_context();
    auto type_ctx = func_ctx->get_type_context();
    auto arr_ptr_type = type_ctx->get_type<ptr<arr>>();
    return core::ArrayAddrToFirstElemAddr{
        .array_ptr_type = arr_ptr_type,
        .array_ptr      = newRC<core::Expr>(core::LocalAllocAddr{
            .alloc_type  = type(),
            .alloc_index = alloc_index_
        })
    };
}

CUJ_NAMESPACE_END(cuj::dsl)
