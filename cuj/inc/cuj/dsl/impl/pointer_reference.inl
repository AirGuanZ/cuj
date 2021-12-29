#pragma once

#include <cuj/core/stat.h>
#include <cuj/dsl/function.h>
#include <cuj/dsl/pointer_reference.h>
#include <cuj/dsl/pointer_temp_var.h>

CUJ_NAMESPACE_BEGIN(cuj::dsl)

template<typename T>
ref<Pointer<T>>::ref(const Pointer<T> &ptr)
{
    addr_ = ptr.address();
}

template<typename T>
ref<Pointer<T>>::ref(const ref &ref)
{
    addr_ = ref.address();
}

template<typename T>
ref<Pointer<T>>::ref(ref &&other) noexcept
    : addr_(std::move(other.addr_))
{
    
}

template<typename T>
ref<Pointer<T>> &ref<Pointer<T>>::operator=(const ref &other)
{
    core::Store store = {
        .dst_addr = addr_._load(),
        .val      = other._load()
    };
    auto func_ctx = FunctionContext::get_func_context();
    func_ctx->append_statement(std::move(store));
    return *this;
}

template<typename T>
ref<Pointer<T>> &ref<Pointer<T>>::operator=(const Pointer<T> &other)
{
    core::Store store = {
        .dst_addr = addr_._load(),
        .val      = other._load()
    };
    auto func_ctx = FunctionContext::get_func_context();
    func_ctx->append_statement(std::move(store));
    return *this;
}

template<typename T>
template<typename U> requires std::is_integral_v<U>
Pointer<T> ref<Pointer<T>>::operator+(const Arithmetic<U> &rhs) const
{
    static_assert(!std::is_same_v<T, CujVoid>);
    return static_cast<const Pointer<T>&>(*this) + rhs;
}

template<typename T>
template<typename U> requires std::is_integral_v<U>
Pointer<T> ref<Pointer<T>>::operator-(const Arithmetic<U> &rhs) const
{
    static_assert(!std::is_same_v<T, CujVoid>);
    return static_cast<const Pointer<T>&>(*this) - rhs;
}

template<typename T>
template<typename U> requires std::is_integral_v<U>
add_reference_t<T> ref<Pointer<T>>::operator[](const Arithmetic<U> &rhs) const
{
    return *(*this + rhs);
}

template<typename T>
template<typename U> requires std::is_integral_v<U>
Pointer<T> ref<Pointer<T>>::operator+(U rhs) const
{
    return *this + Arithmetic(rhs);
}

template<typename T>
template<typename U> requires std::is_integral_v<U>
Pointer<T> ref<Pointer<T>>::operator-(U rhs) const
{
    return *this - Arithmetic(rhs);
}

template<typename T>
template<typename U> requires std::is_integral_v<U>
add_reference_t<T> ref<Pointer<T>>::operator[](U rhs) const
{
    return *(*this + rhs);
}

template<typename T>
template<typename U> requires std::is_integral_v<U>
Pointer<T> ref<Pointer<T>>::operator+(const ref<Arithmetic<U>> &rhs) const
{
    static_assert(!std::is_same_v<T, CujVoid>);
    return static_cast<const Pointer<T>&>(*this) + rhs;
}

template<typename T>
template<typename U> requires std::is_integral_v<U>
Pointer<T> ref<Pointer<T>>::operator-(const ref<Arithmetic<U>> &rhs) const
{
    static_assert(!std::is_same_v<T, CujVoid>);
    return static_cast<const Pointer<T>&>(*this) - rhs;
}

template<typename T>
template<typename U> requires std::is_integral_v<U>
add_reference_t<T> ref<Pointer<T>>::operator[](const ref<Arithmetic<U>> &rhs) const
{
    return *(*this + rhs);
}

template<typename T>
Pointer<Pointer<T>> ref<Pointer<T>>::address() const
{
    Pointer<Pointer<T>> ret;
    ret = addr_;
    return ret;
}

template<typename T>
add_reference_t<T> ref<Pointer<T>>::deref() const
{
    static_assert(!std::is_same_v<T, CujVoid>);
    return add_reference_t<T>::_from_ptr(*this);
}

template<typename T>
add_reference_t<T> ref<Pointer<T>>::operator*() const
{
    return deref();
}

template<typename T>
add_reference_t<T> *ref<Pointer<T>>::operator->() const
{
    auto temp_var_ctx = PointerTempVarContext::get_context();
    auto ret = newRC<add_reference_t<T>>(this->deref());
    temp_var_ctx->add_var(ret);
    return ret.get();
}

template<typename T>
core::Load ref<Pointer<T>>::_load() const
{
    auto type = FunctionContext::get_func_context()->get_type_context()
        ->get_type<Pointer<T>>();
    return core::Load{
        .val_type = type,
        .src_addr = newRC<core::Expr>(addr_._load())
    };
}

template<typename T>
ref<Pointer<T>> ref<Pointer<T>>::_from_ptr(const Pointer<Pointer<T>> &ptr)
{
    ref ret;
    ret.addr_ = ptr;
    return ret;
}

template<typename U, typename T> requires std::is_integral_v<U>
Pointer<T> operator+(const Arithmetic<U> &lhs, const ref<Pointer<T>> &rhs)
{
    return rhs + lhs;
}


template<typename U, typename T> requires std::is_integral_v<U>
Pointer<T> operator+(const ref<Arithmetic<U>> &lhs, const ref<Pointer<T>> &rhs)
{
    return rhs + lhs;
}

CUJ_NAMESPACE_END(cuj::dsl)
