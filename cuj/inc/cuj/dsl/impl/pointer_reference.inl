#pragma once

#include <cuj/core/stat.h>
#include <cuj/dsl/function.h>
#include <cuj/dsl/pointer_reference.h>
#include <cuj/dsl/pointer_temp_var.h>

CUJ_NAMESPACE_BEGIN(cuj::dsl)

template<typename T>
ref<ptr<T>>::ref(const ptr<T> &ptr)
{
    addr_ = ptr.address();
}

template<typename T>
ref<ptr<T>>::ref(const ref &ref)
{
    addr_ = ref.address();
}

template<typename T>
ref<ptr<T>>::ref(ref &&other) noexcept
    : addr_(std::move(other.addr_))
{
    
}

template<typename T>
ref<ptr<T>> &ref<ptr<T>>::operator=(const ref &other)
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
ref<ptr<T>> &ref<ptr<T>>::operator=(const ptr<T> &other)
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
ptr<T> ref<ptr<T>>::operator+(const num<U> &rhs) const
{
    static_assert(!std::is_same_v<T, CujVoid>);
    return static_cast<const ptr<T>&>(*this) + rhs;
}

template<typename T>
template<typename U> requires std::is_integral_v<U>
ptr<T> ref<ptr<T>>::operator-(const num<U> &rhs) const
{
    static_assert(!std::is_same_v<T, CujVoid>);
    return static_cast<const ptr<T>&>(*this) - rhs;
}

template<typename T>
template<typename U> requires std::is_integral_v<U>
add_reference_t<T> ref<ptr<T>>::operator[](const num<U> &rhs) const
{
    return *(*this + rhs);
}

template<typename T>
template<typename U> requires std::is_integral_v<U>
ptr<T> ref<ptr<T>>::operator+(U rhs) const
{
    return *this + num(rhs);
}

template<typename T>
template<typename U> requires std::is_integral_v<U>
ptr<T> ref<ptr<T>>::operator-(U rhs) const
{
    return *this - num(rhs);
}

template<typename T>
template<typename U> requires std::is_integral_v<U>
add_reference_t<T> ref<ptr<T>>::operator[](U rhs) const
{
    return *(*this + rhs);
}

template<typename T>
template<typename U> requires std::is_integral_v<U>
ptr<T> ref<ptr<T>>::operator+(const ref<num<U>> &rhs) const
{
    static_assert(!std::is_same_v<T, CujVoid>);
    return static_cast<const ptr<T>&>(*this) + rhs;
}

template<typename T>
template<typename U> requires std::is_integral_v<U>
ptr<T> ref<ptr<T>>::operator-(const ref<num<U>> &rhs) const
{
    static_assert(!std::is_same_v<T, CujVoid>);
    return static_cast<const ptr<T>&>(*this) - rhs;
}

template<typename T>
template<typename U> requires std::is_integral_v<U>
add_reference_t<T> ref<ptr<T>>::operator[](const ref<num<U>> &rhs) const
{
    return *(*this + rhs);
}

template<typename T>
ptr<ptr<T>> ref<ptr<T>>::address() const
{
    ptr<ptr<T>> ret;
    ret = addr_;
    return ret;
}

template<typename T>
add_reference_t<T> ref<ptr<T>>::deref() const
{
    static_assert(!std::is_same_v<T, CujVoid>);
    return add_reference_t<T>::_from_ptr(*this);
}

template<typename T>
add_reference_t<T> ref<ptr<T>>::operator*() const
{
    return deref();
}

template<typename T>
add_reference_t<T> *ref<ptr<T>>::operator->() const
{
    auto temp_var_ctx = PointerTempVarContext::get_context();
    auto ret = newRC<add_reference_t<T>>(this->deref());
    temp_var_ctx->add_var(ret);
    return ret.get();
}

template<typename T>
core::Load ref<ptr<T>>::_load() const
{
    auto type = FunctionContext::get_func_context()->get_type_context()
        ->get_type<ptr<T>>();
    return core::Load{
        .val_type = type,
        .src_addr = newRC<core::Expr>(addr_._load())
    };
}

template<typename T>
ref<ptr<T>> ref<ptr<T>>::_from_ptr(const ptr<ptr<T>> &ptr)
{
    ref ret;
    ret.addr_ = ptr;
    return ret;
}

template<typename U, typename T> requires std::is_integral_v<U>
ptr<T> operator+(const num<U> &lhs, const ref<ptr<T>> &rhs)
{
    return rhs + lhs;
}


template<typename U, typename T> requires std::is_integral_v<U>
ptr<T> operator+(const ref<num<U>> &lhs, const ref<ptr<T>> &rhs)
{
    return rhs + lhs;
}

template<typename T>
num<bool> operator==(const ref<ptr<T>> &lhs, const ref<ptr<T>> &rhs)
{
    return bitcast<num<size_t>>(lhs) == bitcast<num<size_t>>(rhs);
}

template<typename T>
num<bool> operator!=(const ref<ptr<T>> &lhs, const ref<ptr<T>> &rhs)
{
    return bitcast<num<size_t>>(lhs) != bitcast<num<size_t>>(rhs);
}

CUJ_NAMESPACE_END(cuj::dsl)
