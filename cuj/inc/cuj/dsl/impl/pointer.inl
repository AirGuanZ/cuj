#pragma once

#include <cuj/dsl/arithmetic.h>
#include <cuj/dsl/bitcast.h>
#include <cuj/dsl/function.h>
#include <cuj/dsl/pointer.h>
#include <cuj/dsl/pointer_temp_var.h>

CUJ_NAMESPACE_BEGIN(cuj::dsl)

template<typename T>
const core::Type *ptr<T>::type()
{
    return FunctionContext::get_func_context()
        ->get_type_context()->get_type<ptr>();
}

template<typename T>
ptr<T>::ptr()
{
    static_assert(is_cuj_var_v<T> || std::is_same_v<T, CujVoid>);
    auto func_ctx = FunctionContext::get_func_context();
    alloc_index_ = func_ctx->alloc_local_var(type());
}

template<typename T>
ptr<T>::ptr(std::nullptr_t)
    : ptr()
{
    core::Store store = {
        .dst_addr = _addr(),
        .val      = core::NullPtr{ type() }
    };
    auto func_ctx = FunctionContext::get_func_context();
    func_ctx->append_statement(std::move(store));
}

template<typename T>
ptr<T>::ptr(const ref<ptr<T>> &ref)
    : ptr()
{
    *this = ptr::_from_expr(ref._load());
}

template<typename T>
ptr<T>::ptr(const ptr &other)
    : ptr()
{
    *this = other;
}

template<typename T>
ptr<T>::ptr(ptr &&other) noexcept
    : alloc_index_(other.alloc_index_)
{
    static_assert(is_cuj_var_v<T> || std::is_same_v<T, CujVoid>);
}

template<typename T>
ptr<T> &ptr<T>::operator=(const ptr &other)
{
    if(other.alloc_index_ == alloc_index_)
        return *this;
    core::Store store = {
        .dst_addr = _addr(),
        .val      = other._load()
    };
    auto func_ctx = FunctionContext::get_func_context();
    func_ctx->append_statement(std::move(store));
    return *this;
}

template<typename T>
template<typename U> requires std::is_integral_v<U>
ptr<T> ptr<T>::operator+(const num<U> &rhs) const
{
    static_assert(!std::is_same_v<T, CujVoid>);

    auto func_ctx = FunctionContext::get_func_context();
    auto type_ctx = func_ctx->get_type_context();

    core::PointerOffset ptr_offset = {
        .ptr_type    = type(),
        .offset_type = type_ctx->get_type<num<U>>(),
        .ptr_val     = newRC<core::Expr>(_load()),
        .offset_val  = newRC<core::Expr>(rhs._load()),
        .negative    = false
    };

    return _from_expr(std::move(ptr_offset));
}

template<typename T>
template<typename U> requires std::is_integral_v<U>
ptr<T> ptr<T>::operator-(const num<U> &rhs) const
{
    static_assert(!std::is_same_v<T, CujVoid>);

    auto func_ctx = FunctionContext::get_func_context();
    auto type_ctx = func_ctx->get_type_context();

    core::PointerOffset ptr_offset = {
        .ptr_type    = type(),
        .offset_type = type_ctx->get_type<num<U>>(),
        .ptr_val     = newRC<core::Expr>(_load()),
        .offset_val  = newRC<core::Expr>(rhs._load()),
        .negative    = true
    };

    return _from_expr(std::move(ptr_offset));
}

template<typename T>
template<typename U> requires std::is_integral_v<U>
add_reference_t<T> ptr<T>::operator[](const num<U> &rhs) const
{
    return *(*this + rhs);
}

template<typename T>
template<typename U> requires std::is_integral_v<U>
ptr<T> ptr<T>::operator+(U rhs) const
{
    return *this + num(rhs);
}

template<typename T>
template<typename U> requires std::is_integral_v<U>
ptr<T> ptr<T>::operator-(U rhs) const
{
    return *this - num(rhs);
}

template<typename T>
template<typename U> requires std::is_integral_v<U>
add_reference_t<T> ptr<T>::operator[](U rhs) const
{
    return *(*this + rhs);
}

template<typename T>
template<typename U> requires std::is_integral_v<U>
ptr<T> ptr<T>::operator+(const ref<num<U>> &rhs) const
{
    static_assert(!std::is_same_v<T, CujVoid>);

    auto func_ctx = FunctionContext::get_func_context();
    auto type_ctx = func_ctx->get_type_context();

    core::PointerOffset ptr_offset = {
        .ptr_type    = type(),
        .offset_type = type_ctx->get_type<num<U>>(),
        .ptr_val     = newRC<core::Expr>(_load()),
        .offset_val  = newRC<core::Expr>(rhs._load()),
        .negative    = false
    };

    return _from_expr(std::move(ptr_offset));
}

template<typename T>
template<typename U> requires std::is_integral_v<U>
ptr<T> ptr<T>::operator-(const ref<num<U>> &rhs) const
{
    static_assert(!std::is_same_v<T, CujVoid>);

    auto func_ctx = FunctionContext::get_func_context();
    auto type_ctx = func_ctx->get_type_context();

    core::PointerOffset ptr_offset = {
        .ptr_type    = type(),
        .offset_type = type_ctx->get_type<num<U>>(),
        .ptr_val     = newRC<core::Expr>(_load()),
        .offset_val  = newRC<core::Expr>(rhs._load()),
        .negative    = true
    };

    return _from_expr(std::move(ptr_offset));
}

template<typename T>
template<typename U> requires std::is_integral_v<U>
add_reference_t<T> ptr<T>::operator[](const ref<num<U>> &rhs) const
{
    return *(*this + rhs);
}

template<typename T>
ptr<ptr<T>> ptr<T>::address() const
{
    ptr<ptr> ret;
    core::Store store = {
        .dst_addr = ret._addr(),
        .val      = _addr()
    };
    auto func_ctx = FunctionContext::get_func_context();
    func_ctx->append_statement(std::move(store));
    return ret;
}

template<typename T>
add_reference_t<T> ptr<T>::deref() const
{
    static_assert(!std::is_same_v<T, CujVoid>);
    return add_reference_t<T>::_from_ptr(*this);
}

template<typename T>
add_reference_t<T> ptr<T>::operator*() const
{
    return deref();
}

template<typename T>
add_reference_t<T> *ptr<T>::operator->() const
{
    auto temp_var_ctx = PointerTempVarContext::get_context();
    auto ret = newRC<add_reference_t<T>>(this->deref());
    temp_var_ctx->add_var(ret);
    return ret.get();
}

template<typename T>
ptr<T> ptr<T>::_from_expr(core::Expr expr)
{
    ptr ret;
    core::Store store = {
        .dst_addr = ret._addr(),
        .val      = expr
    };
    auto func_ctx = FunctionContext::get_func_context();
    func_ctx->append_statement(std::move(store));
    return ret;
}

template<typename T>
core::LocalAllocAddr ptr<T>::_addr() const
{
    return core::LocalAllocAddr{
        .alloc_type  = type(),
        .alloc_index = alloc_index_
    };
}

template<typename T>
core::Load ptr<T>::_load() const
{
    return core::Load{
        .val_type = type(),
        .src_addr = newRC<core::Expr>(_addr())
    };
}

template<typename U, typename T> requires std::is_integral_v<U>
ptr<T> operator+(const num<U> &lhs, const ptr<T> &rhs)
{
    return rhs + lhs;
}

template<typename U, typename T> requires std::is_integral_v<U>
ptr<T> operator+(U lhs, const ptr<T> &rhs)
{
    return rhs + lhs;
}

template<typename U, typename T> requires std::is_integral_v<U>
ptr<T> operator+(const ref<num<U>> &lhs, const ptr<T> &rhs)
{
    return rhs + lhs;
}

template<typename T>
num<bool> operator==(const ptr<T> &lhs, const ptr<T> &rhs)
{
    return bitcast<num<size_t>>(lhs) == bitcast<num<size_t>>(rhs);
}

template<typename T>
num<bool> operator!=(const ptr<T> &lhs, const ptr<T> &rhs)
{
    return bitcast<num<size_t>>(lhs) != bitcast<num<size_t>>(rhs);
}

template<typename T>
ptr<cxx<T>> import_pointer(T *pointer)
{
    num v = reinterpret_cast<size_t>(pointer);
    return bitcast<ptr<cxx<T>>>(v);
}

template<typename T>
ptr<cxx<T>> import_pointer(const T *pointer)
{
    num v = reinterpret_cast<size_t>(pointer);
    return bitcast<ptr<cxx<T>>>(v);
}

CUJ_NAMESPACE_END(cuj::dsl)
