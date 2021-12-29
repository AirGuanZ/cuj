#pragma once

#include <cuj/dsl/arithmetic.h>
#include <cuj/dsl/function.h>
#include <cuj/dsl/pointer.h>
#include <cuj/dsl/pointer_temp_var.h>

CUJ_NAMESPACE_BEGIN(cuj::dsl)

template<typename T>
const core::Type *Pointer<T>::type()
{
    return FunctionContext::get_func_context()
        ->get_type_context()->get_type<Pointer>();
}

template<typename T>
Pointer<T>::Pointer()
{
    static_assert(is_cuj_var_v<T> || std::is_same_v<T, CujVoid>);
    auto func_ctx = FunctionContext::get_func_context();
    alloc_index_ = func_ctx->alloc_local_var(type());
}

template<typename T>
Pointer<T>::Pointer(std::nullptr_t)
    : Pointer()
{
    core::Store store = {
        .dst_addr = _addr(),
        .val      = core::NullPtr{ type() }
    };

    auto func_ctx = FunctionContext::get_func_context();
    func_ctx->append_statement(std::move(store));
}

template<typename T>
Pointer<T>::Pointer(const ref<Pointer<T>> &ref)
    : Pointer()
{
    *this = Pointer::_from_expr(ref._load());
}

template<typename T>
Pointer<T>::Pointer(const Pointer &other)
    : Pointer()
{
    *this = other;
}

template<typename T>
Pointer<T>::Pointer(Pointer &&other) noexcept
    : alloc_index_(other.alloc_index_)
{
    static_assert(is_cuj_var_v<T> || std::is_same_v<T, CujVoid>);
}

template<typename T>
Pointer<T> &Pointer<T>::operator=(const Pointer &other)
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
Pointer<T> Pointer<T>::operator+(const Arithmetic<U> &rhs) const
{
    static_assert(!std::is_same_v<T, CujVoid>);

    auto func_ctx = FunctionContext::get_func_context();
    auto type_ctx = func_ctx->get_type_context();

    core::PointerOffset ptr_offset = {
        .ptr_type    = type(),
        .offset_type = type_ctx->get_type<Arithmetic<U>>(),
        .ptr_val     = newRC<core::Expr>(_load()),
        .offset_val  = newRC<core::Expr>(rhs._load()),
        .negative    = false
    };

    return _from_expr(std::move(ptr_offset));
}

template<typename T>
template<typename U> requires std::is_integral_v<U>
Pointer<T> Pointer<T>::operator-(const Arithmetic<U> &rhs) const
{
    static_assert(!std::is_same_v<T, CujVoid>);

    auto func_ctx = FunctionContext::get_func_context();
    auto type_ctx = func_ctx->get_type_context();

    core::PointerOffset ptr_offset = {
        .ptr_type    = type(),
        .offset_type = type_ctx->get_type<Arithmetic<U>>(),
        .ptr_val     = newRC<core::Expr>(_load()),
        .offset_val  = newRC<core::Expr>(rhs._load()),
        .negative    = true
    };

    return _from_expr(std::move(ptr_offset));
}

template<typename T>
template<typename U> requires std::is_integral_v<U>
add_reference_t<T> Pointer<T>::operator[](const Arithmetic<U> &rhs) const
{
    return *(*this + rhs);
}

template<typename T>
template<typename U> requires std::is_integral_v<U>
Pointer<T> Pointer<T>::operator+(U rhs) const
{
    return *this + Arithmetic(rhs);
}

template<typename T>
template<typename U> requires std::is_integral_v<U>
Pointer<T> Pointer<T>::operator-(U rhs) const
{
    return *this - Arithmetic(rhs);
}

template<typename T>
template<typename U> requires std::is_integral_v<U>
add_reference_t<T> Pointer<T>::operator[](U rhs) const
{
    return *(*this + rhs);
}

template<typename T>
template<typename U> requires std::is_integral_v<U>
Pointer<T> Pointer<T>::operator+(const ref<Arithmetic<U>> &rhs) const
{
    static_assert(!std::is_same_v<T, CujVoid>);

    auto func_ctx = FunctionContext::get_func_context();
    auto type_ctx = func_ctx->get_type_context();

    core::PointerOffset ptr_offset = {
        .ptr_type    = type(),
        .offset_type = type_ctx->get_type<Arithmetic<U>>(),
        .ptr_val     = newRC<core::Expr>(_load()),
        .offset_val  = newRC<core::Expr>(rhs._load()),
        .negative    = false
    };

    return _from_expr(std::move(ptr_offset));
}

template<typename T>
template<typename U> requires std::is_integral_v<U>
Pointer<T> Pointer<T>::operator-(const ref<Arithmetic<U>> &rhs) const
{
    static_assert(!std::is_same_v<T, CujVoid>);

    auto func_ctx = FunctionContext::get_func_context();
    auto type_ctx = func_ctx->get_type_context();

    core::PointerOffset ptr_offset = {
        .ptr_type    = type(),
        .offset_type = type_ctx->get_type<Arithmetic<U>>(),
        .ptr_val     = newRC<core::Expr>(_load()),
        .offset_val  = newRC<core::Expr>(rhs._load()),
        .negative    = true
    };

    return _from_expr(std::move(ptr_offset));
}

template<typename T>
template<typename U> requires std::is_integral_v<U>
add_reference_t<T> Pointer<T>::operator[](const ref<Arithmetic<U>> &rhs) const
{
    return *(*this + rhs);
}

template<typename U, typename T> requires std::is_integral_v<U>
Pointer<T> operator+(const Arithmetic<U> &lhs, const Pointer<T> &rhs)
{
    return rhs + lhs;
}

template<typename U, typename T> requires std::is_integral_v<U>
Pointer<T> operator+(U lhs, const Pointer<T> &rhs)
{
    return rhs + lhs;
}

template<typename U, typename T> requires std::is_integral_v<U>
Pointer<T> operator+(const ref<Arithmetic<U>> &lhs, const Pointer<T> &rhs)
{
    return rhs + lhs;
}

template<typename T>
Pointer<Pointer<T>> Pointer<T>::address() const
{
    Pointer<Pointer> ret;
    core::Store store = {
        .dst_addr = ret._addr(),
        .val      = _addr()
    };
    auto func_ctx = FunctionContext::get_func_context();
    func_ctx->append_statement(std::move(store));
    return ret;
}

template<typename T>
add_reference_t<T> Pointer<T>::deref() const
{
    static_assert(!std::is_same_v<T, CujVoid>);
    return add_reference_t<T>::_from_ptr(*this);
}

template<typename T>
add_reference_t<T> Pointer<T>::operator*() const
{
    return deref();
}

template<typename T>
add_reference_t<T> *Pointer<T>::operator->() const
{
    auto temp_var_ctx = PointerTempVarContext::get_context();
    auto ret = newRC<add_reference_t<T>>(this->deref());
    temp_var_ctx->add_var(ret);
    return ret.get();
}

template<typename T>
Pointer<T> Pointer<T>::_from_expr(core::Expr expr)
{
    Pointer ret;
    core::Store store = {
        .dst_addr = ret._addr(),
        .val      = expr
    };
    auto func_ctx = FunctionContext::get_func_context();
    func_ctx->append_statement(std::move(store));
    return ret;
}

template<typename T>
core::LocalAllocAddr Pointer<T>::_addr() const
{
    return core::LocalAllocAddr{
        .alloc_type  = type(),
        .alloc_index = alloc_index_
    };
}

template<typename T>
core::Load Pointer<T>::_load() const
{
    return core::Load{
        .val_type = type(),
        .src_addr = newRC<core::Expr>(_addr())
    };
}

CUJ_NAMESPACE_END(cuj::dsl)
