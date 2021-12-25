#pragma once

#include <cuj/dsl/arithmetic.h>
#include <cuj/dsl/function.h>
#include <cuj/dsl/if.h>
#include <cuj/dsl/pointer.h>

CUJ_NAMESPACE_BEGIN(cuj::dsl)

template<typename T> requires std::is_arithmetic_v<T>
Arithmetic<T>::Arithmetic()
{
    auto func_ctx = FunctionContext::get_func_context();
    auto type_ctx = func_ctx->get_type_context();

    auto type = type_ctx->get_type<Arithmetic>();
    alloc_index_ = func_ctx->alloc_local_var(type);
}

template<typename T> requires std::is_arithmetic_v<T>
Arithmetic<T>::Arithmetic(T immediate_value)
    : Arithmetic()
{
    core::Store store = {
        .dst_addr = _addr(),
        .val = core::Immediate{
            .value = immediate_value
        }
    };
    auto func_ctx = FunctionContext::get_func_context();
    func_ctx->append_statement(newRC<core::Stat>(std::move(store)));
}

template<typename T> requires std::is_arithmetic_v<T>
Arithmetic<T>::Arithmetic(const Arithmetic &other)
    : Arithmetic()
{
    *this = other;
}

template<typename T> requires std::is_arithmetic_v<T>
template<typename U> requires !std::is_same_v<T, U>
Arithmetic<T>::Arithmetic(const Arithmetic<U> &other)
    : Arithmetic()
{
    *this = other.template as<Arithmetic>();
}

template<typename T> requires std::is_arithmetic_v<T>
template<typename U> requires !std::is_same_v<T, U>
Arithmetic<T>::Arithmetic(const ref<Arithmetic<U>> &other)
    : Arithmetic()
{
    *this = other.template as<Arithmetic>();
}

template<typename T> requires std::is_arithmetic_v<T>
Arithmetic<T>::Arithmetic(const ref<Arithmetic<T>> &ref)
    : Arithmetic()
{
    *this = Arithmetic::_from_expr(ref._load());
}

template<typename T> requires std::is_arithmetic_v<T>
Arithmetic<T>::Arithmetic(Arithmetic &&other) noexcept
    : alloc_index_(other.alloc_index_)
{
    
}

template<typename T> requires std::is_arithmetic_v<T>
Arithmetic<T> &Arithmetic<T>::operator=(const Arithmetic &other)
{
    if(other.alloc_index_ == alloc_index_)
        return *this;
    auto type = FunctionContext::get_func_context()
        ->get_type_context()->get_type<Arithmetic>();
    auto load = core::Load{
        .val_type = type,
        .src_addr = newBox<core::Expr>(other._addr())
    };
    auto store = core::Store{
        .dst_addr = _addr(),
        .val      = load
    };
    auto func_ctx = FunctionContext::get_func_context();
    func_ctx->append_statement(newRC<core::Stat>(std::move(store)));
    return *this;
}

template<typename T> requires std::is_arithmetic_v<T>
template<typename U> requires is_cuj_arithmetic_v<U>
U Arithmetic<T>::as() const
{
    using URaw = typename U::RawType;
    auto src_type = FunctionContext::get_func_context()
        ->get_type_context()->get_type<Arithmetic>();
    auto dst_type = FunctionContext::get_func_context()
        ->get_type_context()->get_type<U>();
    core::ArithmeticCast cast = {
        .dst_type = dst_type,
        .src_type = src_type,
        .src_val  = newRC<core::Expr>(_load())
    };
    return U::_from_expr(cast);
}

template<typename T> requires std::is_arithmetic_v<T>
Arithmetic<T> Arithmetic<T>::operator-() const
{
    static_assert(!std::is_same_v<T, bool>);
    auto type_ctx = FunctionContext::get_func_context()->get_type_context();
    auto type = type_ctx->get_type<Arithmetic>();
    core::Unary unary = {
        .op       = core::Unary::Op::Neg,
        .val      = newRC<core::Expr>(_load()),
        .val_type = type
    };
    return _from_expr(unary);
}

template<typename T> requires std::is_arithmetic_v<T>
Arithmetic<T> Arithmetic<T>::operator+(const Arithmetic &rhs) const
{
    static_assert(!std::is_same_v<T, bool>);
    auto type_ctx = FunctionContext::get_func_context()->get_type_context();
    auto type = type_ctx->get_type<Arithmetic>();
    core::Binary binary = {
        .op       = core::Binary::Op::Add,
        .lhs      = newRC<core::Expr>(_load()),
        .rhs      = newRC<core::Expr>(rhs._load()),
        .lhs_type = type,
        .rhs_type = type
    };
    return _from_expr(binary);
}

template<typename T> requires std::is_arithmetic_v<T>
Arithmetic<T> Arithmetic<T>::operator-(const Arithmetic &rhs) const
{
    static_assert(!std::is_same_v<T, bool>);
    auto type_ctx = FunctionContext::get_func_context()->get_type_context();
    auto type = type_ctx->get_type<Arithmetic>();
    core::Binary binary = {
        .op       = core::Binary::Op::Sub,
        .lhs      = newRC<core::Expr>(_load()),
        .rhs      = newRC<core::Expr>(rhs._load()),
        .lhs_type = type,
        .rhs_type = type
    };
    return _from_expr(binary);
}

template<typename T> requires std::is_arithmetic_v<T>
Arithmetic<T> Arithmetic<T>::operator*(const Arithmetic &rhs) const
{
    static_assert(!std::is_same_v<T, bool>);
    auto type_ctx = FunctionContext::get_func_context()->get_type_context();
    auto type = type_ctx->get_type<Arithmetic>();
    core::Binary binary = {
        .op       = core::Binary::Op::Mul,
        .lhs      = newRC<core::Expr>(_load()),
        .rhs      = newRC<core::Expr>(rhs._load()),
        .lhs_type = type,
        .rhs_type = type
    };
    return _from_expr(binary);
}

template<typename T> requires std::is_arithmetic_v<T>
Arithmetic<T> Arithmetic<T>::operator/(const Arithmetic &rhs) const
{
    static_assert(!std::is_same_v<T, bool>);
    auto type_ctx = FunctionContext::get_func_context()->get_type_context();
    auto type = type_ctx->get_type<Arithmetic>();
    core::Binary binary = {
        .op       = core::Binary::Op::Div,
        .lhs      = newRC<core::Expr>(_load()),
        .rhs      = newRC<core::Expr>(rhs._load()),
        .lhs_type = type,
        .rhs_type = type
    };
    return _from_expr(binary);
}

template<typename T> requires std::is_arithmetic_v<T>
Arithmetic<T> Arithmetic<T>::operator%(const Arithmetic &rhs) const
{
    static_assert(!std::is_same_v<T, bool>);
    static_assert(std::is_integral_v<T>);
    auto type_ctx = FunctionContext::get_func_context()->get_type_context();
    auto type = type_ctx->get_type<Arithmetic>();
    core::Binary binary = {
        .op       = core::Binary::Op::Mod,
        .lhs      = newRC<core::Expr>(_load()),
        .rhs      = newRC<core::Expr>(rhs._load()),
        .lhs_type = type,
        .rhs_type = type
    };
    return _from_expr(binary);
}

template<typename T> requires std::is_arithmetic_v<T>
Arithmetic<bool> Arithmetic<T>::operator==(const Arithmetic &rhs) const
{
    auto type = FunctionContext::get_func_context()
        ->get_type_context()->get_type<Arithmetic>();
    return Arithmetic<bool>::_from_expr(core::Binary{
        .op       = core::Binary::Op::Equal,
        .lhs      = newRC<core::Expr>(_load()),
        .rhs      = newRC<core::Expr>(rhs._load()),
        .lhs_type = type,
        .rhs_type = type
    });
}

template<typename T> requires std::is_arithmetic_v<T>
Arithmetic<bool> Arithmetic<T>::operator!=(const Arithmetic &rhs) const
{
    auto type = FunctionContext::get_func_context()
        ->get_type_context()->get_type<Arithmetic>();
    return Arithmetic<bool>::_from_expr(core::Binary{
        .op       = core::Binary::Op::NotEqual,
        .lhs      = newRC<core::Expr>(_load()),
        .rhs      = newRC<core::Expr>(rhs._load()),
        .lhs_type = type,
        .rhs_type = type
    });
}

template<typename T> requires std::is_arithmetic_v<T>
Arithmetic<bool> Arithmetic<T>::operator<(const Arithmetic &rhs) const
{
    auto type = FunctionContext::get_func_context()
        ->get_type_context()->get_type<Arithmetic>();
    return Arithmetic<bool>::_from_expr(core::Binary{
        .op       = core::Binary::Op::Less,
        .lhs      = newRC<core::Expr>(_load()),
        .rhs      = newRC<core::Expr>(rhs._load()),
        .lhs_type = type,
        .rhs_type = type
    });
}

template<typename T> requires std::is_arithmetic_v<T>
Arithmetic<bool> Arithmetic<T>::operator<=(const Arithmetic &rhs) const
{
    auto type = FunctionContext::get_func_context()
        ->get_type_context()->get_type<Arithmetic>();
    return Arithmetic<bool>::_from_expr(core::Binary{
        .op       = core::Binary::Op::LessEqual,
        .lhs      = newRC<core::Expr>(_load()),
        .rhs      = newRC<core::Expr>(rhs._load()),
        .lhs_type = type,
        .rhs_type = type
    });
}

template<typename T> requires std::is_arithmetic_v<T>
Arithmetic<bool> Arithmetic<T>::operator>(const Arithmetic &rhs) const
{
    auto type = FunctionContext::get_func_context()
        ->get_type_context()->get_type<Arithmetic>();
    return Arithmetic<bool>::_from_expr(core::Binary{
        .op       = core::Binary::Op::Greater,
        .lhs      = newRC<core::Expr>(_load()),
        .rhs      = newRC<core::Expr>(rhs._load()),
        .lhs_type = type,
        .rhs_type = type
    });
}

template<typename T> requires std::is_arithmetic_v<T>
Arithmetic<bool> Arithmetic<T>::operator>=(const Arithmetic &rhs) const
{
    auto type = FunctionContext::get_func_context()
        ->get_type_context()->get_type<Arithmetic>();
    return Arithmetic<bool>::_from_expr(core::Binary{
        .op       = core::Binary::Op::GreaterEqual,
        .lhs      = newRC<core::Expr>(_load()),
        .rhs      = newRC<core::Expr>(rhs._load()),
        .lhs_type = type,
        .rhs_type = type
    });
}

template<typename T> requires std::is_arithmetic_v<T>
Arithmetic<bool> Arithmetic<T>::operator==(T rhs) const
{
    return *this == Arithmetic(rhs);
}

template<typename T> requires std::is_arithmetic_v<T>
Arithmetic<bool> Arithmetic<T>::operator!=(T rhs) const
{
    return *this != Arithmetic(rhs);
}

template<typename T> requires std::is_arithmetic_v<T>
Arithmetic<bool> Arithmetic<T>::operator<(T rhs) const
{
    return *this < Arithmetic(rhs);
}

template<typename T> requires std::is_arithmetic_v<T>
Arithmetic<bool> Arithmetic<T>::operator<=(T rhs) const
{
    return *this <= Arithmetic(rhs);
}

template<typename T> requires std::is_arithmetic_v<T>
Arithmetic<bool> Arithmetic<T>::operator>(T rhs) const
{
    return *this > Arithmetic(rhs);
}

template<typename T> requires std::is_arithmetic_v<T>
Arithmetic<bool> Arithmetic<T>::operator>=(T rhs) const
{
    return *this >= Arithmetic(rhs);
}

template<typename T> requires std::is_arithmetic_v<T>
Arithmetic<T> Arithmetic<T>::operator>>(const Arithmetic &rhs) const
{
    static_assert(std::is_integral_v<T> && !std::is_signed_v<T>);
    auto type = FunctionContext::get_func_context()
        ->get_type_context()->get_type<Arithmetic>();
    return Arithmetic::_from_expr(core::Binary{
        .op       = core::Binary::Op::RightShift,
        .lhs      = newRC<core::Expr>(this->_load()),
        .rhs      = newRC<core::Expr>(rhs._load()),
        .lhs_type = type,
        .rhs_type = type
    });
}

template<typename T> requires std::is_arithmetic_v<T>
Arithmetic<T> Arithmetic<T>::operator<<(const Arithmetic &rhs) const
{
    static_assert(std::is_integral_v<T>);
    auto type = FunctionContext::get_func_context()
        ->get_type_context()->get_type<Arithmetic>();
    return Arithmetic::_from_expr(core::Binary{
        .op       = core::Binary::Op::LeftShift,
        .lhs      = newRC<core::Expr>(this->_load()),
        .rhs      = newRC<core::Expr>(rhs._load()),
        .lhs_type = type,
        .rhs_type = type
    });
}

template<typename T> requires std::is_arithmetic_v<T>
Arithmetic<T> Arithmetic<T>::operator&(const Arithmetic &rhs) const
{
    static_assert(std::is_integral_v<T>);
    auto type = FunctionContext::get_func_context()
        ->get_type_context()->get_type<Arithmetic>();
    return Arithmetic::_from_expr(core::Binary{
        .op       = core::Binary::Op::BitwiseAnd,
        .lhs      = newRC<core::Expr>(this->_load()),
        .rhs      = newRC<core::Expr>(rhs._load()),
        .lhs_type = type,
        .rhs_type = type
    });
}

template<typename T> requires std::is_arithmetic_v<T>
Arithmetic<T> Arithmetic<T>::operator|(const Arithmetic &rhs) const
{
    static_assert(std::is_integral_v<T>);
    auto type = FunctionContext::get_func_context()
        ->get_type_context()->get_type<Arithmetic>();
    return Arithmetic::_from_expr(core::Binary{
        .op       = core::Binary::Op::BitwiseOr,
        .lhs      = newRC<core::Expr>(this->_load()),
        .rhs      = newRC<core::Expr>(rhs._load()),
        .lhs_type = type,
        .rhs_type = type
    });
}

template<typename T> requires std::is_arithmetic_v<T>
Arithmetic<T> Arithmetic<T>::operator^(const Arithmetic &rhs) const
{
    static_assert(std::is_integral_v<T>);
    auto type = FunctionContext::get_func_context()
        ->get_type_context()->get_type<Arithmetic>();
    return Arithmetic::_from_expr(core::Binary{
        .op       = core::Binary::Op::BitwiseXOr,
        .lhs      = newRC<core::Expr>(this->_load()),
        .rhs      = newRC<core::Expr>(rhs._load()),
        .lhs_type = type,
        .rhs_type = type
    });
}

template<typename T> requires std::is_arithmetic_v<T>
Arithmetic<T> Arithmetic<T>::operator~() const
{
    auto type = FunctionContext::get_func_context()
        ->get_type_context()->get_type<Arithmetic>();
    return Arithmetic::_from_expr(core::Unary{
        .op       = core::Unary::Op::BitwiseNot,
        .val      = newRC<core::Expr>(_load()),
        .val_type = type
    });
}

template<typename T> requires std::is_arithmetic_v<T>
Pointer<Arithmetic<T>> Arithmetic<T>::address() const
{
    Pointer<Arithmetic> ret;
    core::Store store = {
        .dst_addr = ret._addr(),
        .val      = _addr()
    };
    auto func_ctx = FunctionContext::get_func_context();
    func_ctx->append_statement(std::move(store));
    return ret;
}

template<typename T> requires std::is_arithmetic_v<T>
Arithmetic<T> Arithmetic<T>::_from_expr(core::Expr expr)
{
    Arithmetic ret;
    core::Store store = {
        .dst_addr = ret._addr(),
        .val      = std::move(expr)
    };
    auto func_ctx = FunctionContext::get_func_context();
    func_ctx->append_statement(newRC<core::Stat>(std::move(store)));
    return ret;
}

template<typename T> requires std::is_arithmetic_v<T>
core::Load Arithmetic<T>::_load() const
{
    auto type = FunctionContext::get_func_context()
        ->get_type_context()->get_type<Arithmetic>();
    return core::Load{
        .val_type = type,
        .src_addr = newRC<core::Expr>(_addr())
    };
}

template<typename T> requires std::is_arithmetic_v<T>
core::LocalAllocAddr Arithmetic<T>::_addr() const
{
    auto type = FunctionContext::get_func_context()
        ->get_type_context()->get_type<Arithmetic>();
    return core::LocalAllocAddr{
        .alloc_type  = type,
        .alloc_index = alloc_index_
    };
}

template<typename T>
Arithmetic<T> operator+(T lhs, const Arithmetic<T> &rhs)
{
    return rhs + lhs;
}

template<typename T>
Arithmetic<T> operator-(T lhs, const Arithmetic<T> &rhs)
{
    return Arithmetic(lhs) - rhs;
}

template<typename T>
Arithmetic<T> operator*(T lhs, const Arithmetic<T> &rhs)
{
    return rhs * lhs;
}

template<typename T>
Arithmetic<T> operator/(T lhs, const Arithmetic<T> &rhs)
{
    return Arithmetic(lhs) / rhs;
}

template<typename T>
Arithmetic<T> operator%(T lhs, const Arithmetic<T> &rhs)
{
    return Arithmetic(lhs) % rhs;
}

template<typename T> requires std::is_arithmetic_v<T>
Arithmetic<bool> operator==(T lhs, const Arithmetic<T> &rhs)
{
    return Arithmetic(lhs) == rhs;
}

template<typename T> requires std::is_arithmetic_v<T>
Arithmetic<bool> operator!=(T lhs, const Arithmetic<T> &rhs)
{
    return Arithmetic(lhs) != rhs;
}

template<typename T> requires std::is_arithmetic_v<T>
Arithmetic<bool> operator<(T lhs, const Arithmetic<T> &rhs)
{
    return Arithmetic(lhs) < rhs;
}

template<typename T> requires std::is_arithmetic_v<T>
Arithmetic<bool> operator<=(T lhs, const Arithmetic<T> &rhs)
{
    return Arithmetic(lhs) <= rhs;
}

template<typename T> requires std::is_arithmetic_v<T>
Arithmetic<bool> operator>(T lhs, const Arithmetic<T> &rhs)
{
    return Arithmetic(lhs) > rhs;
}

template<typename T> requires std::is_arithmetic_v<T>
Arithmetic<bool> operator>=(T lhs, const Arithmetic<T> &rhs)
{
    return Arithmetic(lhs) >= rhs;
}

inline Arithmetic<bool> operator!(const Arithmetic<bool> &val)
{
    auto type = FunctionContext::get_func_context()
        ->get_type_context()->get_type<Arithmetic<bool>>();
    return Arithmetic<bool>::_from_expr(core::Unary{
        .op       = core::Unary::Op::Not,
        .val      = newRC<core::Expr>(val._load()),
        .val_type = type
    });
}

template<typename T> requires std::is_integral_v<T> && !std::is_signed_v<T>
Arithmetic<T> operator>>(T lhs, const Arithmetic<T> &rhs)
{
    return Arithmetic(lhs) >> rhs;
}

template<typename T> requires std::is_integral_v<T>
Arithmetic<T> operator<<(T lhs, const Arithmetic<T> &rhs)
{
    return Arithmetic(lhs) << rhs;
}

template<typename T> requires std::is_integral_v<T>
Arithmetic<T> operator&(T lhs, const Arithmetic<T> &rhs)
{
    return Arithmetic(lhs) & rhs;
}

template<typename T> requires std::is_integral_v<T>
Arithmetic<T> operator|(T lhs, const Arithmetic<T> &rhs)
{
    return Arithmetic(lhs) | rhs;
}

template<typename T> requires std::is_integral_v<T>
Arithmetic<T> operator^(T lhs, const Arithmetic<T> &rhs)
{
    return Arithmetic(lhs) ^ rhs;
}

CUJ_NAMESPACE_END(cuj::dsl)
