#pragma once

#include <cuj/core/stat.h>
#include <cuj/dsl/arithmetic.h>
#include <cuj/dsl/arithmetic_reference.h>
#include <cuj/dsl/function.h>

CUJ_NAMESPACE_BEGIN(cuj::dsl)

template<typename T> requires std::is_arithmetic_v<T>
ref<Arithmetic<T>>::ref(const Arithmetic<T> &var)
{
    addr_ = var.address();
}

template<typename T> requires std::is_arithmetic_v<T>
ref<Arithmetic<T>>::ref(const ref &ref)
{
    addr_ = ref.address();
}

template<typename T> requires std::is_arithmetic_v<T>
ref<Arithmetic<T>>::ref(ref &&other) noexcept
    : addr_(std::move(other.addr_))
{
    
}

template<typename T> requires std::is_arithmetic_v<T>
ref<Arithmetic<T>> &ref<Arithmetic<T>>::operator=(
    const ref &other)
{
    core::Store store = {
        .dst_addr = addr_._load(),
        .val      = other._load()
    };
    auto func_ctx = FunctionContext::get_func_context();
    func_ctx->append_statement(std::move(store));
    return *this;
}

template<typename T> requires std::is_arithmetic_v<T>
ref<Arithmetic<T>> &ref<Arithmetic<T>>::operator=(
    const Arithmetic<T> &other)
{
    core::Store store = {
        .dst_addr = addr_._load(),
        .val      = other._load()
    };
    auto func_ctx = FunctionContext::get_func_context();
    func_ctx->append_statement(std::move(store));
    return *this;
}

template<typename T> requires std::is_arithmetic_v<T>
template<typename U> requires is_cuj_arithmetic_v<U>
U ref<Arithmetic<T>>::as() const
{
    auto type_ctx = FunctionContext::get_func_context()->get_type_context();
    auto src_type = type_ctx->get_type<Arithmetic<T>>();
    auto dst_type = type_ctx->get_type<U>();
    core::ArithmeticCast cast = {
        .dst_type = dst_type ,
        .src_type = src_type,
        .src_val  = newRC<core::Expr>(_load())
    };
    return U::_from_expr(cast);
}

template<typename T> requires std::is_arithmetic_v<T>
Arithmetic<T> ref<Arithmetic<T>>::operator-() const
{
    auto type_ctx = FunctionContext::get_func_context()->get_type_context();
    auto type = type_ctx->get_type<Arithmetic<T>>();
    core::Unary unary = {
        .op       = core::Unary::Op::Neg,
        .val      = newRC<core::Expr>(_load()),
        .val_type = type
    };
    return Arithmetic<T>::_from_expr(unary);
}

template<typename T> requires std::is_arithmetic_v<T>
Arithmetic<T> ref<Arithmetic<T>>::operator+(const Arithmetic<T> &rhs) const
{
    auto type_ctx = FunctionContext::get_func_context()->get_type_context();
    auto type = type_ctx->get_type<Arithmetic<T>>();
    core::Binary binary = {
        .op       = core::Binary::Op::Add,
        .lhs      = newRC<core::Expr>(_load()),
        .rhs      = newRC<core::Expr>(rhs._load()),
        .lhs_type = type,
        .rhs_type = type
    };
    return Arithmetic<T>::_from_expr(binary);
}

template<typename T> requires std::is_arithmetic_v<T>
Arithmetic<T> ref<Arithmetic<T>>::operator-(const Arithmetic<T> &rhs) const
{
    auto type_ctx = FunctionContext::get_func_context()->get_type_context();
    auto type = type_ctx->get_type<Arithmetic<T>>();
    core::Binary binary = {
        .op       = core::Binary::Op::Sub,
        .lhs      = newRC<core::Expr>(_load()),
        .rhs      = newRC<core::Expr>(rhs._load()),
        .lhs_type = type,
        .rhs_type = type
    };
    return Arithmetic<T>::_from_expr(binary);
}

template<typename T> requires std::is_arithmetic_v<T>
Arithmetic<T> ref<Arithmetic<T>>::operator*(const Arithmetic<T> &rhs) const
{
    auto type_ctx = FunctionContext::get_func_context()->get_type_context();
    auto type = type_ctx->get_type<Arithmetic<T>>();
    core::Binary binary = {
        .op       = core::Binary::Op::Mul,
        .lhs      = newRC<core::Expr>(_load()),
        .rhs      = newRC<core::Expr>(rhs._load()),
        .lhs_type = type,
        .rhs_type = type
    };
    return Arithmetic<T>::_from_expr(binary);
}

template<typename T> requires std::is_arithmetic_v<T>
Arithmetic<T> ref<Arithmetic<T>>::operator/(const Arithmetic<T> &rhs) const
{
    auto type_ctx = FunctionContext::get_func_context()->get_type_context();
    auto type = type_ctx->get_type<Arithmetic<T>>();
    core::Binary binary = {
        .op       = core::Binary::Op::Div,
        .lhs      = newRC<core::Expr>(_load()),
        .rhs      = newRC<core::Expr>(rhs._load()),
        .lhs_type = type,
        .rhs_type = type
    };
    return Arithmetic<T>::_from_expr(binary);
}

template<typename T> requires std::is_arithmetic_v<T>
Arithmetic<T> ref<Arithmetic<T>>::operator%(const Arithmetic<T> &rhs) const
{
    static_assert(std::is_integral_v<T>);
    auto type_ctx = FunctionContext::get_func_context()->get_type_context();
    auto type = type_ctx->get_type<Arithmetic<T>>();
    core::Binary binary = {
        .op       = core::Binary::Op::Mod,
        .lhs      = newRC<core::Expr>(_load()),
        .rhs      = newRC<core::Expr>(rhs._load()),
        .lhs_type = type,
        .rhs_type = type
    };
    return Arithmetic<T>::_from_expr(binary);
}

template<typename T> requires std::is_arithmetic_v<T>
Arithmetic<bool> ref<Arithmetic<T>>::operator==(const Arithmetic<T> &rhs) const
{
    auto type = FunctionContext::get_func_context()
        ->get_type_context()->get_type<Arithmetic<T>>();
    return Arithmetic<bool>::_from_expr(core::Binary{
        .op       = core::Binary::Op::Equal,
        .lhs      = newRC<core::Expr>(_load()),
        .rhs      = newRC<core::Expr>(rhs._load()),
        .lhs_type = type,
        .rhs_type = type
    });
}

template<typename T> requires std::is_arithmetic_v<T>
Arithmetic<bool> ref<Arithmetic<T>>::operator!=(const Arithmetic<T> &rhs) const
{
    auto type = FunctionContext::get_func_context()
        ->get_type_context()->get_type<Arithmetic<T>>();
    return Arithmetic<bool>::_from_expr(core::Binary{
        .op       = core::Binary::Op::NotEqual,
        .lhs      = newRC<core::Expr>(_load()),
        .rhs      = newRC<core::Expr>(rhs._load()),
        .lhs_type = type,
        .rhs_type = type
    });
}

template<typename T> requires std::is_arithmetic_v<T>
Arithmetic<bool> ref<Arithmetic<T>>::operator<(const Arithmetic<T> &rhs) const
{
    auto type = FunctionContext::get_func_context()
        ->get_type_context()->get_type<Arithmetic<T>>();
    return Arithmetic<bool>::_from_expr(core::Binary{
        .op       = core::Binary::Op::Less,
        .lhs      = newRC<core::Expr>(_load()),
        .rhs      = newRC<core::Expr>(rhs._load()),
        .lhs_type = type,
        .rhs_type = type
    });
}

template<typename T> requires std::is_arithmetic_v<T>
Arithmetic<bool> ref<Arithmetic<T>>::operator<=(const Arithmetic<T> &rhs) const
{
    auto type = FunctionContext::get_func_context()
        ->get_type_context()->get_type<Arithmetic<T>>();
    return Arithmetic<bool>::_from_expr(core::Binary{
        .op       = core::Binary::Op::LessEqual,
        .lhs      = newRC<core::Expr>(_load()),
        .rhs      = newRC<core::Expr>(rhs._load()),
        .lhs_type = type,
        .rhs_type = type
    });
}

template<typename T> requires std::is_arithmetic_v<T>
Arithmetic<bool> ref<Arithmetic<T>>::operator>(const Arithmetic<T> &rhs) const
{
    auto type = FunctionContext::get_func_context()
        ->get_type_context()->get_type<Arithmetic<T>>();
    return Arithmetic<bool>::_from_expr(core::Binary{
        .op       = core::Binary::Op::Greater,
        .lhs      = newRC<core::Expr>(_load()),
        .rhs      = newRC<core::Expr>(rhs._load()),
        .lhs_type = type,
        .rhs_type = type
    });
}

template<typename T> requires std::is_arithmetic_v<T>
Arithmetic<bool> ref<Arithmetic<T>>::operator>=(const Arithmetic<T> &rhs) const
{
    auto type = FunctionContext::get_func_context()
        ->get_type_context()->get_type<Arithmetic<T>>();
    return Arithmetic<bool>::_from_expr(core::Binary{
        .op       = core::Binary::Op::GreaterEqual,
        .lhs      = newRC<core::Expr>(_load()),
        .rhs      = newRC<core::Expr>(rhs._load()),
        .lhs_type = type,
        .rhs_type = type
    });
}

template<typename T> requires std::is_arithmetic_v<T>
Arithmetic<T> ref<Arithmetic<T>>::operator>>(const Arithmetic<T> &rhs) const
{
    static_assert(std::is_integral_v<T> && !std::is_signed_v<T>);
    auto type = FunctionContext::get_func_context()
        ->get_type_context()->get_type<Arithmetic<T>>();
    return Arithmetic<T>::_from_expr(core::Binary{
        .op       = core::Binary::Op::RightShift,
        .lhs      = newRC<core::Expr>(this->_load()),
        .rhs      = newRC<core::Expr>(rhs._load()),
        .lhs_type = type,
        .rhs_type = type
    });
}

template<typename T> requires std::is_arithmetic_v<T>
Arithmetic<T> ref<Arithmetic<T>>::operator<<(const Arithmetic<T> &rhs) const
{
    static_assert(std::is_integral_v<T>);
    auto type = FunctionContext::get_func_context()
        ->get_type_context()->get_type<Arithmetic<T>>();
    return Arithmetic<T>::_from_expr(core::Binary{
        .op       = core::Binary::Op::LeftShift,
        .lhs      = newRC<core::Expr>(this->_load()),
        .rhs      = newRC<core::Expr>(rhs._load()),
        .lhs_type = type,
        .rhs_type = type
    });
}

template<typename T> requires std::is_arithmetic_v<T>
Arithmetic<T> ref<Arithmetic<T>>::operator&(const Arithmetic<T> &rhs) const
{
    static_assert(std::is_integral_v<T>);
    auto type = FunctionContext::get_func_context()
        ->get_type_context()->get_type<Arithmetic<T>>();
    return Arithmetic<T>::_from_expr(core::Binary{
        .op       = core::Binary::Op::BitwiseAnd,
        .lhs      = newRC<core::Expr>(this->_load()),
        .rhs      = newRC<core::Expr>(rhs._load()),
        .lhs_type = type,
        .rhs_type = type
    });
}

template<typename T> requires std::is_arithmetic_v<T>
Arithmetic<T> ref<Arithmetic<T>>::operator|(const Arithmetic<T> &rhs) const
{
    static_assert(std::is_integral_v<T>);
    auto type = FunctionContext::get_func_context()
        ->get_type_context()->get_type<Arithmetic<T>>();
    return Arithmetic<T>::_from_expr(core::Binary{
        .op       = core::Binary::Op::BitwiseOr,
        .lhs      = newRC<core::Expr>(this->_load()),
        .rhs      = newRC<core::Expr>(rhs._load()),
        .lhs_type = type,
        .rhs_type = type
    });
}

template<typename T> requires std::is_arithmetic_v<T>
Arithmetic<T> ref<Arithmetic<T>>::operator^(const Arithmetic<T> &rhs) const
{
    static_assert(std::is_integral_v<T>);
    auto type = FunctionContext::get_func_context()
        ->get_type_context()->get_type<Arithmetic<T>>();
    return Arithmetic<T>::_from_expr(core::Binary{
        .op       = core::Binary::Op::BitwiseXOr,
        .lhs      = newRC<core::Expr>(this->_load()),
        .rhs      = newRC<core::Expr>(rhs._load()),
        .lhs_type = type,
        .rhs_type = type
    });
}

template<typename T> requires std::is_arithmetic_v<T>
Pointer<Arithmetic<T>> ref<Arithmetic<T>>::address() const
{
    Pointer<Arithmetic<T>> ret;
    ret = addr_;
    return ret;
}

template<typename T> requires std::is_arithmetic_v<T>
core::Load ref<Arithmetic<T>>::_load() const
{
    auto type = FunctionContext::get_func_context()->get_type_context()
        ->get_type<Arithmetic<T>>();
    return core::Load{
        .val_type = type,
        .src_addr = newRC<core::Expr>(addr_._load())
    };
}

template<typename T> requires std::is_arithmetic_v<T>
ref<Arithmetic<T>> ref<Arithmetic<T>>::_from_ptr(const Pointer<Arithmetic<T>> &ptr)
{
    ref ret;
    ret.addr_ = ptr;
    return ret;
}

template<typename T>
Arithmetic<T> operator+(T lhs, const ref<Arithmetic<T>> &rhs)
{
    return Arithmetic(lhs) + rhs;
}

template<typename T>
Arithmetic<T> operator-(T lhs, const ref<Arithmetic<T>> &rhs)
{
    return Arithmetic(lhs) - rhs;
}

template<typename T>
Arithmetic<T> operator*(T lhs, const ref<Arithmetic<T>> &rhs)
{
    return Arithmetic(lhs) * rhs;
}

template<typename T>
Arithmetic<T> operator/(T lhs, const ref<Arithmetic<T>> &rhs)
{
    return Arithmetic(lhs) / rhs;
}

template<typename T>
Arithmetic<T> operator%(T lhs, const ref<Arithmetic<T>> &rhs)
{
    return Arithmetic(lhs) % rhs;
}

template<typename T> requires std::is_arithmetic_v<T>
Arithmetic<bool> operator==(T lhs, const ref<Arithmetic<T>> &rhs)
{
    return Arithmetic(lhs) == rhs;
}

template<typename T> requires std::is_arithmetic_v<T>
Arithmetic<bool> operator!=(T lhs, const ref<Arithmetic<T>> &rhs)
{
    return Arithmetic(lhs) != rhs;
}

template<typename T> requires std::is_arithmetic_v<T>
Arithmetic<bool> operator<(T lhs, const ref<Arithmetic<T>> &rhs)
{
    return Arithmetic(lhs) < rhs;
}

template<typename T> requires std::is_arithmetic_v<T>
Arithmetic<bool> operator<=(T lhs, const ref<Arithmetic<T>> &rhs)
{
    return Arithmetic(lhs) <= rhs;
}

template<typename T> requires std::is_arithmetic_v<T>
Arithmetic<bool> operator>(T lhs, const ref<Arithmetic<T>> &rhs)
{
    return Arithmetic(lhs) > rhs;
}

template<typename T> requires std::is_arithmetic_v<T>
Arithmetic<bool> operator>=(T lhs, const ref<Arithmetic<T>> &rhs)
{
    return Arithmetic(lhs) >= rhs;
}

inline Arithmetic<bool> operator!(const ref<Arithmetic<bool>> &val)
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
Arithmetic<T> operator>>(T lhs, const ref<Arithmetic<T>> &rhs)
{
    return Arithmetic(lhs) >> rhs;
}

template<typename T> requires std::is_integral_v<T>
Arithmetic<T> operator<<(T lhs, const ref<Arithmetic<T>> &rhs)
{
    return Arithmetic(lhs) << rhs;
}

template<typename T> requires std::is_integral_v<T>
Arithmetic<T> operator&(T lhs, const ref<Arithmetic<T>> &rhs)
{
    return Arithmetic(lhs) & rhs;
}

template<typename T> requires std::is_integral_v<T>
Arithmetic<T> operator|(T lhs, const ref<Arithmetic<T>> &rhs)
{
    return Arithmetic(lhs) | rhs;
}

template<typename T> requires std::is_integral_v<T>
Arithmetic<T> operator^(T lhs, const ref<Arithmetic<T>> &rhs)
{
    return Arithmetic(lhs) ^ rhs;
}

CUJ_NAMESPACE_END(cuj::dsl)
