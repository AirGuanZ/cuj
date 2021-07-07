#pragma once

#include <functional>

#include <cuj/ast/stat.h>
#include <cuj/util/uncopyable.h>

CUJ_NAMESPACE_BEGIN(cuj::ast)

class IfBuilder : public Uncopyable
{
    struct ThenUnit
    {
        RC<InternalArithmeticValue<bool>> cond;
        RC<Block>                         block;
    };
    
    std::vector<ThenUnit> then_units_;
    RC<Block>             else_block_;

public:

    ~IfBuilder();

    template<typename T>
    IfBuilder &operator+(const ArithmeticValue<T> &cond);

    template<typename T>
    IfBuilder &operator+(const PointerImpl<T> &cond);

    IfBuilder &operator+(const std::function<void()> &then_body);

    IfBuilder &operator-(const std::function<void()> &else_body);
};

class WhileBuilder : public Uncopyable
{
    RC<Block>                         calc_cond_;
    RC<InternalArithmeticValue<bool>> cond_;
    RC<Block>                         body_;

    template<typename T>
    void init_cond(const ArithmeticValue<T> &cond);

    template<typename T>
    void init_cond(const PointerImpl<T> &cond);

    template<typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
    void init_cond(T cond);

public:

    template<typename F>
    explicit WhileBuilder(const F &calc_cond_func);

    template<typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
    explicit WhileBuilder(T cond) : WhileBuilder(create_literial(cond)) { }

    ~WhileBuilder();

    void operator+(const std::function<void()> &body_func);
};

class ReturnBuilder : public Uncopyable
{
public:

    ReturnBuilder();

    template<typename T>
    ReturnBuilder(const ArithmeticValue<T> &val);

    template<typename T>
    ReturnBuilder(const PointerImpl<T> &val);

    template<typename T>
    ReturnBuilder(const ClassValue<T> &val);

    template<typename T, size_t N>
    ReturnBuilder(const ArrayImpl<T, N> &val);

    template<typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
    ReturnBuilder(T val);
};

CUJ_NAMESPACE_END(cuj::ast)
