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

    IfBuilder &operator+(const std::function<void()> &then_body);

    IfBuilder &operator-(const std::function<void()> &else_body);
};

class WhileBuilder : public Uncopyable
{
    RC<InternalArithmeticValue<bool>> cond_;
    RC<Block>                         block_;

public:

    template<typename T>
    explicit WhileBuilder(const ArithmeticValue<T> &cond);

    ~WhileBuilder();

    void operator+(const std::function<void()> &body_func);
};

CUJ_NAMESPACE_END(cuj::ast)
