#pragma once

#include <cuj/dsl/arithmetic.h>
#include <cuj/utils/uncopyable.h>

CUJ_NAMESPACE_BEGIN(cuj::dsl)

class IfBuilder : public Uncopyable
{
    struct ThenUnit
    {
        RC<core::Block> cond_calc;
        core::Expr      cond;
        RC<core::Stat>  body;
    };

    std::vector<ThenUnit> then_units_;
    RC<core::Stat>        else_body_;

public:

    ~IfBuilder();

    template<typename F>
    IfBuilder &operator*(F &&cond_func);

    template<typename F>
    IfBuilder &operator/(F &&then_func);

    template<typename F>
    void operator-(F &&else_func);
};

#define CUJ_IF(COND)                                                            \
    ::cuj::dsl::IfBuilder()                                                     \
    *[&]()->::cuj::dsl::Arithmetic<bool>{return (COND);}/[&]()->void
#define CUJ_ELIF(COND)                                                          \
    *[&]()->::cuj::dsl::Arithmetic<bool>{return (COND);}/[&]()->void
#define CUJ_ELSE -[&]()->void

#define $if   CUJ_IF
#define $elif CUJ_ELIF
#define $else CUJ_ELSE

CUJ_NAMESPACE_END(cuj::dsl)
