#pragma once

#include <cuj/core/stat.h>
#include <cuj/dsl/arithmetic.h>
#include <cuj/utils/uncopyable.h>

CUJ_NAMESPACE_BEGIN(cuj::dsl)

class SwitchBuilder : public Uncopyable
{
    core::Builtin val_type_;
    core::Switch  switch_s_;

public:

    static SwitchBuilder *get_current_switch_builder();
    
    static void set_current_switch_builder(SwitchBuilder *builder);

    template<typename T> requires std::is_integral_v<T>
    explicit SwitchBuilder(const Arithmetic<T> &value);

    template<typename F>
    void operator*(F &&stat_func);

    template<typename T> requires std::is_integral_v<T>
    SwitchBuilder &add_case(T cond);

    template<typename F>
    void operator+(F &&case_body_func);

    void set_current_body_fallthrough();

    template<typename F>
    void operator-(F &&default_body_func);
};

#define CUJ_SWITCH(VAL) ::cuj::dsl::SwitchBuilder(VAL)*[&]()->void
#define CUJ_CASE(COND)                                                          \
    ::cuj::dsl::SwitchBuilder::get_current_switch_builder()                     \
                                ->add_case(COND)+[&]()->void
#define CUJ_DEFAULT                                                             \
    (*::cuj::dsl::SwitchBuilder::get_current_switch_builder())-[&]()->void
#define CUJ_FALLTHROUGH                                                         \
    ::cuj::dsl::SwitchBuilder::get_current_switch_builder()                     \
                                ->set_current_body_fallthrough()

#define $switch      CUJ_SWITCH
#define $case        CUJ_CASE
#define $default     CUJ_DEFAULT
#define $fallthrough CUJ_FALLTHROUGH

CUJ_NAMESPACE_END(cuj::dsl)
