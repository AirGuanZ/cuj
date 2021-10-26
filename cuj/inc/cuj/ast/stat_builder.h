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

class SwitchBuilderInterface
{
public:

    using Int = Variant<
        int8_t, int16_t, int32_t, int64_t,
        uint8_t, uint16_t, uint32_t, uint64_t>;

    virtual ~SwitchBuilderInterface() = default;

    virtual void new_case(Int cond, RC<Block> body, bool fallthrough) = 0;

    virtual void new_default(RC<Block> block) = 0;

    static SwitchBuilderInterface *get_current_builder();

protected:

    static SwitchBuilderInterface *&current_builder_storage();

    static void set_current_builder(SwitchBuilderInterface *builder);
};

template<typename T>
class SwitchBuilder : public Uncopyable, public SwitchBuilderInterface
{
    static_assert(std::is_integral_v<T>);

    using Case = typename Switch<T>::Case;

    RC<InternalArithmeticValue<T>> value_;
    std::vector<Case>              cases_;
    RC<Block>                      default_;

    Case *recording_case_ = nullptr;

public:

    SwitchBuilder(const ArithmeticValue<T> &val);

    void operator+(const std::function<void()> &body);

    void new_case(Int cond, RC<Block> body, bool fallthrough) override;

    void new_default(RC<Block> block) override;
};

class SwitchCaseBuilderInterface
{
public:

    virtual ~SwitchCaseBuilderInterface() = default;

    virtual void set_fallthrough() = 0;

    static SwitchCaseBuilderInterface *get_current_builder();

protected:

    static SwitchCaseBuilderInterface *&current_builder_storage();

    static void set_current_builder(SwitchCaseBuilderInterface *builder);
};

template<typename T>
class SwitchCaseBuilder : public Uncopyable, public SwitchCaseBuilderInterface
{
    static_assert(std::is_integral_v<T>);

    T    value_;
    bool fallthrough_;

public:

    SwitchCaseBuilder(T value);

    void operator+(const std::function<void()> &body_func);

    void set_fallthrough() override;
};

class SwitchDefaultBuilder : public Uncopyable
{
public:

    void operator+(const std::function<void()> &body_func);
};

CUJ_NAMESPACE_END(cuj::ast)
