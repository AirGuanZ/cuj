#pragma once

#include <functional>
#include <sstream>

#include <cuj/dsl/function.h>

CUJ_NAMESPACE_BEGIN(cuj)

class TextBuilder
{
public:

    void set_indent_unit(std::string unit);

    void new_line();

    void push_indent();

    void pop_indent();

    void with_indent(std::function<void()> func);

    template<typename...Args>
    void append(Args &&...args);

    template<typename...Args>
    void appendl(Args &&...args);

    std::string get_str() const;

private:

    int         indent_cnt_  = 0;
    std::string indent_unit_ = "    ";
    std::string indent_str_;

    bool              newline_ = true;
    std::stringstream ss_;
};

class Printer
{
public:
    
    template<typename F>
    std::string print(const dsl::Function<F> &function);

private:

    void print(TextBuilder &b, const dsl::FunctionContext &function);

    // statements

    void print(TextBuilder &b, const core::Stat &s);

    void print(TextBuilder &b, const core::Store &store);

    void print(TextBuilder &b, const core::Copy &copy);

    void print(TextBuilder &b, const core::Block &block);

    void print(TextBuilder &b, const core::Return &ret);

    void print(TextBuilder &b, const core::If &stat);

    void print(TextBuilder &b, const core::Loop &stat);

    void print(TextBuilder &b, const core::Break &stat);

    void print(TextBuilder &b, const core::Continue &stat);

    void print(TextBuilder &b, const core::Switch &stat);

    void print(TextBuilder &b, const core::CallFuncStat &call);

    // expression

    void print(TextBuilder &b, const core::Expr &e);

    void print(TextBuilder &b, const core::FuncArgAddr &addr);

    void print(TextBuilder &b, const core::LocalAllocAddr &addr);

    void print(TextBuilder &b, const core::Load &load);

    void print(TextBuilder &b, const core::Immediate &imm);

    void print(TextBuilder &b, const core::NullPtr &null_ptr);

    void print(TextBuilder &b, const core::ArithmeticCast &cast);

    void print(TextBuilder &b, const core::BitwiseCast &cast);

    void print(TextBuilder &b, const core::PointerOffset &ptr_offset);

    void print(TextBuilder &b, const core::ClassPointerToMemberPointer &mem);

    void print(TextBuilder &b, const core::DerefClassPointer &deref);

    void print(TextBuilder &b, const core::DerefArrayPointer &deref);

    void print(TextBuilder &b, const core::SaveClassIntoLocalAlloc &deref);

    void print(TextBuilder &b, const core::SaveArrayIntoLocalAlloc &deref);

    void print(TextBuilder &b, const core::ArrayAddrToFirstElemAddr &to);

    void print(TextBuilder &b, const core::Binary &binary);

    void print(TextBuilder &b, const core::Unary &unary);

    void print(TextBuilder &b, const core::CallFunc &call);

    // type

    void print(TextBuilder &b, const core::Type &type);

    void print(TextBuilder &b, core::Builtin builtin);

    void print(TextBuilder &b, const core::Struct &s);

    void print(TextBuilder &b, const core::Array &a);

    void print(TextBuilder &b, const core::Pointer &p);
};

template<typename...Args>
void TextBuilder::append(Args &&...args)
{
    if(newline_)
    {
        ss_ << indent_str_;
        newline_ = false;
    }
    ((ss_ << args), ...);
}

template<typename...Args>
void TextBuilder::appendl(Args &&...args)
{
    append(std::forward<Args>(args)...);
    new_line();
}

template<typename F>
std::string Printer::print(const dsl::Function<F> &function)
{
    TextBuilder builder;
    print(builder, *function._get_context());
    return builder.get_str();
}

CUJ_NAMESPACE_END(cuj)
