#pragma once

#include <sstream>
#include <string>

#include <cuj/util/uncopyable.h>

CUJ_NAMESPACE_BEGIN(cuj)

class IndentedStringBuilder : public Uncopyable
{
public:

    void set_single_indent(std::string indent)
    {
        indent_ = std::move(indent);
    }

    void push_indent()
    {
        ++indent_count_;
        update_full_indent();
    }

    void pop_indent()
    {
        CUJ_INTERNAL_ASSERT(indent_count_ > 0);
        --indent_count_;
        update_full_indent();
    }

    void new_line()
    {
        stream_ << "\n";
        is_current_line_empty_ = true;
    }

    template<typename...Args>
    void append(const Args &...args)
    {
        (append_single(args), ...);
    }

    std::string get_result() const
    {
        return stream_.str();
    }

private:

    void update_full_indent()
    {
        full_indent_ = {};
        for(int i = 0; i < indent_count_; ++i)
            full_indent_ += indent_;
    }

    template<typename T>
    void append_single(const T &s)
    {
        if constexpr(std::is_same_v<T, std::string>)
            append_single_string(s);
        else if constexpr(std::is_same_v<T, const char *> ||
                          std::is_array_v<T> ||
                          std::is_same_v<T, std::string_view>)
            append_single_string(std::string(s));
        else
        {
            std::stringstream ss;
            ss.precision(40);
            ss << std::hexfloat << s;
            append_single_string(ss.str());
        }
    }

    void append_single_string(const std::string &s)
    {
        if(is_current_line_empty_ && !s.empty())
        {
            stream_ << full_indent_;
            is_current_line_empty_ = false;
        }
        stream_ << s;
    }

    std::string indent_ = "    ";
    int indent_count_   = 0;

    std::string full_indent_;

    bool is_current_line_empty_ = true;
    std::stringstream stream_;
};

CUJ_NAMESPACE_END(cuj)
