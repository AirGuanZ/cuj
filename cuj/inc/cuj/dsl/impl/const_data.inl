#pragma once

#include <cuj/core/expr.h>
#include <cuj/dsl/const_data.h>
#include <cuj/dsl/function.h>
#include <cuj/dsl/pointer.h>

CUJ_NAMESPACE_BEGIN(cuj::dsl)

template<typename T>
ptr<T> const_data(const void *data, size_t bytes, size_t alignment)
{
    auto type_ctx = FunctionContext::get_func_context()->get_type_context();
    auto pointed_type = type_ctx->get_type<T>();
    return ptr<T>::_from_expr(core::GlobalConstAddr{
        .pointed_type = pointed_type,
        .alignment    = alignment,
        .data         = std::vector<unsigned char>{
            static_cast<const unsigned char *>(data),
            static_cast<const unsigned char *>(data) + bytes,
        }
    });
}

template<typename T>
ptr<cxx<std::remove_cvref_t<T>>> const_data(std::span<T> data)
{
    return const_data<cxx<std::remove_cvref_t<T>>>(data.data(), data.size() * sizeof(T), alignof(T));
}

inline ptr<num<char>> string_literial(const std::string &str)
{
    return const_data(std::span{ str.data(), str.data() + str.size() + 1 });
}

CUJ_NAMESPACE_END(cuj::dsl)
