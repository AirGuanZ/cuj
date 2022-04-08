#pragma once

#include <span>

#include <cuj/dsl/variable_forward.h>

CUJ_NAMESPACE_BEGIN(cuj::dsl)

template<typename T>
ptr<T> const_data(const void *data, size_t bytes, size_t alignment = 1);

template<typename T>
ptr<cxx<std::remove_cvref_t<T>>> const_data(std::span<const T> data);

template<typename T>
ptr<cxx<std::remove_cvref_t<T>>> const_data(const std::vector<T> &data);

ptr<num<char>> string_literial(const std::string &str);

CUJ_NAMESPACE_END(cuj::dsl)
