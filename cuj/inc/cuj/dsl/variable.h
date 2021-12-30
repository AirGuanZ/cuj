#pragma once

#include <cuj/dsl/arithmetic.h>
#include <cuj/dsl/array.h>
#include <cuj/dsl/pointer.h>

CUJ_NAMESPACE_BEGIN(cuj::dsl)

template<typename T>
class var;

namespace var_detail
{

    template<typename T>
    struct RemoveVarWrapper
    {
        using Type = T;
    };

    template<typename T>
    struct RemoveVarWrapper<var<T>>
    {
        using Type = T;
    };

} // namespace var_detail

template<typename T>
using remove_var_wrapper_t = typename var_detail::RemoveVarWrapper<T>::Type;

template<typename T>
class var<num<T>> : public num<T>
{
public:

    using num<T>::num;

    var(num<T> other) : num<T>(std::move(other)) { }
};

template<typename T>
class var<ptr<T>> : public ptr<T>
{
public:

    using ptr<T>::ptr;

    var(ptr<T> other) : ptr<T>(std::move(other)) { }
};

template<typename T, size_t N>
class var<arr<T, N>> : public arr<T, N>
{
public:

    using arr<T, N>::arr;

    var(arr<T, N> other) : arr<T, N>(std::move(other)) { }
};

template<typename T> requires is_cuj_class_v<T>
class var<T> : public T
{
public:

    using T::T;

    var(T other) : T(std::move(other)) { }
};

template<typename T> requires std::is_arithmetic_v<T>
var(T)->var<num<T>>;

template<typename T> requires std::is_same_v<T, std::nullptr_t>
var(T)->var<ptr<CujVoid>>;

template<typename T>
var(num<T>)->var<num<T>>;

template<typename T>
var(ptr<T>)->var<ptr<T>>;

template<typename T, size_t N>
var(arr<T, N>)->var<arr<T, N>>;

template<typename T> requires is_cuj_class_v<T>
var(T)->var<T>;

template<typename T> requires is_cuj_class_v<T>
var(var<T>)->var<T>;

template<typename T>
var(ref<num<T>>)->var<num<T>>;

template<typename T>
var(ref<ptr<T>>)->var<ptr<T>>;

template<typename T, size_t N>
var(ref<arr<T, N>>)->var<arr<T, N>>;

template<typename T> requires is_cuj_class_v<T>
var(ref<T>)->var<T>;

CUJ_NAMESPACE_END(cuj::dsl)
