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
class var<Arithmetic<T>> : public Arithmetic<T>
{
public:

    using Arithmetic<T>::Arithmetic;

    var(Arithmetic<T> other) : Arithmetic<T>(std::move(other)) { }
};

template<typename T>
class var<Pointer<T>> : public Pointer<T>
{
public:

    using Pointer<T>::Pointer;

    var(Pointer<T> other) : Pointer<T>(std::move(other)) { }
};

template<typename T, size_t N>
class var<Array<T, N>> : public Array<T, N>
{
public:

    using Array<T, N>::Array;

    var(Array<T, N> other) : Array<T, N>(std::move(other)) { }
};

template<typename T> requires is_cuj_class_v<T>
class var<T> : public T
{
public:

    using T::T;

    var(T other) : T(std::move(other)) { }
};

template<typename T> requires std::is_arithmetic_v<T>
var(T)->var<Arithmetic<T>>;

template<typename T> requires std::is_same_v<T, std::nullptr_t>
var(T)->var<Pointer<CujVoid>>;

template<typename T>
var(Arithmetic<T>)->var<Arithmetic<T>>;

template<typename T>
var(Pointer<T>)->var<Pointer<T>>;

template<typename T, size_t N>
var(Array<T, N>)->var<Array<T, N>>;

template<typename T> requires is_cuj_class_v<T>
var(T)->var<T>;

template<typename T> requires is_cuj_class_v<T>
var(var<T>)->var<T>;

template<typename T>
var(ref<Arithmetic<T>>)->var<Arithmetic<T>>;

template<typename T>
var(ref<Pointer<T>>)->var<Pointer<T>>;

template<typename T, size_t N>
var(ref<Array<T, N>>)->var<Array<T, N>>;

template<typename T> requires is_cuj_class_v<T>
var(ref<T>)->var<T>;

CUJ_NAMESPACE_END(cuj::dsl)
