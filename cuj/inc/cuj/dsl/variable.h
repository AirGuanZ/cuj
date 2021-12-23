#pragma once

#include <cuj/dsl/arithmetic.h>
#include <cuj/dsl/array.h>
#include <cuj/dsl/pointer.h>

CUJ_NAMESPACE_BEGIN(cuj::dsl)

template<typename T>
class Variable;

template<typename T>
class Variable<Arithmetic<T>> : public Arithmetic<T>
{
public:

    using Arithmetic<T>::Arithmetic;

    Variable(Arithmetic<T> other) : Arithmetic<T>(std::move(other)) { }
};

template<typename T>
class Variable<Pointer<T>> : public Pointer<T>
{
public:

    using Pointer<T>::Pointer;

    Variable(Pointer<T> other) : Pointer<T>(std::move(other)) { }
};

template<typename T, size_t N>
class Variable<Array<T, N>> : public Array<T, N>
{
public:

    using Array<T, N>::Array;

    Variable(Array<T, N> other) : Array<T, N>(std::move(other)) { }
};

template<typename T> requires is_cuj_class_v<T>
class Variable<T> : public T
{
public:

    using T::T;

    Variable(T other) : T(std::move(other)) { }
};

template<typename T> requires std::is_arithmetic_v<T>
Variable(T)->Variable<Arithmetic<T>>;

template<typename T> requires std::is_same_v<T, std::nullptr_t>
Variable(T)->Variable<Pointer<CujVoid>>;

template<typename T>
Variable(Arithmetic<T>)->Variable<Arithmetic<T>>;

template<typename T>
Variable(Pointer<T>)->Variable<Pointer<T>>;

template<typename T, size_t N>
Variable(Array<T, N>)->Variable<Array<T, N>>;

template<typename T> requires is_cuj_class_v<T>
Variable(T)->Variable<T>;

template<typename T> requires is_cuj_class_v<T>
Variable(Variable<T>)->Variable<T>;

template<typename T>
Variable(ref<Arithmetic<T>>)->Variable<Arithmetic<T>>;

template<typename T>
Variable(ref<Pointer<T>>)->Variable<Pointer<T>>;

template<typename T, size_t N>
Variable(ref<Array<T, N>>)->Variable<Array<T, N>>;

template<typename T> requires is_cuj_class_v<T>
Variable(ref<T>)->Variable<T>;

using var = Variable;

CUJ_NAMESPACE_END(cuj::dsl)
