#pragma once

#include <cuj/dsl/arithmetic_reference.h>
#include <cuj/dsl/array_reference.h>
#include <cuj/dsl/class.h>
#include <cuj/dsl/pointer_reference.h>

CUJ_NAMESPACE_BEGIN(cuj::dsl)

template<typename T>
class ref;

template<typename T>
class var;

template<typename T>
ref(num<T>)->ref<num<T>>;

template<typename T>
ref(ptr<T>)->ref<ptr<T>>;

template<typename T, size_t N>
ref(arr<T, N>)->ref<arr<T, N>>;

template<typename T> requires is_cuj_class_v<T>
ref(var<T>)->ref<T>;

template<typename T> requires is_cuj_class_v<T>
ref(T)->ref<T>;

CUJ_NAMESPACE_END(cuj::dsl)
