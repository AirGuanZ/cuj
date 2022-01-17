#pragma once

#include <cuj/dsl/variable.h>
#include <cuj/dsl/variable_forward.h>

CUJ_NAMESPACE_BEGIN(cuj::dsl)

// num -> num

template<typename To, typename From>
    requires is_cuj_arithmetic_v<To> &&
             is_cuj_arithmetic_v<From> &&
             (sizeof(typename To::RawType) == sizeof(typename From::RawType))
To bitcast(From from);

// ptr to ptr

template<typename To, typename From>
    requires is_cuj_pointer_v<To> && is_cuj_pointer_v<From>
To bitcast(From from);

// num -> ptr

template<typename To>
    requires is_cuj_pointer_v<To>
To bitcast(num<uint64_t> from);
    
template<typename To>
    requires is_cuj_pointer_v<To>
To bitcast(num<int64_t> from);

// ptr -> num

template<typename To, typename From>
    requires is_cuj_arithmetic_v<To> && is_cuj_pointer_v<From>
To bitcast(From from);

// ref -> *

template<typename To, typename From>
    requires is_cuj_ref_v<From>
To bitcast(From from);

// var -> *

template<typename To, typename From>
To bitcast(var<From> from);

CUJ_NAMESPACE_END(cuj::dsl)
