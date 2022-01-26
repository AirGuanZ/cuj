#pragma once

#include <cuj/dsl/dsl.h>

CUJ_NAMESPACE_BEGIN(cuj::cstd)

inline void _assert_fail(
    ptr<char_t> message, ptr<char_t> file, i32 line, ptr<char_t> function);

#define CUJ_ASSERT(COND) CUJ_ASSERT_IMPL(COND)
#define CUJ_ASSERT_IMPL(COND)                                                   \
    do {                                                                        \
        $if(!(COND))                                                            \
        {                                                                       \
            ::cuj::cstd::_assert_fail(                                          \
                ::cuj::dsl::string_literial(#COND),                             \
                ::cuj::dsl::string_literial(__FILE__),                          \
                ::cuj::i32(__LINE__),                                           \
                ::cuj::dsl::string_literial(__FUNCTION__));                     \
        };                                                                      \
    } while(false)

CUJ_NAMESPACE_END(cuj::cstd)

#include <cuj/cstd/impl/assert.inl>
