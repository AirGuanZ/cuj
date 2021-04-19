#pragma once

#define CUJ_MACRO_OVERLOADING_RETURN_ARG_COUNT(_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, CNT, ...) CNT
#define CUJ_MACRO_OVERLOADING_EXPAND_ARGS(ARGS)                                 \
    CUJ_MACRO_OVERLOADING_RETURN_ARG_COUNT ARGS
#define CUJ_MACRO_OVERLOADING_COUNT_ARGS(...)                                   \
    CUJ_MACRO_OVERLOADING_EXPAND_ARGS(                                          \
        (__VA_ARGS__, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0))

#define CUJ_MACRO_OVERLOADING_OVERLOAD_MACRO2(NAME, COUNT) NAME##COUNT
#define CUJ_MACRO_OVERLOADING_OVERLOAD_MACRO1(NAME, COUNT)                      \
    CUJ_MACRO_OVERLOADING_OVERLOAD_MACRO2(NAME, COUNT)
#define CUJ_MACRO_OVERLOADING_OVERLOAD_MACRO(NAME, COUNT)                       \
    CUJ_MACRO_OVERLOADING_OVERLOAD_MACRO1(NAME, COUNT)

#define CUJ_MACRO_OVERLOADING_CAT(X, Y) X Y

#define CUJ_MACRO_OVERLOADING(NAME, ...)                                        \
    CUJ_MACRO_OVERLOADING_CAT(                                                  \
        CUJ_MACRO_OVERLOADING_OVERLOAD_MACRO(                                   \
            NAME, CUJ_MACRO_OVERLOADING_COUNT_ARGS(__VA_ARGS__)),               \
        (__VA_ARGS__))

#define CUJ_DECL_TEMPLATE_TYPENAMES1(INTERNAL)          template<>
#define CUJ_DECL_TEMPLATE_TYPENAMES2(INTERNAL, X)       template<typename X>
#define CUJ_DECL_TEMPLATE_TYPENAMES3(INTERNAL, X, Y)    template<typename X, typename Y>
#define CUJ_DECL_TEMPLATE_TYPENAMES4(INTERNAL, X, Y, Z) template<typename X, typename Y, typename Z>

#define CUJ_DECL_TEMPLATE_ARGUMENTS1(CLASS)          CLASS
#define CUJ_DECL_TEMPLATE_ARGUMENTS2(CLASS, X)       CLASS<X>
#define CUJ_DECL_TEMPLATE_ARGUMENTS3(CLASS, X, Y)    CLASS<X, Y>
#define CUJ_DECL_TEMPLATE_ARGUMENTS4(CLASS, X, Y, Z) CLASS<X, Y, Z>

#define CUJ_DECL_TEMPLATE_TYPENAMES(...)                                        \
    CUJ_MACRO_OVERLOADING(CUJ_DECL_TEMPLATE_TYPENAMES, __VA_ARGS__)

#define CUJ_DECL_TEMPLATE_ARGUMENTS(...)                                        \
    CUJ_MACRO_OVERLOADING(CUJ_DECL_TEMPLATE_ARGUMENTS, __VA_ARGS__)
