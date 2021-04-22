#pragma once

#include <cuj/ast/expr.h>
#include <cuj/ast/type_record.h>

CUJ_NAMESPACE_BEGIN(cuj::ast)

template<typename C>
class ClassBase
{
    StructTypeRecorder                 *type_recorder_;
    RC<InternalArithmeticValue<size_t>> ref_pointer_;

    int member_count_ = 0;

protected:

    template<typename T, typename...Args>
    Value<T> new_member(Args &&...args);

public:

    struct CUJClassFlag { };

    using ClassAddress = RC<InternalArithmeticValue<size_t>>;

    explicit ClassBase(StructTypeRecorder *type_recorder);

    explicit ClassBase(ClassAddress ref_pointer);

    ClassBase(const ClassBase &) = delete;

    ClassBase &operator=(const ClassBase &other);
};

CUJ_NAMESPACE_END(cuj::ast)
