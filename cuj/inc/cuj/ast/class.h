#pragma once

#include <cuj/ast/expr.h>
#include <cuj/ast/type_record.h>

CUJ_NAMESPACE_BEGIN(cuj::ast)

template<typename C>
class ClassBase
{
    StructTypeRecorder         *type_recorder_;
    RC<InternalPointerValue<C>> ref_pointer_;

    int member_count_ = 0;
    
public:

    struct CUJClassFlag { };

    using ClassAddress = RC<InternalPointerValue<C>>;

    explicit ClassBase(StructTypeRecorder *type_recorder);

    explicit ClassBase(ClassAddress ref_pointer);

    ClassBase(ClassAddress ref_pointer, UninitializeFlag);

    ClassBase(const ClassBase &) = delete;

    ClassBase &operator=(const ClassBase &other);

    template<typename T, typename...Args>
    RC<typename Value<T>::ImplType> new_member(Args &&...args);
};

CUJ_NAMESPACE_END(cuj::ast)
