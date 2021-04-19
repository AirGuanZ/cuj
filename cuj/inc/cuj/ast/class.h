#pragma once

#include <cuj/ast/expr.h>
#include <cuj/ast/type_record.h>

CUJ_NAMESPACE_BEGIN(cuj::ast)

template<typename C>
class ClassBase
{
    struct MemberRecordBase
    {
        virtual ~MemberRecordBase() = default;
    };

    template<typename T>
    struct MemberRecord : MemberRecordBase
    {
        Value<T> member;

        explicit MemberRecord(const Value<T> &member_value)
            : member(member_value)
        {
            
        }
    };

    StructTypeRecorder                 *type_recorder_;
    RC<InternalArithmeticValue<size_t>> ref_pointer_;

    std::vector<Box<MemberRecordBase>> member_records_;

    template<typename T>
    Value<T> commit_member_record(Value<T> member);

protected:

    template<typename T, typename...Args>
    Value<T> new_member(Args &&...args);

public:

    struct CUJClassFlag { };
    
    explicit ClassBase(StructTypeRecorder *type_recorder);

    explicit ClassBase(RC<InternalArithmeticValue<size_t>> ref_pointer);

    ClassBase(const ClassBase &) = delete;

    ClassBase &operator=(const ClassBase &other);
};

CUJ_NAMESPACE_END(cuj::ast)
