#pragma once

#include <functional>

#include <cuj/core/prog.h>

CUJ_NAMESPACE_BEGIN(cuj::core)

class Visitor
{
public:

    void visit(const Stat         &stat);
    void visit(const Store        &store);
    void visit(const Block        &block);
    void visit(const Return       &ret);
    void visit(const If           &if_s);
    void visit(const Loop         &loop);
    void visit(const Break        &break_s);
    void visit(const Continue     &continue_s);
    void visit(const Switch       &switch_s);
    void visit(const CallFuncStat &call);

    void visit(const Expr                        &expr);
    void visit(const FuncArgAddr                 &expr);
    void visit(const LocalAllocAddr              &expr);
    void visit(const Load                        &expr);
    void visit(const Immediate                   &expr);
    void visit(const NullPtr                     &expr);
    void visit(const ArithmeticCast              &expr);
    void visit(const PointerOffset               &expr);
    void visit(const ClassPointerToMemberPointer &expr);
    void visit(const DerefClassPointer           &expr);
    void visit(const DerefArrayPointer           &expr);
    void visit(const SaveClassIntoLocalAlloc     &expr);
    void visit(const SaveArrayIntoLocalAlloc     &expr);
    void visit(const ArrayAddrToFirstElemAddr    &expr);
    void visit(const Binary                      &expr);
    void visit(const Unary                       &expr);
    void visit(const CallFunc                    &expr);

    std::function<void(const Stat &)>         on_stat;
    std::function<void(const Store &)>        on_store;
    std::function<void(const Block &)>        on_block;
    std::function<void(const Return &)>       on_return;
    std::function<void(const If &)>           on_if;
    std::function<void(const Loop &)>         on_loop;
    std::function<void(const Break &)>        on_break;
    std::function<void(const Continue &)>     on_continue;
    std::function<void(const Switch &)>       on_switch;
    std::function<void(const CallFuncStat &)> on_call_func_stat;
    
    std::function<void(const Expr                        &)> on_expr;
    std::function<void(const FuncArgAddr                 &)> on_func_arg_addr;
    std::function<void(const LocalAllocAddr              &)> on_local_alloc_addr;
    std::function<void(const Load                        &)> on_load;
    std::function<void(const Immediate                   &)> on_immediate;
    std::function<void(const NullPtr                     &)> on_nullptr;
    std::function<void(const ArithmeticCast              &)> on_arithmetic_cast;
    std::function<void(const PointerOffset               &)> on_pointer_offset;
    std::function<void(const ClassPointerToMemberPointer &)> on_class_ptr_to_member_ptr;
    std::function<void(const DerefClassPointer           &)> on_deref_class_ptr;
    std::function<void(const DerefArrayPointer           &)> on_deref_array_ptr;
    std::function<void(const SaveClassIntoLocalAlloc     &)> on_save_class_into_local_alloc;
    std::function<void(const SaveArrayIntoLocalAlloc     &)> on_save_array_into_local_alloc;
    std::function<void(const ArrayAddrToFirstElemAddr    &)> on_array_ptr_to_first_elem_ptr;
    std::function<void(const Binary                      &)> on_binary;
    std::function<void(const Unary                       &)> on_unary;
    std::function<void(const CallFunc                    &)> on_call_func;
};

CUJ_NAMESPACE_END(cuj::core)
