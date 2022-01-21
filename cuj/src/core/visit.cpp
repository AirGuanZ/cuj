#include <cuj/core/visit.h>

CUJ_NAMESPACE_BEGIN(cuj::core)

void Visitor::visit(const Stat &stat)
{
    if(on_stat)
        on_stat(stat);
    stat.match([&](auto &_s) { visit(_s); });
}

void Visitor::visit(const Store &store)
{
    if(on_store)
        on_store(store);
    visit(store.dst_addr);
    visit(store.val);
}

void Visitor::visit(const Copy &copy)
{
    if(on_copy)
        on_copy(copy);
    visit(copy.dst_addr);
    visit(copy.src_addr);
}

void Visitor::visit(const Block &block)
{
    if(on_block)
        on_block(block);
    for(auto &s : block.stats)
        visit(*s);
}

void Visitor::visit(const Return &ret)
{
    if(on_return)
        on_return(ret);
    visit(ret.val);
}

void Visitor::visit(const If &if_s)
{
    if(on_if)
        on_if(if_s);
    visit(if_s.cond);
    visit(*if_s.then_body);
    if(if_s.else_body)
        visit(*if_s.else_body);
}

void Visitor::visit(const Loop &loop)
{
    if(on_loop)
        on_loop(loop);
    visit(*loop.body);
}

void Visitor::visit(const Break &break_s)
{
    if(on_break)
        on_break(break_s);
}

void Visitor::visit(const Continue &continue_s)
{
    if(on_continue)
        on_continue(continue_s);
}

void Visitor::visit(const Switch &switch_s)
{
    if(on_switch)
        on_switch(switch_s);
    for(auto &b : switch_s.branches)
    {
        visit(b.cond);
        visit(*b.body);
    }
    if(switch_s.default_body)
        visit(*switch_s.default_body);
}

void Visitor::visit(const CallFuncStat &call)
{
    if(on_call_func_stat)
        on_call_func_stat(call);
    visit(call.call_expr);
}

void Visitor::visit(const MakeScope &make_scope)
{
    if(on_make_scope)
        on_make_scope(make_scope);
    visit(*make_scope.body);
}

void Visitor::visit(const ExitScope &exit_scope)
{
    if(on_exit_scope)
        on_exit_scope(exit_scope);
}

void Visitor::visit(const InlineAsm &inline_asm)
{
    if(on_inline_asm)
        on_inline_asm(inline_asm);
    for(auto &i : inline_asm.input_values)
        visit(i);
    for(auto &o : inline_asm.output_addresses)
        visit(o);
}

void Visitor::visit(const Expr &expr)
{
    if(on_expr)
        on_expr(expr);
    expr.match([&](auto &_e) { visit(_e); });
}

void Visitor::visit(const FuncArgAddr &expr)
{
    if(on_func_arg_addr)
        on_func_arg_addr(expr);
}

void Visitor::visit(const LocalAllocAddr &expr)
{
    if(on_local_alloc_addr)
        on_local_alloc_addr(expr);
}

void Visitor::visit(const Load &expr)
{
    if(on_load)
        on_load(expr);
    visit(*expr.src_addr);
}

void Visitor::visit(const Immediate &expr)
{
    if(on_immediate)
        on_immediate(expr);
}

void Visitor::visit(const NullPtr &expr)
{
    if(on_nullptr)
        on_nullptr(expr);
}

void Visitor::visit(const ArithmeticCast &expr)
{
    if(on_arithmetic_cast)
        on_arithmetic_cast(expr);
    visit(*expr.src_val);
}

void Visitor::visit(const BitwiseCast &expr)
{
    if(on_bitwise_cast)
        on_bitwise_cast(expr);
    visit(*expr.src_val);
}

void Visitor::visit(const PointerOffset &expr)
{
    if(on_pointer_offset)
        on_pointer_offset(expr);
    visit(*expr.ptr_val);
    visit(*expr.offset_val);
}

void Visitor::visit(const ClassPointerToMemberPointer &expr)
{
    if(on_class_ptr_to_member_ptr)
        on_class_ptr_to_member_ptr(expr);
    visit(*expr.class_ptr);
}

void Visitor::visit(const DerefClassPointer &expr)
{
    if(on_deref_class_ptr)
        on_deref_class_ptr(expr);
    visit(*expr.class_ptr);
}

void Visitor::visit(const DerefArrayPointer &expr)
{
    if(on_deref_array_ptr)
        on_deref_array_ptr(expr);
    visit(*expr.array_ptr);
}

void Visitor::visit(const SaveClassIntoLocalAlloc &expr)
{
    if(on_save_class_into_local_alloc)
        on_save_class_into_local_alloc(expr);
    visit(*expr.class_val);
}

void Visitor::visit(const SaveArrayIntoLocalAlloc &expr)
{
    if(on_save_array_into_local_alloc)
        on_save_array_into_local_alloc(expr);
    visit(*expr.array_val);
}

void Visitor::visit(const ArrayAddrToFirstElemAddr &expr)
{
    if(on_array_ptr_to_first_elem_ptr)
        on_array_ptr_to_first_elem_ptr(expr);
    visit(*expr.array_ptr);
}

void Visitor::visit(const Binary &expr)
{
    if(on_binary)
        on_binary(expr);
    visit(*expr.lhs);
    visit(*expr.rhs);
}

void Visitor::visit(const Unary &expr)
{
    if(on_unary)
        on_unary(expr);
    visit(*expr.val);
}

void Visitor::visit(const CallFunc &expr)
{
    if(on_call_func)
        on_call_func(expr);
    for(auto &arg : expr.args)
        visit(*arg);
}

void Visitor::visit(const GlobalVarAddr &expr)
{
    if(on_global_var_addr)
        on_global_var_addr(expr);
}

CUJ_NAMESPACE_END(cuj::core)
