#include <cassert>
#include <set>

#include <cuj/core/visit.h>
#include <cuj/utils/printer.h>
#include <cuj/utils/scope_guard.h>
#include <cuj/utils/unreachable.h>

CUJ_NAMESPACE_BEGIN(cuj)

namespace
{

    const char *binary_op_to_str(core::Binary::Op op)
    {
        switch(op)
        {
        case core::Binary::Op::Add:          return "+";
        case core::Binary::Op::Sub:          return "-";
        case core::Binary::Op::Mul:          return "*";
        case core::Binary::Op::Div:          return "/";
        case core::Binary::Op::Mod:          return "%";
        case core::Binary::Op::Equal:        return "==";
        case core::Binary::Op::NotEqual:     return "!=";
        case core::Binary::Op::Greater:      return ">";
        case core::Binary::Op::GreaterEqual: return ">=";
        case core::Binary::Op::Less:         return "<";
        case core::Binary::Op::LessEqual:    return "<=";
        case core::Binary::Op::LeftShift:    return "<<";
        case core::Binary::Op::RightShift:   return ">>";
        case core::Binary::Op::BitwiseAnd:   return "&";
        case core::Binary::Op::BitwiseOr:    return "|";
        case core::Binary::Op::BitwiseXOr:   return "^";
        }
        unreachable();
    }

    const char *unary_op_to_str(core::Unary::Op op)
    {
        switch(op)
        {
        case core::Unary::Op::Neg:        return "-";
        case core::Unary::Op::Not:        return "!";
        case core::Unary::Op::BitwiseNot: return "~";
        }
        unreachable();
    }

    const char *builtin_type_to_str(core::Builtin builtin)
    {
        switch(builtin)
        {
        case core::Builtin::S8:     return "s8";
        case core::Builtin::S16:    return "s16";
        case core::Builtin::S32:    return "s32";
        case core::Builtin::S64:    return "s64";
        case core::Builtin::U8:     return "u8";
        case core::Builtin::U16:    return "u16";
        case core::Builtin::U32:    return "u32";
        case core::Builtin::U64:    return "u64";
        case core::Builtin::F32:    return "f32";
        case core::Builtin::F64:    return "f64";
        case core::Builtin::Char:   return "char";
        case core::Builtin::Bool:   return "bool";
        case core::Builtin::Void:   return "void";
        }
        unreachable();
    }

} // namespace anonymous

void TextBuilder::set_indent_unit(std::string unit)
{
    indent_unit_ = std::move(unit);
}

void TextBuilder::new_line()
{
    ss_ << std::endl;
    newline_ = true;
}

void TextBuilder::push_indent()
{
    indent_cnt_++;
    indent_str_ = {};
    for(int i = 0; i < indent_cnt_; ++i)
        indent_str_ += indent_unit_;
}

void TextBuilder::pop_indent()
{
    assert(indent_cnt_ > 0);
    --indent_cnt_;
    indent_str_ = {};
    for(int i = 0; i < indent_cnt_; ++i)
        indent_str_ += indent_unit_;
}

void TextBuilder::with_indent(std::function<void()> func)
{
    push_indent();
    CUJ_SCOPE_EXIT{ pop_indent(); };
    func();
}

std::string TextBuilder::get_str() const
{
    return ss_.str();
}

void Printer::print(TextBuilder &b, const dsl::FunctionContext &function)
{
    std::set<RC<core::GlobalVar>> global_vars;
    core::Visitor visitor;
    visitor.on_global_var_addr = [&](const core::GlobalVarAddr &addr)
    {
        global_vars.insert(addr.var);
    };
    visitor.visit(*function.get_core_func()->root_block);

    for(auto &var : global_vars)
    {
        b.append(var->symbol_name, " : [");
        switch(var->memory_type)
        {
        case core::GlobalVar::MemoryType::Regular:
            b.append("global");
            break;
        case core::GlobalVar::MemoryType::Constant:
            b.append("constant");
            break;
        }
        b.append("] ");
        print(b, var->type);
        b.new_line();
    }

    auto &args = function.get_core_func()->argument_types;
    b.append("function(");
    for(size_t i = 0; i < function.get_core_func()->argument_types.size(); ++i)
    {
        if(i > 0)
            b.append(", ");
        if(args[i].is_reference)
            b.append("ref ");
        print(b, *args[i].type);
    }
    b.append(") -> ");
    if(function.get_core_func()->return_type.is_reference)
        b.append("ref ");
    print(b, *function.get_core_func()->return_type.type);
    b.new_line();

    b.appendl("{");
    b.with_indent([&]
    {
        auto &local_alloc_types = function.get_core_func()->local_alloc_types;
        for(size_t i = 0; i < local_alloc_types.size(); ++i)
        {
            b.append("var", i, " : ");
            print(b, *local_alloc_types[i]);
            b.new_line();
        }

        for(auto &s : function.get_core_func()->root_block->stats)
            print(b, *s);
    });
    b.appendl("}");
}

void Printer::print(TextBuilder &b, const core::Stat &s)
{
    s.match([&](auto &_s) { print(b, _s); });
}

void Printer::print(TextBuilder &b, const core::Store &s)
{
    if(auto alloc_addr = s.dst_addr.as_if<core::LocalAllocAddr>())
        b.append("var", alloc_addr->alloc_index);
    else
    {
        b.append("*(");
        print(b, s.dst_addr);
        b.append(")");
    }
    b.append(" <- ");
    print(b, s.val);
    b.new_line();
}

void Printer::print(TextBuilder &b, const core::Copy &copy)
{
    b.append("copy from (");
    print(b, copy.src_addr);
    b.append(") to (");
    print(b, copy.dst_addr);
    b.append(")");
}

void Printer::print(TextBuilder &b, const core::Block &block)
{
    b.appendl("{");
    b.with_indent([&]
    {
        for(auto &s : block.stats)
            print(b, *s);
    });
    b.appendl("}");
}

void Printer::print(TextBuilder &b, const core::Return &ret)
{
    if(auto builtin = ret.return_type->as_if<core::Builtin>();
       builtin && *builtin == core::Builtin::Void)
    {
        b.appendl("return");
    }
    else
    {
        b.append("return ");
        print(b, ret.val);
        b.new_line();
    }
}

void Printer::print(TextBuilder &b, const core::If &stat)
{
    print(b, *stat.calc_cond);
    b.append("if(");
    print(b, stat.cond);
    b.append(")");
    b.new_line();
    b.with_indent([&]
    {
        print(b, *stat.then_body);
    });
    if(stat.else_body)
    {
        b.append("else");
        b.with_indent([&]
        {
            print(b, *stat.else_body);
        });
    }
}

void Printer::print(TextBuilder &b, const core::Loop &stat)
{
    b.append("loop");
    b.new_line();
    b.with_indent([&]
    {
        print(b, *stat.body);
    });
}

void Printer::print(TextBuilder &b, const core::Break &stat)
{
    b.appendl("break");
}

void Printer::print(TextBuilder &b, const core::Continue &stat)
{
    b.appendl("continue");
}

void Printer::print(TextBuilder &b, const core::Switch &stat)
{
    b.append("switch(");
    print(b, stat.value);
    b.appendl(")");
    b.appendl("{");
    b.with_indent([&]
    {
        for(auto &branch : stat.branches)
        {
            b.append("case ");
            print(b, branch.cond);
            b.appendl(":");
            print(b, *branch.body);
            if(branch.fallthrough)
                b.appendl("fallthrough");
        }
        if(stat.default_body)
        {
            b.appendl("default:");
            print(b, *stat.default_body);
        }
    });
    b.appendl("}");
}

void Printer::print(TextBuilder &b, const core::CallFuncStat &call)
{
    print(b, call.call_expr);
    b.new_line();
}

void Printer::print(TextBuilder &b, const core::MakeScope &make_scope)
{
    b.appendl("make_scope");
    b.appendl("{");
    b.with_indent([&]
    {
        print(b, *make_scope.body);
    });
    b.appendl("}");
}

void Printer::print(TextBuilder &b, const core::ExitScope &exit_scope)
{
    b.appendl("exit_scope");
}

void Printer::print(TextBuilder &b, const core::InlineAsm &inline_asm)
{
    b.append("asm ");
    if(inline_asm.side_effects)
        b.append("side_effects ");
    b.append(inline_asm.asm_string, " ");
    b.append(inline_asm.output_constraints, " { ");
    for(size_t i = 0; i < inline_asm.output_addresses.size(); ++i)
    {
        if(i > 0)
            b.append(", ");
        print(b, inline_asm.output_addresses[i]);
    }
    b.append(" } ", inline_asm.input_constraints, " { ");
    for(size_t i = 0; i < inline_asm.input_values.size(); ++i)
    {
        if(i > 0)
            b.append(", ");
        print(b, inline_asm.input_values[i]);
    }
    b.append(" }", inline_asm.clobber_constraints);
    b.new_line();
}

void Printer::print(TextBuilder &b, const core::Expr &e)
{
    e.match([&](auto &_e) { print(b, _e); });
}

void Printer::print(TextBuilder &b, const core::FuncArgAddr &addr)
{
    b.append("&arg", addr.arg_index);
}

void Printer::print(TextBuilder &b, const core::LocalAllocAddr &addr)
{
    b.append("&var", addr.alloc_index);
}

void Printer::print(TextBuilder &b, const core::Load &load)
{
    if(auto alloc_addr = load.src_addr->as_if<core::LocalAllocAddr>())
        b.append("var", alloc_addr->alloc_index);
    else
    {
        b.append("load(");
        print(b, *load.src_addr);
        b.append(")");
    }
}

void Printer::print(TextBuilder &b, const core::Immediate &imm)
{
    imm.value.match([&](auto v) { b.append(v); });
}

void Printer::print(TextBuilder &b, const core::NullPtr &null_ptr)
{
    b.append("nullptr");
}

void Printer::print(TextBuilder &b, const core::ArithmeticCast &cast)
{
    b.append("cast<");
    print(b, cast.dst_type->as<core::Builtin>());
    b.append(">(");
    print(b, *cast.src_val);
    b.append(")");
}

void Printer::print(TextBuilder &b, const core::BitwiseCast &cast)
{
    b.append("bitcast<");
    print(b, *cast.dst_type);
    b.append(">(");
    print(b, *cast.src_val);
    b.append(")");
}

void Printer::print(TextBuilder &b, const core::PointerOffset &ptr_offset)
{
    print(b, *ptr_offset.ptr_val);
    b.append(" ", ptr_offset.negative ? "-" : "+", " ");
    print(b, *ptr_offset.offset_val);
}

void Printer::print(TextBuilder &b, const core::ClassPointerToMemberPointer &mem)
{
    b.append("member_ptr(");
    print(b, *mem.class_ptr);
    b.append(", ", mem.member_index, ")");
}

void Printer::print(TextBuilder &b, const core::DerefClassPointer &deref)
{
    b.append("deref class(");
    print(b, *deref.class_ptr);
    b.append(")");
}

void Printer::print(TextBuilder &b, const core::DerefArrayPointer &deref)
{
    b.append("deref array(");
    print(b, *deref.array_ptr);
    b.append(")");
}

void Printer::print(TextBuilder &b, const core::SaveClassIntoLocalAlloc &deref)
{
    b.append("alloc_for(");
    print(b, *deref.class_val);
    b.append(")");
}

void Printer::print(TextBuilder &b, const core::SaveArrayIntoLocalAlloc &deref)
{
    b.append("alloc_for(");
    print(b, *deref.array_val);
    b.append(")");
}

void Printer::print(TextBuilder &b, const core::ArrayAddrToFirstElemAddr &to)
{
    b.append("get_arr_head(");
    print(b, *to.array_ptr);
    b.append(")");
}

void Printer::print(TextBuilder &b, const core::Binary &binary)
{
    print(b, *binary.lhs);
    b.append(" ", binary_op_to_str(binary.op), " ");
    print(b, *binary.rhs);
}

void Printer::print(TextBuilder &b, const core::Unary &unary)
{
    b.append(unary_op_to_str(unary.op));
    print(b, *unary.val);
}

void Printer::print(TextBuilder &b, const core::CallFunc &call)
{
    if(call.contextless_func)
        b.append("nameless_function(");
    else if(call.intrinsic != core::Intrinsic::None)
        b.append("intrinsic ", intrinsic_name(call.intrinsic), "(");
    else
        b.append("func", call.contexted_func_index, "(");

    for(size_t i = 0; i < call.args.size(); ++i)
    {
        if(i > 0)
            b.append(", ");
        print(b, *call.args[i]);
    }
    b.append(")");
}

void Printer::print(TextBuilder &b, const core::GlobalVarAddr &global_var_addr)
{
    b.append("&(global ", global_var_addr.var->symbol_name, ")");
}

void Printer::print(TextBuilder &b, const core::Type &type)
{
    type.match([&](auto &t) { print(b, t); });
}

void Printer::print(TextBuilder &b, core::Builtin builtin)
{
    b.append(builtin_type_to_str(builtin));
}

void Printer::print(TextBuilder &b, const core::Struct &s)
{
    b.append("struct{ ");
    if(!s.members.empty())
        print(b, *s.members[0]);
    for(size_t i = 1; i < s.members.size(); ++i)
    {
        b.append(", ");
        print(b, *s.members[i]);
    }
    b.append( " }");
}

void Printer::print(TextBuilder &b, const core::Array &a)
{
    print(b, *a.element);
    b.append("[", a.size, "]");
}

void Printer::print(TextBuilder &b, const core::Pointer &p)
{
    b.append("ptr<");
    print(b, *p.pointed);
    b.append(">");
}

CUJ_NAMESPACE_END(cuj)
