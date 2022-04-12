#include <cassert>

#include <cuj/core/visit.h>
#include <cuj/gen/cpp.h>
#include <cuj/utils/unreachable.h>

CUJ_NAMESPACE_BEGIN(cuj::gen)

namespace
{

    std::string cat_max_align(const std::vector<std::string> &v, size_t s)
    {
        if(s == 1)
            return v[0];
        if(s == 2)
            return "_cuj_constexpr_max(" + v[0] + ", " + v[1] + ")";
        return "_cuj_constexpr_max(" + cat_max_align(v, s - 1) + ", " + v[s - 1] + ")";
    }

} // namespace anonymous

void CPPCodeGenerator::set_target(Target target)
{
    target_ = target;
}

void CPPCodeGenerator::set_assert(bool enabled)
{
    enable_assert_ = enabled;
}

const std::string &CPPCodeGenerator::get_cpp_string() const
{
    return result_;
}

void CPPCodeGenerator::generate(const dsl::Module &mod)
{
    const auto prog = mod._generate_prog();
    prog_ = &prog;

    define_types(prog);

    generate_global_variables(prog);

    generate_global_consts(prog);

    for(auto &f : prog.funcs)
    {
        declare_function(*f, false);
        builder_.appendl(";");
    }

    for(auto &f : prog.funcs)
    {
        if(!f->is_declaration)
            define_function(*f);
    }

    result_.clear();
    if(target_ == Target::PTX)
    {
        result_ =
            "#define CUJ_IS_CUDA 1\n"
            "#define CUJ_FUNCTION_PREFIX __device__\n"
            "#define CUJ_STD\n";
    }
    else
    {
        result_ =
            "#include <cmath>\n"
            "#define CUJ_FUNCTION_PREFIX\n"
            "#define CUJ_STD std::\n";
    }
    result_.append(
#include "cpp_prefix.inl"
    );
    result_.append(builder_.get_str());
}

std::map<const core::Type*, std::type_index> CPPCodeGenerator::build_type_to_index(const core::Prog &prog)
{
    std::map<const core::Type *, std::type_index> result;
    auto handle_type_set = [&](const core::TypeSet &s)
    {
        for(auto &[type, index] : s.type_to_index)
            result.try_emplace(type, index);
    };
    handle_type_set(*prog.global_type_set);
    for(auto &f : prog.funcs)
    {
        if(f->type_set)
            handle_type_set(*f->type_set);
    }
    return result;
}

std::map<std::type_index, std::string> CPPCodeGenerator::build_index_to_name(
    const std::map<const core::Type*, std::type_index> &type_to_index,
    const std::map<std::type_index, const core::Type*> &index_to_type)
{
    std::map<std::type_index, std::string> result;
    for(auto &[index, type] : index_to_type)
    {
        const std::string suffix = type->match(
            [&](core::Builtin t) -> std::string
            {
                switch(t)
                {
                case core::Builtin::S8:   return "S8";
                case core::Builtin::S16:  return "S16";
                case core::Builtin::S32:  return "S32";
                case core::Builtin::S64:  return "S64";
                case core::Builtin::U8:   return "U8";
                case core::Builtin::U16:  return "U16";
                case core::Builtin::U32:  return "U32";
                case core::Builtin::U64:  return "U64";
                case core::Builtin::F32:  return "F32";
                case core::Builtin::F64:  return "F64";
                case core::Builtin::Char: return "Char";
                case core::Builtin::Bool: return "Bool";
                case core::Builtin::Void: return "Void";
                }
                unreachable();
            },
            [&](const core::Array &a)
            {
                return "Array" + std::to_string(result.size());
            },
                [&](const core::Pointer &p)
            {
                return "Pointer" + std::to_string(result.size());
            },
                [&](const core::Struct &s)
            {
                return "Struct" + std::to_string(result.size());
            });

        result[index] = "CujType" + suffix;
    }
    return result;
}

void CPPCodeGenerator::define_types(const core::Prog &prog)
{
    // build type -> name

    const auto type_to_index = build_type_to_index(prog);

    std::map<std::type_index, const core::Type *> index_to_type;
    for(auto &[type, index] : type_to_index)
        index_to_type.try_emplace(index, type);

    auto index_to_name = build_index_to_name(type_to_index, index_to_type);
    for(auto &[type, index] : type_to_index)
        type_names_[type] = index_to_name.at(index);

    // declare

    std::map<std::string, TypeDefineState> define_states;
    for(auto &[_, type] : index_to_type)
        declare_type(define_states, type);

    // define

    for(auto &[_, type] : index_to_type)
        define_type(define_states, type);
}

void CPPCodeGenerator::declare_type(std::map<std::string, TypeDefineState> &states, const core::Type *type)
{
    const std::string &name = type_names_.at(type);
    auto &state = states[name];
    if(state.declared)
        return;

    state.declared = true;
    type->match(
            [&](core::Builtin t)
        {
            std::string raw_name;
            switch(t)
            {
            case core::Builtin::S8:   raw_name = "signed char";        break;
            case core::Builtin::S16:  raw_name = "short";              break;
            case core::Builtin::S32:  raw_name = "int";                break;
            case core::Builtin::S64:  raw_name = "long long";          break;
            case core::Builtin::U8:   raw_name = "unsigned char";      break;
            case core::Builtin::U16:  raw_name = "unsigned short";     break;
            case core::Builtin::U32:  raw_name = "unsigned int";       break;
            case core::Builtin::U64:  raw_name = "unsigned long long"; break;
            case core::Builtin::F32:  raw_name = "float";              break;
            case core::Builtin::F64:  raw_name = "double";             break;
            case core::Builtin::Char: raw_name = "char";               break;
            case core::Builtin::Bool: raw_name = "bool";               break;
            case core::Builtin::Void: raw_name = "void";               break;
            }
            builder_.appendl("using ", name, " = ", raw_name, ";");
            state.complete = true;
        },
            [&](const core::Array &a)
        {
            declare_type(states, a.element);
            builder_.appendl("using ", name, " = ", type_names_.at(a.element), "[", a.size, "];");
            state.complete = false;
        },
            [&](const core::Pointer &p)
        {
            declare_type(states, p.pointed);
            builder_.appendl("using ", name, " = ", type_names_.at(p.pointed), "*;");
            state.complete = true;
        },
            [&](const core::Struct &s)
        {
            builder_.appendl("struct ", name, ";");
            state.complete = false;
        });
}

void CPPCodeGenerator::define_type(std::map<std::string, TypeDefineState> &states, const core::Type *type)
{
    const std::string &name = type_names_.at(type);
    auto &state = states.at(name);
    if(state.complete)
        return;

    if(auto arr = type->as_if<core::Array>())
    {
        define_type(states, arr->element);
        state.complete = true;
    }
    else
    {
        auto &struct_type = type->as<core::Struct>();
        for(auto m : struct_type.members)
            define_type(states, m);

        if(struct_type.custom_alignment)
            builder_.appendl("struct alignas(", struct_type.custom_alignment, ") ", name);
        else
            builder_.appendl("struct ", name);
        builder_.appendl("{");
        builder_.with_indent([&]
            {
                for(size_t i = 0; i < struct_type.members.size(); ++i)
                {
                    auto &mem_typename = type_names_.at(struct_type.members[i]);
                    builder_.appendl(mem_typename, " m", std::to_string(i), ";");
                }
            });
        builder_.appendl("};");

        state.complete = true;
    }
}

void CPPCodeGenerator::generate_global_variables(const core::Prog &prog)
{
    for(auto &pv : prog.global_vars)
    {
        auto &var = *pv;

        std::string memory_prefix;
        if(var.memory_type == core::GlobalVar::MemoryType::Regular)
        {
            if(target_ == Target::PTX)
                memory_prefix = "__device__ ";
        }
        else
        {
            assert(var.memory_type == core::GlobalVar::MemoryType::Constant);
            if(target_ == Target::PTX)
                memory_prefix = "__constant__ ";
            else
                memory_prefix = "const ";
        }

        const std::string &type_name = type_names_.at(var.type);
        builder_.appendl(memory_prefix, type_name, " ", var.symbol_name, " = {};");
    }
}

void CPPCodeGenerator::generate_global_consts(const core::Prog &prog)
{
    std::map<std::vector<unsigned char>, std::vector<std::string>> data_to_align_specifiers;

    core::Visitor visitor;
    visitor.on_global_const_addr = [&](const core::GlobalConstAddr &e)
    {
        auto align_specifier = "alignof(" + type_names_.at(e.pointed_type) + ")";
        data_to_align_specifiers[e.data].push_back(std::move(align_specifier));
    };
    for(auto &f : prog.funcs)
        visitor.visit(*f->root_block);

    std::string prefix;
    if(target_ == Target::PTX)
        prefix = "__device__ const unsigned char ";
    else
        prefix = "const unsigned char ";

    size_t index = 0;
    for(auto &[data, align_specifiers] : data_to_align_specifiers)
    {
        builder_.append(prefix);
        if(target_ == Target::Native)
            builder_.append("alignas(", cat_max_align(align_specifiers, align_specifiers.size()), ") ");
        builder_.append("_cuj_global_", std::to_string(index), "[] = { ");
        for(size_t i = 0; i < data.size(); ++i)
        {
            if(i > 0)
                builder_.append(", ");
            builder_.append(static_cast<int>(data[i]));
        }
        builder_.appendl(" };");

        if(target_ == Target::PTX)
        {
            builder_.append("static_assert(256 >= (");
            builder_.append(cat_max_align(align_specifiers, align_specifiers.size()));
            builder_.appendl("));");
        }

        global_const_indices_[data] = index++;
    }
}

void CPPCodeGenerator::declare_function(const core::Func &func, bool var_name)
{
    std::string prefix;
    if(func.type == core::Func::Regular)
    {
        if(target_ == Target::PTX)
            prefix = "__device__ ";
    }
    else
    {
        assert(func.type == core::Func::Kernel);
        if(target_ == Target::Native)
            throw CujException("non-ptx backend doesn't support kernel function");
        prefix = "__global__ ";
    }
    builder_.append("extern \"C\" ", prefix);
    
    builder_.append(type_names_.at(func.return_type.type));
    if(func.return_type.is_reference)
        builder_.append("*");
    builder_.append(" ");

    builder_.append(func.name, "(");
    for(size_t i = 0; i < func.argument_types.size(); ++i)
    {
        if(i > 0)
            builder_.append(", ");
        builder_.append(type_names_.at(func.argument_types[i].type));
        if(func.argument_types[i].is_reference)
            builder_.append("*");
        if(var_name)
            builder_.append(" _cuj_a", i);
    }
    builder_.append(")");
}

void CPPCodeGenerator::define_function(const core::Func &func)
{
    declare_function(func, true);

    next_label_index_ = 0;
    local_temp_index_ = 0;

    builder_.new_line();
    builder_.appendl("{");
    builder_.with_indent([&]
    {
        generate_local_allocas(func);
        generate_local_temp_allocas(func);
        generate(*func.root_block);
    });
    builder_.appendl("}");
}

void CPPCodeGenerator::generate_local_allocas(const core::Func &func)
{
    for(size_t i = 0; i < func.local_alloc_types.size(); ++i)
        builder_.appendl(type_names_.at(func.local_alloc_types[i]), " _cuj_v", i, ";");
}

void CPPCodeGenerator::generate_local_temp_allocas(const core::Func &func)
{
    core::Visitor visitor;
    size_t index = 0;
    visitor.on_save_class_into_local_alloc = [&](const core::SaveClassIntoLocalAlloc &e)
    {
        auto class_type = e.class_ptr_type->as<core::Pointer>().pointed;
        builder_.appendl(type_names_.at(class_type), " _cuj_t", index++, ";");
    };
    visitor.on_save_array_into_local_alloc = [&](const core::SaveArrayIntoLocalAlloc &e)
    {
        auto array_type = e.array_ptr_type->as<core::Pointer>().pointed;
        builder_.appendl(type_names_.at(array_type), " _cuj_t", index++, ";");
    };
    visitor.visit(*func.root_block);
}

void CPPCodeGenerator::generate(const core::Stat &s)
{
    s.match([this](auto &_s) { generate(_s); });
}

void CPPCodeGenerator::generate(const core::Store &s)
{
    builder_.appendl("*(", generate(s.dst_addr), ") = ", generate(s.val), ";");
}

void CPPCodeGenerator::generate(const core::Copy &s)
{
    builder_.appendl("*(", generate(s.dst_addr), ") = *(", generate(s.src_addr), ");");
}

void CPPCodeGenerator::generate(const core::Block &s)
{
    builder_.appendl("{");
    builder_.with_indent([&]
    {
        for(auto &_s : s.stats)
            generate(*_s);
    });
    builder_.appendl("}");
}

void CPPCodeGenerator::generate(const core::Return &s)
{
    if(auto b = s.return_type->as_if<core::Builtin>(); b && *b == core::Builtin::Void)
        builder_.appendl("return;");
    else
        builder_.appendl("return ", generate(s.val), ";");
}

void CPPCodeGenerator::generate(const core::If &s)
{
    generate(*s.calc_cond);
    builder_.appendl("if(", generate(s.cond), ")");
    builder_.appendl("{");
    builder_.with_indent([&]{ generate(*s.then_body); });
    builder_.appendl("}");
    if(s.else_body)
    {
        builder_.appendl("else");
        builder_.appendl("{");
        builder_.with_indent([&] { generate(*s.else_body); });
        builder_.appendl("}");
    }
}

void CPPCodeGenerator::generate(const core::Loop &s)
{
    const auto new_break_label = "_cuj_break_dest" + std::to_string(next_label_index_++);
    break_dest_label_names_.push(new_break_label);

    builder_.appendl("while(true)");
    builder_.appendl("{");
    builder_.with_indent([&] { generate(*s.body); });
    builder_.appendl("}");

    builder_.appendl(new_break_label, ":");
    break_dest_label_names_.pop();
}

void CPPCodeGenerator::generate(const core::Break &s)
{
    builder_.appendl("goto ", break_dest_label_names_.top(), ";");
}

void CPPCodeGenerator::generate(const core::Continue &s)
{
    builder_.appendl("continue;");
}

void CPPCodeGenerator::generate(const core::Switch &s)
{
    builder_.appendl("switch(", generate(s.value), ")");
    builder_.appendl("{");
    for(auto &c : s.branches)
    {
        builder_.appendl("case ", generate(c.cond), ":");
        builder_.with_indent([&]
        {
            builder_.with_indent([&] { generate(*c.body); });
            if(!c.fallthrough)
                builder_.appendl("break;");
        });
    }
    if(s.default_body)
    {
        builder_.appendl("default:");
        builder_.with_indent([&]
        {
            generate(*s.default_body);
            builder_.appendl("break;");
        });
    }
    builder_.appendl("}");
}

void CPPCodeGenerator::generate(const core::CallFuncStat &s)
{
    builder_.appendl(generate(s.call_expr), ";");
}

void CPPCodeGenerator::generate(const core::MakeScope &s)
{
    const std::string exit_scope_label = "_cuj_exit_scope" + std::to_string(next_label_index_++);
    exit_scope_label_names_.push(exit_scope_label);
    builder_.with_indent([&] { generate(*s.body); });
    exit_scope_label_names_.pop();
    builder_.appendl(exit_scope_label, ":");
}

void CPPCodeGenerator::generate(const core::ExitScope &s)
{
    builder_.appendl("goto ", exit_scope_label_names_.top(), ";");
}

void CPPCodeGenerator::generate(const core::InlineAsm &s)
{
    builder_.appendl("asm volatile(");
    builder_.with_indent([&]
    {
        builder_.appendl("\"", s.asm_string, "\":");
        for(size_t i = 0; i < s.output_constraints.size(); ++i)
        {
            if(i > 0)
                builder_.append(", ");
            builder_.append("\"", s.output_constraints[i], "\"(*(");
            builder_.append(generate(s.output_addresses[i]), "))");
        }
        builder_.appendl(":");
        for(size_t i = 0; i < s.input_constraints.size(); ++i)
        {
            if(i > 0)
                builder_.append(", ");
            builder_.append("\"", s.input_constraints[i], "\"(");
            builder_.append(generate(s.input_values[i]), ")");
        }
        builder_.appendl(":");
        for(size_t i = 0; i < s.clobber_constraints.size(); ++i)
        {
            if(i > 0)
                builder_.append(", ");
            builder_.append(s.clobber_constraints[i]);
        }
    });
    builder_.appendl(");");
}

std::string CPPCodeGenerator::generate(const core::Expr &e) const
{
    return e.match([this](auto &_s) { return generate(_s); });
}

std::string CPPCodeGenerator::generate(const core::FuncArgAddr &e) const
{
    return "(&_cuj_a" + std::to_string(e.arg_index) + ")";
}

std::string CPPCodeGenerator::generate(const core::LocalAllocAddr &e) const
{
    return "(&_cuj_v" + std::to_string(e.alloc_index) + ")";
}

std::string CPPCodeGenerator::generate(const core::Load &e) const
{
    return "(*" + generate(*e.src_addr) + ")";
}

std::string CPPCodeGenerator::generate(const core::Immediate &e) const
{
    auto convert_float = [](auto v)
    {
        std::stringstream ss;
        ss.precision(40);
        ss << std::hexfloat << v;
        return ss.str();
    };
    return e.value.match(
        [&](uint8_t v)  { return "((unsigned char)("      + std::to_string(v) + "))"; },
        [&](uint16_t v) { return "((unsigned short)("     + std::to_string(v) + "))"; },
        [&](uint32_t v) { return "((unsigned int)("       + std::to_string(v) + "u))"; },
        [&](uint64_t v) { return "((unsigned long long)(" + std::to_string(v) + "ull))"; },
        [&](int8_t v)   { return "((signed char)("        + std::to_string(v) + "))"; },
        [&](int16_t v)  { return "(short("                + std::to_string(v) + "))"; },
        [&](int32_t v)  { return "(int("                  + std::to_string(v) + "))"; },
        [&](int64_t v)  { return "((long long)("          + std::to_string(v) + "ll))"; },
        [&](float v)    { return "(float("                + convert_float (v) + "))"; },
        [&](double v)   { return "(double("               + convert_float (v) + "))"; },
        [&](char v)     { return "(char("                 + std::to_string(v) + "))"; },
        [&](bool v)     { return "(bool("                 + std::to_string(v) + "))"; });
}

std::string CPPCodeGenerator::generate(const core::NullPtr &e) const
{
    return "nullptr";
}

std::string CPPCodeGenerator::generate(const core::ArithmeticCast &e) const
{
    return "((" + type_names_.at(e.dst_type) + ")(" + generate(*e.src_val) + "))";
}

std::string CPPCodeGenerator::generate(const core::BitwiseCast &e) const
{
    return "(_cuj_bitcast<" + type_names_.at(e.dst_type) + ">(" + generate(*e.src_val) + "))";
}

std::string CPPCodeGenerator::generate(const core::PointerOffset &e) const
{
    auto l = generate(*e.ptr_val);
    auto r = generate(*e.offset_val);
    return "(" + l + (e.negative ? "-" : "+") + r + ")";
}

std::string CPPCodeGenerator::generate(const core::ClassPointerToMemberPointer &e) const
{
    return "(&((" + generate(*e.class_ptr) + ")->m" + std::to_string(e.member_index) + "))";
}

std::string CPPCodeGenerator::generate(const core::DerefClassPointer &e) const
{
    return "(*(" + generate(*e.class_ptr) + "))";
}

std::string CPPCodeGenerator::generate(const core::DerefArrayPointer &e) const
{
    return "(*(" + generate(*e.array_ptr) + "))";
}

std::string CPPCodeGenerator::generate(const core::SaveClassIntoLocalAlloc &e) const
{
    return "(&(_cuj_t" + std::to_string(local_temp_index_++) + " = " + generate(*e.class_val) + "))";
}

std::string CPPCodeGenerator::generate(const core::SaveArrayIntoLocalAlloc &e) const
{
    return "(&(_cuj_t" + std::to_string(local_temp_index_++) + " = " + generate(*e.array_val) + "))";
}

std::string CPPCodeGenerator::generate(const core::ArrayAddrToFirstElemAddr &e) const
{
    return "(&((*(" + generate(*e.array_ptr) + "))[0]))";
}

std::string CPPCodeGenerator::generate(const core::Binary &e) const
{
    auto lhs = generate(*e.lhs);
    auto rhs = generate(*e.rhs);

    std::string op;
    switch(e.op)
    {
    case core::Binary::Op::Add:          op = "+";  break;
    case core::Binary::Op::Sub:          op = "-";  break;
    case core::Binary::Op::Mul:          op = "*";  break;
    case core::Binary::Op::Div:          op = "/";  break;
    case core::Binary::Op::Mod:          op = "%";  break;
    case core::Binary::Op::Equal:        op = "=="; break;
    case core::Binary::Op::NotEqual:     op = "!="; break;
    case core::Binary::Op::Less:         op = "<";  break;
    case core::Binary::Op::LessEqual:    op = "<="; break;
    case core::Binary::Op::Greater:      op = ">";  break;
    case core::Binary::Op::GreaterEqual: op = ">="; break;
    case core::Binary::Op::LeftShift:    op = "<<"; break;
    case core::Binary::Op::RightShift:   op = ">>"; break;
    case core::Binary::Op::BitwiseAnd:   op = "&";  break;
    case core::Binary::Op::BitwiseOr:    op = "|";  break;
    case core::Binary::Op::BitwiseXOr:   op = "^";  break;
    }

    return "(" + lhs + " " + op + " " + rhs + ")";
}

std::string CPPCodeGenerator::generate(const core::Unary &e) const
{
    std::string op;
    switch(e.op)
    {
    case core::Unary::Op::Neg:        op = "-"; break;
    case core::Unary::Op::Not:        op = "!"; break;
    case core::Unary::Op::BitwiseNot: op = "~"; break;
    }
    return "(" + op + "(" + generate(*e.val) + "))";
}

std::string CPPCodeGenerator::generate(const core::CallFunc &e) const
{
    if(e.intrinsic != core::Intrinsic::None)
        return generate_intrinsic_call(e);

    std::string result;
    if(e.contextless_func)
        result = e.contextless_func->name;
    else
        result = prog_->funcs[e.contexted_func_index]->name;

    result.append("(");
    for(size_t i = 0; i < e.args.size(); ++i)
    {
        if(i > 0)
            result.append(", ");
        result.append(generate(*e.args[i]));
    }
    result.append(")");

    return "(" + result + ")";
}

std::string CPPCodeGenerator::generate(const core::GlobalVarAddr &e) const
{
    return "(&" + e.var->symbol_name + ")";
}

std::string CPPCodeGenerator::generate(const core::GlobalConstAddr &e) const
{
    return "((" + type_names_.at(e.pointed_type) + "*)(_cuj_global_" +
           std::to_string(global_const_indices_.at(e.data)) + "))";
}

std::string CPPCodeGenerator::generate_intrinsic_call(const core::CallFunc &e) const
{
    assert(e.intrinsic != core::Intrinsic::None);

    if(e.intrinsic == core::Intrinsic::assert_fail && !enable_assert_)
        return "void(0)";

    std::string callee;
    switch(e.intrinsic)
    {
    case core::Intrinsic::f32_abs:           callee = "_cuj_f32_abs";            break;
    case core::Intrinsic::f32_mod:           callee = "_cuj_f32_mod";            break;
    case core::Intrinsic::f32_rem:           callee = "_cuj_f32_rem";            break;
    case core::Intrinsic::f32_exp:           callee = "_cuj_f32_exp";            break;
    case core::Intrinsic::f32_exp2:          callee = "_cuj_f32_exp2";           break;
    case core::Intrinsic::f32_exp10:         callee = "_cuj_f32_exp10";          break;
    case core::Intrinsic::f32_log:           callee = "_cuj_f32_log";            break;
    case core::Intrinsic::f32_log2:          callee = "_cuj_f32_log2";           break;
    case core::Intrinsic::f32_log10:         callee = "_cuj_f32_log10";          break;
    case core::Intrinsic::f32_pow:           callee = "_cuj_f32_pow";            break;
    case core::Intrinsic::f32_sqrt:          callee = "_cuj_f32_sqrt";           break;
    case core::Intrinsic::f32_rsqrt:         callee = "_cuj_f32_rsqrt";          break;
    case core::Intrinsic::f32_sin:           callee = "_cuj_f32_sin";            break;
    case core::Intrinsic::f32_cos:           callee = "_cuj_f32_cos";            break;
    case core::Intrinsic::f32_tan:           callee = "_cuj_f32_tan";            break;
    case core::Intrinsic::f32_asin:          callee = "_cuj_f32_asin";           break;
    case core::Intrinsic::f32_acos:          callee = "_cuj_f32_acos";           break;
    case core::Intrinsic::f32_atan:          callee = "_cuj_f32_atan";           break;
    case core::Intrinsic::f32_atan2:         callee = "_cuj_f32_atan2";          break;
    case core::Intrinsic::f32_ceil:          callee = "_cuj_f32_ceil";           break;
    case core::Intrinsic::f32_floor:         callee = "_cuj_f32_floor";          break;
    case core::Intrinsic::f32_trunc:         callee = "_cuj_f32_trunc";          break;
    case core::Intrinsic::f32_round:         callee = "_cuj_f32_round";          break;
    case core::Intrinsic::f32_isfinite:      callee = "_cuj_f32_isfinite";       break;
    case core::Intrinsic::f32_isinf:         callee = "_cuj_f32_isinf";          break;
    case core::Intrinsic::f32_isnan:         callee = "_cuj_f32_isnan";          break;
    case core::Intrinsic::f32_min:           callee = "_cuj_f32_min";            break;
    case core::Intrinsic::f32_max:           callee = "_cuj_f32_max";            break;
    case core::Intrinsic::f32_saturate:      callee = "_cuj_f32_saturate";       break;
    case core::Intrinsic::f64_abs:           callee = "_cuj_f64_abs";            break;
    case core::Intrinsic::f64_mod:           callee = "_cuj_f64_mod";            break;
    case core::Intrinsic::f64_rem:           callee = "_cuj_f64_rem";            break;
    case core::Intrinsic::f64_exp:           callee = "_cuj_f64_exp";            break;
    case core::Intrinsic::f64_exp2:          callee = "_cuj_f64_exp2";           break;
    case core::Intrinsic::f64_exp10:         callee = "_cuj_f64_exp10";          break;
    case core::Intrinsic::f64_log:           callee = "_cuj_f64_log";            break;
    case core::Intrinsic::f64_log2:          callee = "_cuj_f64_log2";           break;
    case core::Intrinsic::f64_log10:         callee = "_cuj_f64_log10";          break;
    case core::Intrinsic::f64_pow:           callee = "_cuj_f64_pow";            break;
    case core::Intrinsic::f64_sqrt:          callee = "_cuj_f64_sqrt";           break;
    case core::Intrinsic::f64_rsqrt:         callee = "_cuj_f64_rsqrt";          break;
    case core::Intrinsic::f64_sin:           callee = "_cuj_f64_sin";            break;
    case core::Intrinsic::f64_cos:           callee = "_cuj_f64_cos";            break;
    case core::Intrinsic::f64_tan:           callee = "_cuj_f64_tan";            break;
    case core::Intrinsic::f64_asin:          callee = "_cuj_f64_asin";           break;
    case core::Intrinsic::f64_acos:          callee = "_cuj_f64_acos";           break;
    case core::Intrinsic::f64_atan:          callee = "_cuj_f64_atan";           break;
    case core::Intrinsic::f64_atan2:         callee = "_cuj_f64_atan2";          break;
    case core::Intrinsic::f64_ceil:          callee = "_cuj_f64_ceil";           break;
    case core::Intrinsic::f64_floor:         callee = "_cuj_f64_floor";          break;
    case core::Intrinsic::f64_trunc:         callee = "_cuj_f64_trunc";          break;
    case core::Intrinsic::f64_round:         callee = "_cuj_f64_round";          break;
    case core::Intrinsic::f64_isfinite:      callee = "_cuj_f64_isfinite";       break;
    case core::Intrinsic::f64_isinf:         callee = "_cuj_f64_isinf";          break;
    case core::Intrinsic::f64_isnan:         callee = "_cuj_f64_isnan";          break;
    case core::Intrinsic::f64_min:           callee = "_cuj_f64_min";            break;
    case core::Intrinsic::f64_max:           callee = "_cuj_f64_max";            break;
    case core::Intrinsic::f64_saturate:      callee = "_cuj_f64_saturate";       break;
    case core::Intrinsic::i32_min:           callee = "_cuj_i32_min";            break;
    case core::Intrinsic::i32_max:           callee = "_cuj_i32_max";            break;
    case core::Intrinsic::u32_min:           callee = "_cuj_u32_min";            break;
    case core::Intrinsic::u32_max:           callee = "_cuj_u32_max";            break;
    case core::Intrinsic::i64_min:           callee = "_cuj_i64_min";            break;
    case core::Intrinsic::i64_max:           callee = "_cuj_i64_max";            break;
    case core::Intrinsic::u64_min:           callee = "_cuj_u64_min";            break;
    case core::Intrinsic::u64_max:           callee = "_cuj_u64_max";            break;
    case core::Intrinsic::thread_idx_x:      callee = "_cuj_thread_idx_x";       break;
    case core::Intrinsic::thread_idx_y:      callee = "_cuj_thread_idx_y";       break;
    case core::Intrinsic::thread_idx_z:      callee = "_cuj_thread_idx_z";       break;
    case core::Intrinsic::block_idx_x:       callee = "_cuj_block_idx_x";        break;
    case core::Intrinsic::block_idx_y:       callee = "_cuj_block_idx_y";        break;
    case core::Intrinsic::block_idx_z:       callee = "_cuj_block_idx_z";        break;
    case core::Intrinsic::block_dim_x:       callee = "_cuj_block_dim_x";        break;
    case core::Intrinsic::block_dim_y:       callee = "_cuj_block_dim_y";        break;
    case core::Intrinsic::block_dim_z:       callee = "_cuj_block_dim_z";        break;
    case core::Intrinsic::store_f32x4:       callee = "_cuj_store_f32x4";        break;
    case core::Intrinsic::store_u32x4:       callee = "_cuj_store_u32x4";        break;
    case core::Intrinsic::store_i32x4:       callee = "_cuj_store_i32x4";        break;
    case core::Intrinsic::store_f32x3:       callee = "_cuj_store_f32x3";        break;
    case core::Intrinsic::store_u32x3:       callee = "_cuj_store_u32x3";        break;
    case core::Intrinsic::store_i32x3:       callee = "_cuj_store_i32x3";        break;
    case core::Intrinsic::store_f32x2:       callee = "_cuj_store_f32x2";        break;
    case core::Intrinsic::store_u32x2:       callee = "_cuj_store_u32x2";        break;
    case core::Intrinsic::store_i32x2:       callee = "_cuj_store_i32x2";        break;
    case core::Intrinsic::load_f32x4:        callee = "_cuj_load_f32x4";         break;
    case core::Intrinsic::load_u32x4:        callee = "_cuj_load_u32x4";         break;
    case core::Intrinsic::load_i32x4:        callee = "_cuj_load_i32x4";         break;
    case core::Intrinsic::load_f32x3:        callee = "_cuj_load_f32x3";         break;
    case core::Intrinsic::load_u32x3:        callee = "_cuj_load_u32x3";         break;
    case core::Intrinsic::load_i32x3:        callee = "_cuj_load_i32x3";         break;
    case core::Intrinsic::load_f32x2:        callee = "_cuj_load_f32x2";         break;
    case core::Intrinsic::load_u32x2:        callee = "_cuj_load_u32x2";         break;
    case core::Intrinsic::load_i32x2:        callee = "_cuj_load_i32x2";         break;
    case core::Intrinsic::atomic_add_i32:    callee = "_cuj_i32_atomic_add";     break;
    case core::Intrinsic::atomic_add_u32:    callee = "_cuj_u32_atomic_add";     break;
    case core::Intrinsic::atomic_add_f32:    callee = "_cuj_f32_atomic_add";     break;
    case core::Intrinsic::cmpxchg_i32:       callee = "_cuj_i32_atomic_cmpxchg"; break;
    case core::Intrinsic::cmpxchg_u32:       callee = "_cuj_u32_atomic_cmpxchg"; break;
    case core::Intrinsic::cmpxchg_u64:       callee = "_cuj_u64_atomic_cmpxchg"; break;
    case core::Intrinsic::print:             callee = "_cuj_print";              break;
    case core::Intrinsic::assert_fail:       callee = "_cuj_assertfail";         break;
    case core::Intrinsic::unreachable:       callee = "_cuj_unreachable";        break;
    case core::Intrinsic::sample_tex_2d_f32: callee = "_cuj_sample_tex2d_f32";   break;
    case core::Intrinsic::sample_tex_2d_i32: callee = "_cuj_sample_tex2d_i32";   break;
    case core::Intrinsic::sample_tex_3d_f32: callee = "_cuj_sample_tex3d_f32";   break;
    case core::Intrinsic::sample_tex_3d_i32: callee = "_cuj_sample_tex3d_i32";   break;
    case core::Intrinsic::memcpy:            callee = "_cuj_memcpy";             break;
    default:
        throw CujException(std::string("unknown intrinsic: ") + intrinsic_name(e.intrinsic));
    }

    std::string args;
    args.append("(");
    for(size_t i = 0; i < e.args.size(); ++i)
    {
        if(i > 0)
            args.append(", ");
        args.append(generate(*e.args[i]));
    }
    args.append(")");

    return "(" + callee + args + ")";
}

CUJ_NAMESPACE_END(cuj::gen)
