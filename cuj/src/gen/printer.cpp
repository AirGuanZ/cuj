#include <cuj/gen/printer.h>

CUJ_NAMESPACE_BEGIN(cuj::gen)

namespace
{

    const char *func_type_to_string(ir::Function::Type type)
    {
        switch(type)
        {
        case ir::Function::Type::Default:
            return "default";
        case ir::Function::Type::Kernel:
            return "kernel";
        }
        return "unknown";
    }

    const char *binary_op_to_string(ir::BinaryOp::Type type)
    {
        switch(type)
        {
        case ir::BinaryOp::Type::Add:          return "+";
        case ir::BinaryOp::Type::Sub:          return "-";
        case ir::BinaryOp::Type::Mul:          return "*";
        case ir::BinaryOp::Type::Div:          return "/";
        case ir::BinaryOp::Type::Mod:          return "%";
        case ir::BinaryOp::Type::And:          return "&&";
        case ir::BinaryOp::Type::Or:           return "||";
        case ir::BinaryOp::Type::BitwiseAnd:   return "&";
        case ir::BinaryOp::Type::BitwiseOr:    return "|";
        case ir::BinaryOp::Type::BitwiseXOr:   return "^";
        case ir::BinaryOp::Type::Equal:        return "==";
        case ir::BinaryOp::Type::NotEqual:     return "!=";
        case ir::BinaryOp::Type::Less:         return "<";
        case ir::BinaryOp::Type::LessEqual:    return "<=";
        case ir::BinaryOp::Type::Greater:      return ">";
        case ir::BinaryOp::Type::GreaterEqual: return ">=";
        }
        return "unknown";
    }

    const char *unary_op_to_string(ir::UnaryOp::Type type)
    {
        switch(type)
        {
        case ir::UnaryOp::Type::Neg: return "-";
        case ir::UnaryOp::Type::Not: return "!";
        }
        return "unknown";
    }

} // namespace anonymous

void IRPrinter::set_indent(std::string indent)
{
    str_.set_single_indent(std::move(indent));
}

void IRPrinter::print(const ir::Program &prog)
{
    bool has_struct = false;

    for(auto &p : prog.types)
    {
        if(auto struct_type = p.second->as_if<ir::StructType>())
        {
            has_struct = true;
            CUJ_INTERNAL_ASSERT(!struct_names_.count(struct_type));
            struct_names_.insert({ struct_type, struct_type->name });
        }
    }

    for(auto &p : prog.types)
    {
        if(auto struct_type = p.second->as_if<ir::StructType>())
            print(struct_type);
    }

    if(has_struct)
        str_.new_line();

    for(size_t i = 0; i < prog.funcs.size(); ++i)
    {
        if(i > 0)
            str_.new_line();
        auto func = prog.funcs[i];
        if(func.is<RC<ir::Function>>())
            print(*func.as<RC<ir::Function>>());
        else
        {
            CUJ_INTERNAL_ASSERT(func.is<RC<ir::ImportedHostFunction>>());
            print(*func.as<RC<ir::ImportedHostFunction>>());
        }
    }
}

std::string IRPrinter::get_string() const
{
    return str_.get_result();
}

void IRPrinter::print(const ir::ImportedHostFunction &func)
{
    str_.append("declare ", func.symbol_name);
    str_.append(" : (");
    for(size_t i = 0; i < func.arg_types.size(); ++i)
    {
        if(i > 0)
            str_.append(", ");
        str_.append(get_typename(func.arg_types[i]));
    }
    str_.append(") -> ", get_typename(func.ret_type));
    str_.new_line();
}

void IRPrinter::print(const ir::Function &func)
{
    alloc_names_.clear();

    // function signature

    str_.append(func_type_to_string(func.type), " ", func.name);
    str_.append(" : (");
    for(size_t i = 0; i < func.args.size(); ++i)
    {
        if(i > 0)
            str_.append(", ");

        const int alloc_index = func.args[i].alloc_index;
        auto type = func.index_to_allocs.find(alloc_index)->second->type;
        
        const std::string arg_name = "arg" + std::to_string(i);
        str_.append(arg_name, " : ", get_typename(type));
        alloc_names_[func.args[i].alloc_index] = std::move(arg_name);
    }
    str_.append(") -> ", get_typename(func.ret_type));
    
    str_.new_line();
    str_.push_indent();

    // local allocations

    size_t next_var_index = 0;
    for(auto &p : func.index_to_allocs)
    {
        if(alloc_names_.count(p.first))
            continue;

        const std::string var_name = "var" + std::to_string(next_var_index++);

        alloc_names_[p.first] = var_name;

        str_.append("alloc ", var_name, " : ", get_typename(p.second->type));
        str_.new_line();
    }

    // body

    print(*func.body);

    str_.pop_indent();
    str_.new_line();
}

void IRPrinter::print(const ir::StructType *type)
{
    const auto name = struct_names_[type];
    str_.append("struct ", name);
    str_.new_line();
    str_.push_indent();
    
    for(size_t i = 0; i < type->mem_types.size(); ++i)
    {
        if(i > 0)
            str_.append(", ");
        str_.append(get_typename(type->mem_types[i]));
    }

    str_.new_line();
    str_.pop_indent();
}

void IRPrinter::print(const ir::Statement &stat)
{
    stat.match([this](const auto &_s) { this->print(_s); });
}

void IRPrinter::print(const ir::Store &store)
{
    str_.append(
        "store ", to_string(store.dst_ptr), " ", to_string(store.src_val));
    str_.new_line();
}

void IRPrinter::print(const ir::Assign &assign)
{
    str_.append(
        to_string(assign.lhs), " <- ", to_string(assign.rhs));
    str_.new_line();
}

void IRPrinter::print(const ir::Break &)
{
    str_.append("break");
    str_.new_line();
}

void IRPrinter::print(const ir::Continue &)
{
    str_.append("continue");
    str_.new_line();
}

void IRPrinter::print(const ir::Block &block)
{
    for(auto &s : block.stats)
        print(*s);
}

void IRPrinter::print(const ir::If &if_s)
{
    str_.append("if ", to_string(if_s.cond), " then");
    str_.new_line();
    str_.push_indent();
    print(*if_s.then_block);
    str_.pop_indent();

    if(if_s.else_block)
    {
        str_.append("else");
        str_.new_line();

        str_.push_indent();
        print(*if_s.else_block);
        str_.pop_indent();
    }
}

void IRPrinter::print(const ir::While &while_s)
{
    str_.append("while ", to_string(while_s.cond), " where");
    str_.new_line();
    str_.push_indent();
    print(*while_s.calculate_cond);
    str_.pop_indent();

    str_.append("do");
    str_.new_line();

    str_.push_indent();
    print(*while_s.body);
    str_.pop_indent();
}

void IRPrinter::print(const ir::Switch &switch_s)
{
    str_.append("switch ", to_string(switch_s.value));
    str_.new_line();

    for(auto &c : switch_s.cases)
    {
        str_.append("case ", to_string(c.cond));
        str_.new_line();
        str_.push_indent();
        print(*c.body);
        if(c.fallthrough)
        {
            str_.append("fallthrough");
            str_.new_line();
        }
        str_.pop_indent();
    }

    if(switch_s.default_body)
    {
        str_.append("default");
        str_.new_line();
        str_.push_indent();
        print(*switch_s.default_body);
        str_.pop_indent();
    }

    str_.append("endswitch");
}

void IRPrinter::print(const ir::Return &return_s)
{
    str_.append("return");
    if(return_s.value)
        str_.append(" ", to_string(*return_s.value));
    str_.new_line();
}

void IRPrinter::print(const ir::ReturnClass &return_class)
{
    str_.append("return class ", to_string(return_class.class_ptr));
}

void IRPrinter::print(const ir::ReturnArray &return_array)
{
    str_.append("return array ", to_string(return_array.array_ptr));
}

void IRPrinter::print(const ir::Call &call)
{
    str_.append("call ", call.op.name);
    for(auto &a : call.op.args)
        str_.append(" ", to_string(a));
    str_.new_line();
}

void IRPrinter::print(const ir::IntrinsicCall &call)
{
    str_.append("intrinsic ", call.op.name);
    for(auto &a : call.op.args)
        str_.append(" ", to_string(a));
    str_.new_line();
}

std::string IRPrinter::get_typename(const ir::Type *type) const
{
    return type->match(
        [this](const auto &t) { return this->get_typename(t); });
}

std::string IRPrinter::get_typename(ir::BuiltinType type) const
{
    switch(type)
    {
    case ir::BuiltinType::Void: return "void";
    case ir::BuiltinType::Char: return "char";
    case ir::BuiltinType::U8:   return "u8";
    case ir::BuiltinType::U16:  return "u16";
    case ir::BuiltinType::U32:  return "u32";
    case ir::BuiltinType::U64:  return "u64";
    case ir::BuiltinType::S8:   return "i8";
    case ir::BuiltinType::S16:  return "i16";
    case ir::BuiltinType::S32:  return "i32";
    case ir::BuiltinType::S64:  return "i64";
    case ir::BuiltinType::F32:  return "f32";
    case ir::BuiltinType::F64:  return "f64";
    case ir::BuiltinType::Bool: return "bool";
    }
    return "unknown";
}

std::string IRPrinter::get_typename(const ir::ArrayType &type) const
{
    return get_typename(type.elem_type) + "[" + std::to_string(type.size) + "]";
}

std::string IRPrinter::get_typename(const ir::IntrinsicType &type) const
{
    return "intrinsic " + type.name;
}

std::string IRPrinter::get_typename(const ir::PointerType &type) const
{
    return "ptr<" + get_typename(type.pointed_type) + ">";
}

std::string IRPrinter::get_typename(const ir::StructType &type) const
{
    auto it = struct_names_.find(&type);
    CUJ_INTERNAL_ASSERT(it != struct_names_.end());
    return it->second;
}

std::string IRPrinter::to_string(const ir::Value &value) const
{
    return match_variant(
        value,
        [this](const ir::BasicValue &v)
    {
        return to_string(v);
    },
        [this](const ir::BinaryOp &v)
    {
        auto l = to_string(v.lhs);
        auto r = to_string(v.rhs);
        return l + " " + binary_op_to_string(v.type) + " " + r;
    },
        [this](const ir::UnaryOp &v)
    {
        return unary_op_to_string(v.type) + (" " + to_string(v.input));
    },
        [this](const ir::LoadOp &v)
    {
        if(v.src_ptr.is<ir::AllocAddress>())
            return "read<" + get_typename(v.type) + "> " + to_string(v.src_ptr);
        return "load<" + get_typename(v.type) + "> " + to_string(v.src_ptr);
    },
        [this](const ir::CallOp &v)
    {
        auto result = "call " + v.name;
        for(auto &a : v.args)
            result += " " + to_string(a);
        result += " -> " + get_typename(v.ret_type);
        return result;
    },
        [this](const ir::CastBuiltinOp &v)
    {
        return "builtin_cast<" + get_typename(v.to_type) + "> "
            + to_string(v.val);
    },
        [this](const ir::CastPointerOp &v)
    {
        return "pointer_cast<" + get_typename(v.to_type) + "> "
            + to_string(v.from_val);
    },
        [this](const ir::ArrayElemAddrOp &v)
    {
        return "array_elem_ptr<" + get_typename(v.arr_type) + "> "
            + to_string(v.arr_alloc);
    },
        [this](const ir::IntrinsicOp &v)
    {
        auto result = "intrinsic " + v.name;
        for(auto &a : v.args)
            result += " " + to_string(a);
        return result;
    },
        [this](const ir::MemberPtrOp &v)
    {
        const auto ptr = to_string(v.ptr);
        const auto ptr_type = get_typename(v.ptr_type);
        const auto mem_idx = std::to_string(v.member_index);
        return "member pointer<" + ptr_type + "> " + ptr + " " + mem_idx;
    },
        [this](const ir::PointerOffsetOp &v)
    {
        const auto ptr = to_string(v.ptr);
        const auto elem_type = get_typename(v.elem_type);
        const auto offset = to_string(v.index);
        return "pointer offset<" + elem_type + "> " + ptr + " " + offset;
    },
        [this](const ir::EmptyPointerOp &)
    {
        return std::string("nullptr");
    },
        [this](const ir::PointerDiffOp &v)
    {
        return "pointer diff " + to_string(v.lhs) + " " + to_string(v.rhs);
    },
        [this](const ir::PointerToUIntOp &v)
    {
        return "pointer to uint " + to_string(v.ptr_val);
    },
        [this](const ir::UintToPointerOp &v)
    {
        return "uint to pointer<" + get_typename(v.ptr_type) + ">"
                                  + " " + to_string(v.uint_val);
    });
}

std::string IRPrinter::to_string(const ir::BasicValue &val) const
{
    return match_variant(
        val,
        [this](const ir::BasicTempValue &v)
    {
        return to_string(v);
    },
        [this](const ir::BasicImmediateValue &v)
    {
        return to_string(v);
    },
        [this](const ir::AllocAddress &v)
    {
        auto it = alloc_names_.find(v.alloc_index);
        CUJ_INTERNAL_ASSERT(it != alloc_names_.end());
        return it->second;
    },
        [this](const ir::ConstData &v)
    {
        return "data<" + get_typename(v.elem_type) + ">";
    });
}

std::string IRPrinter::to_string(const ir::BasicTempValue &val) const
{
    return "t" + std::to_string(val.index);
}

std::string IRPrinter::to_string(const ir::BasicImmediateValue &val) const
{
#define IMM_VAL_TO_STR(TYPE, NAME) \
    [](TYPE v) { return #NAME "(" + std::to_string(v) + ")"; }

    return match_variant(
        val.value,
        IMM_VAL_TO_STR(char,     char),
        IMM_VAL_TO_STR(uint8_t,  u8),
        IMM_VAL_TO_STR(uint16_t, u16),
        IMM_VAL_TO_STR(uint32_t, u32),
        IMM_VAL_TO_STR(uint64_t, u64),
        IMM_VAL_TO_STR(int8_t,   i8),
        IMM_VAL_TO_STR(int16_t,  i16),
        IMM_VAL_TO_STR(int32_t,  i32),
        IMM_VAL_TO_STR(int64_t,  i64),
        IMM_VAL_TO_STR(float,    f32),
        IMM_VAL_TO_STR(double,   f64),
        IMM_VAL_TO_STR(bool,     bool));

#undef IMM_VAL_TO_STR
}

CUJ_NAMESPACE_END(cuj::gen)
