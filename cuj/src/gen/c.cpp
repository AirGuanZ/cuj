#include <cuj/gen/c.h>

CUJ_NAMESPACE_BEGIN(cuj::gen)

namespace
{

    const char *binary_op_name(ir::BinaryOp::Type type)
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
        case ir::BinaryOp::Type::LeftShift:    return "<<";
        case ir::BinaryOp::Type::RightShift:   return ">>";
        case ir::BinaryOp::Type::Equal:        return "==";
        case ir::BinaryOp::Type::NotEqual:     return "!=";
        case ir::BinaryOp::Type::Less:         return "<";
        case ir::BinaryOp::Type::LessEqual:    return "<=";
        case ir::BinaryOp::Type::Greater:      return ">";
        case ir::BinaryOp::Type::GreaterEqual: return ">=";
        }
        unreachable();
    }

    const char *unary_op_name(ir::UnaryOp::Type type)
    {
        switch(type)
        {
        case ir::UnaryOp::Type::Neg: return "-";
        case ir::UnaryOp::Type::Not: return "!";
        case ir::UnaryOp::Type::BitwiseNot: return "~";
        }
        unreachable();
    }

    const char *PROG_PREFIX = R"___(#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#ifdef __CUDACC__
#define CUJ_DEVICE_FUNCTION __device__
#define CUJ_KERNEL_FUNCTION __global__
#else
#define CUJ_DEVICE_FUNCTION
#define CUJ_KERNEL_FUNCTION
#endif
CUJ_DEVICE_FUNCTION void CUJAssertFail(const void *message, const void *file, uint32_t line, const void *function)
{
    printf("cuj.assert.fail.%s.%s.%s.%d", (const char *)message, (const char *)file, (const char *)function, line);
}
CUJ_DEVICE_FUNCTION void CUJPrint(const void *message)
{
    printf("%s", (const char *)message);
}
CUJ_DEVICE_FUNCTION float CUJRSqrtF32(float v)
{
    return 1 / sqrtf(v);
}
CUJ_DEVICE_FUNCTION double CUJRSqrtF64(double v)
{
    return 1 / sqrt(v);
}
CUJ_DEVICE_FUNCTION float CUJExp10F32(float v)
{
    return powf(10.0f, v);
}
CUJ_DEVICE_FUNCTION double CUJExp10F64(double v)
{
    return pow(10.0, v);
}
)___";

} // namespace anonymous

void CGenerator::set_cuda()
{
    is_cuda_ = true;
}

std::string CGenerator::get_string() const
{
    IndentedStringBuilder globals;
    for(auto &[data, name] : global_const_data_)
    {
        globals.append("const unsigned char ", name, "[] = { ");

        std::string result;
        for(auto b : data)
        {
            char chs[5];
            sprintf(chs, "0x%02x, ", b);
            result += chs;
        }

        globals.append(result, " };");
        globals.new_line();
    }

    const auto body = str_.get_result();

    return PROG_PREFIX + globals.get_result() + str_.get_result();
}

void CGenerator::print(const ir::Program &prog)
{
    define_types(prog);

    for(auto &f : prog.funcs)
    {
        if(f.is<RC<ir::ImportedHostFunction>>())
        {
            throw CUJException(
                "cuj::gen::CGenerator: ImportedHostFunction is unsupported");
        }
        declare_function(f.as<RC<ir::Function>>().get());
    }

    for(auto &f : prog.funcs)
        define_function(f.as<RC<ir::Function>>().get());
}

std::string CGenerator::generate_type_name(const ir::Type *type)
{
    return type->match(
        [&](ir::BuiltinType t) -> std::string
    {
#define CUJ_GENERATE_BUILTIN_TYPE_NAME(NAME) \
    case ir::BuiltinType::NAME: return "CUJ" #NAME
        switch(t)
        {
            CUJ_GENERATE_BUILTIN_TYPE_NAME(Void);
            CUJ_GENERATE_BUILTIN_TYPE_NAME(Char);
            CUJ_GENERATE_BUILTIN_TYPE_NAME(U8);
            CUJ_GENERATE_BUILTIN_TYPE_NAME(U16);
            CUJ_GENERATE_BUILTIN_TYPE_NAME(U32);
            CUJ_GENERATE_BUILTIN_TYPE_NAME(U64);
            CUJ_GENERATE_BUILTIN_TYPE_NAME(S8);
            CUJ_GENERATE_BUILTIN_TYPE_NAME(S16);
            CUJ_GENERATE_BUILTIN_TYPE_NAME(S32);
            CUJ_GENERATE_BUILTIN_TYPE_NAME(S64);
            CUJ_GENERATE_BUILTIN_TYPE_NAME(F32);
            CUJ_GENERATE_BUILTIN_TYPE_NAME(F64);
            CUJ_GENERATE_BUILTIN_TYPE_NAME(Bool);
        }
#undef CUJ_GENERATE_BUILTIN_TYPE_NAME
        unreachable();
    },
        [&](const ir::ArrayType &)
    {
        return "CUJArray" + std::to_string(generated_array_type_count_++);
    },
        [&](const ir::StructType &t)
    {
        return t.name;
    },
        [&](const ir::PointerType &)
    {
        return "CUJPointer" + std::to_string(generated_pointer_type_count_++);
    });
}

void CGenerator::define_types(const ir::Program &prog)
{
    // register type info

    for(auto &type : prog.types)
    {
        CUJ_INTERNAL_ASSERT(!types_.count(type));
        std::string generated_name = generate_type_name(type);
        types_.insert({ type, { std::move(generated_name) } });
    }

    // declare structure types

    for(auto &type : prog.types)
    {
        if(auto struct_type = type->as_if<ir::StructType>())
        {
            str_.append(
                "typedef struct ", struct_type->name,
                " ", struct_type->name, ";");
            str_.new_line();
        }
    }

    // define types

    for(auto &type : prog.types)
        define_type(type);
}

void CGenerator::define_type(const ir::Type *type)
{
    auto &info = types_.at(type);
    if(info.is_defined)
        return;
    type->match(
        [&](ir::BuiltinType        t) { define_builtin_type(t,  info); },
        [&](const ir::ArrayType   &t) { define_array_type  (&t, info); },
        [&](const ir::PointerType &t) { define_pointer_type(&t, info); },
        [&](const ir::StructType  &t) { define_struct_type (&t, info); });
}

void CGenerator::define_builtin_type(ir::BuiltinType type, TypeInfo &info)
{
    if(info.is_defined)
        return;

#define CUJ_DEFINE_BUILTIN_TYPE(TYPE, NAME)                                     \
    case ir::BuiltinType::NAME:                                                 \
        str_.append("typedef " #TYPE " ", info.generated_name, ";");            \
        str_.new_line();                                                        \
        break

    switch(type)
    {
        CUJ_DEFINE_BUILTIN_TYPE(void,     Void);
        CUJ_DEFINE_BUILTIN_TYPE(char,     Char);
        CUJ_DEFINE_BUILTIN_TYPE(uint8_t,  U8);
        CUJ_DEFINE_BUILTIN_TYPE(uint16_t, U16);
        CUJ_DEFINE_BUILTIN_TYPE(uint32_t, U32);
        CUJ_DEFINE_BUILTIN_TYPE(uint64_t, U64);
        CUJ_DEFINE_BUILTIN_TYPE(int8_t,   S8);
        CUJ_DEFINE_BUILTIN_TYPE(int16_t,  S16);
        CUJ_DEFINE_BUILTIN_TYPE(int32_t,  S32);
        CUJ_DEFINE_BUILTIN_TYPE(int64_t,  S64);
        CUJ_DEFINE_BUILTIN_TYPE(float,    F32);
        CUJ_DEFINE_BUILTIN_TYPE(double,   F64);
        CUJ_DEFINE_BUILTIN_TYPE(int32_t,  Bool);
    }

#undef CUJ_DEFINE_BUILTIN_TYPE

    info.is_defined = true;
    info.is_complete = true;
}

void CGenerator::define_array_type(const ir::ArrayType *type, TypeInfo &info)
{
    if(!type->elem_type->is<ir::StructType>())
        define_type(type->elem_type);

    if(info.is_defined)
        return;

    str_.append("typedef ", types_.at(type->elem_type).generated_name, " ");
    str_.append(info.generated_name, "[", type->size, "];");
    str_.new_line();

    info.is_defined = true;
}

void CGenerator::define_pointer_type(const ir::PointerType *type, TypeInfo &info)
{
    if(!type->pointed_type->is<ir::StructType>())
        define_type(type->pointed_type);

    if(info.is_defined)
        return;

    str_.append("typedef ", types_.at(type->pointed_type).generated_name);
    str_.append(" *", info.generated_name, ";");
    str_.new_line();

    info.is_defined = true;
    info.is_complete = true;
}

void CGenerator::define_struct_type(const ir::StructType *type, TypeInfo &info)
{
    for(auto mem : type->mem_types)
        ensure_complete(mem);

    if(info.is_defined)
        return;

    str_.append("struct ", info.generated_name);
    str_.new_line();
    str_.append("{");
    str_.new_line();
    str_.push_indent();

    for(size_t i = 0; i < type->mem_types.size(); ++i)
    {
        str_.append(types_.at(type->mem_types[i]).generated_name);
        str_.append(" m", i, ";");
        str_.new_line();
    }

    str_.pop_indent();
    str_.append("};");
    str_.new_line();

    info.is_defined  = true;
    info.is_complete = true;
}

void CGenerator::ensure_complete(const ir::Type *type)
{
    auto &info = types_.at(type);
    if(info.is_complete)
        return;
    type->match(
        [&](ir::BuiltinType t)
    {
        CUJ_INTERNAL_ASSERT(!info.is_defined);
        define_builtin_type(t, info);
    },
        [&](const ir::ArrayType &t)
    {
        ensure_complete(t.elem_type);
    },
        [&](const ir::StructType &t)
    {
        CUJ_INTERNAL_ASSERT(!info.is_defined);
        define_struct_type(&t, info);
    },
        [&](const ir::PointerType &t)
    {
        CUJ_INTERNAL_ASSERT(!info.is_defined);
        define_pointer_type(&t, info);
    });
}

void CGenerator::declare_function(const ir::Function *func)
{
    str_.append("#ifdef __cplusplus");
    str_.new_line();
    str_.append("extern \"C\" ");
    str_.new_line();
    str_.append("#endif");
    str_.new_line();

    if(func->type == ir::Function::Type::Kernel)
        str_.append("CUJ_KERNEL_FUNCTION ");
    else
        str_.append("CUJ_DEVICE_FUNCTION ");

    str_.append(types_.at(func->ret_type).generated_name, " ");
    str_.append(func->name, "(");
    if(!func->args.empty())
    {
        auto arg = func->index_to_allocs.at(func->args[0].alloc_index)->type;
        str_.append(types_.at(arg).generated_name);
    }
    for(size_t i = 1; i < func->args.size(); ++i)
    {
        auto arg = func->index_to_allocs.at(func->args[i].alloc_index)->type;
        str_.append(", ", types_.at(arg).generated_name);
    }
    str_.append(");");
    str_.new_line();
}

void CGenerator::define_function(const ir::Function *func)
{
    curr_func_ = func;

    // head

    str_.append("#ifdef __cplusplus");
    str_.new_line();
    str_.append("extern \"C\" ");
    str_.new_line();
    str_.append("#endif");
    str_.new_line();

    if(func->type == ir::Function::Type::Kernel)
        str_.append("CUJ_KERNEL_FUNCTION ");
    else
        str_.append("CUJ_DEVICE_FUNCTION ");

    str_.append(types_.at(func->ret_type).generated_name, " ");
    str_.append(func->name, "(");
    if(!func->args.empty())
    {
        auto arg = func->index_to_allocs.at(func->args[0].alloc_index)->type;
        str_.append(types_.at(arg).generated_name, " arg", 0);
    }
    for(size_t i = 1; i < func->args.size(); ++i)
    {
        auto arg = func->index_to_allocs.at(func->args[i].alloc_index)->type;
        str_.append(", ", types_.at(arg).generated_name, " arg", i);
    }
    str_.append(")");
    str_.new_line();
    str_.append("{");
    str_.new_line();
    str_.push_indent();

    // allocs

    for(auto &[index, alloc] : func->index_to_allocs)
    {
        str_.append(types_.at(alloc->type).generated_name, " var", index, ";");
        str_.new_line();
    }

    // args

    for(size_t i = 0; i < func->args.size(); ++i)
    {
        auto &arg = func->args[i];
        str_.append("var", arg.alloc_index, " = arg", i, ";");
        str_.new_line();
    }

    // statements
    
    generate_block(*func->body, false);

    str_.pop_indent();
    str_.append("}");
    str_.new_line();
}

void CGenerator::generate_statement(const ir::Statement &stat)
{
    stat.match(
        [&](const ir::Store &s)
    {
        str_.append("*(");
        generate_value(s.dst_ptr);
        str_.append(") = (");
        generate_value(s.src_val);
        str_.append(");");
        str_.new_line();
    },
        [&](const ir::Assign &s)
    {
        str_.append(types_.at(s.lhs.type).generated_name, " temp", s.lhs.index);
        str_.append(" = (");
        generate_value(s.rhs);
        str_.append(");");
        str_.new_line();
    },
        [&](const ir::Break &)
    {
        str_.append("break;");
        str_.new_line();
    },
        [&](const ir::Continue &)
    {
        str_.append("continue;");
        str_.new_line();
    },
        [&](const ir::Block &s)
    {
        generate_block(s);
    },
        [&](const ir::If &s)
    {
        str_.append("if(");
        generate_value(s.cond);
        str_.append(")");
        str_.new_line();
        generate_block(*s.then_block);
        if(s.else_block)
        {
            str_.append("else");
            str_.new_line();
            generate_block(*s.else_block);
        }
    },
        [&](const ir::While &s)
    {
        str_.append("while(true)");
        str_.new_line();
        str_.append("{");
        str_.new_line();
        str_.push_indent();
        generate_block(*s.calculate_cond, false);
        str_.append("if(!(");
        generate_value(s.cond);
        str_.append(")) break;");
        str_.new_line();
        generate_block(*s.body, false);
        str_.pop_indent();
        str_.append("}");
        str_.new_line();
    },
        [&](const ir::Switch &s)
    {
        str_.append("switch(");
        generate_value(s.value);
        str_.append(")");
        str_.new_line();
        str_.append("{");
        str_.new_line();

        for(auto &c : s.cases)
        {
            str_.append("case ");
            generate_value(c.cond);
            str_.append(":");
            str_.new_line();
            generate_block(*c.body);
            if(!c.fallthrough)
            {
                str_.append("break;");
                str_.new_line();
            }
        }

        if(s.default_body)
        {
            str_.append("default:");
            str_.new_line();
            generate_block(*s.default_body);
            str_.append("break;");
            str_.new_line();
        }

        str_.append("}");
        str_.new_line();
    },
        [&](const ir::Return &s)
    {
        if(s.value)
        {
            str_.append("return ");
            generate_value(*s.value);
            str_.append(";");
        }
        else
        {
            str_.append("return;");
        }
        str_.new_line();
    },
        [&](const ir::ReturnClass &s)
    {
        str_.append("memcpy(arg0, ");
        generate_value(s.class_ptr);
        str_.append(", sizeof(");

        auto type = get_type(s.class_ptr)->as<ir::PointerType>().pointed_type;
        str_.append(types_.at(type).generated_name);
        str_.append("));");
        str_.new_line();
    },
        [&](const ir::ReturnArray &s)
    {
        str_.append("memcpy(arg0, ");
        generate_value(s.array_ptr);
        str_.append(", sizeof(");

        auto type = get_type(s.array_ptr)->as<ir::PointerType>().pointed_type;
        str_.append(types_.at(type).generated_name);
        str_.append("));");
        str_.new_line();
    },
        [&](const ir::Call &s)
    {
        str_.append(s.op.name, "(");
        if(!s.op.args.empty())
            generate_value(s.op.args[0]);
        for(size_t i = 1; i < s.op.args.size(); ++i)
        {
            str_.append(", ");
            generate_value(s.op.args[i]);
        }
        str_.append(");");
        str_.new_line();
    },
        [&](const ir::IntrinsicCall &s)
    {
        generate_intrinsic(s.op);
    });
}

void CGenerator::generate_block(const ir::Block &stat, bool bound)
{
    if(bound)
    {
        str_.append("{");
        str_.new_line();
        str_.push_indent();
    }
    for(auto &ss : stat.stats)
        generate_statement(*ss);
    if(bound)
    {
        str_.pop_indent();
        str_.append("}");
        str_.new_line();
    }
}

void CGenerator::generate_value(const ir::BasicValue &val)
{
    val.match(
        [&](const ir::BasicTempValue &v)
    {
        str_.append("temp", v.index);
    },
        [&](const ir::BasicImmediateValue &v)
    {
        generate_value(v);
    },
        [&](const ir::AllocAddress &v)
    {
        str_.append("&var", v.alloc_index);
    },
        [&](const ir::ConstData &v)
    {
        auto it = global_const_data_.find(v.bytes);
        if(it == global_const_data_.end())
        {
            auto name =
                "CUJGlobalData" + std::to_string(global_const_data_.size());
            it = global_const_data_.insert({ v.bytes, std::move(name) }).first;
        }
        str_.append("(", types_.at(v.elem_type).generated_name, "*)");
        str_.append(it->second);
    });
}

void CGenerator::generate_value(const ir::BasicImmediateValue &val)
{
#define CUJ_GENERATE_IMMEDIATE_VAL(TYPE) \
    [&](TYPE v) { str_.append("(" #TYPE ")", v); }
    val.value.match(
        CUJ_GENERATE_IMMEDIATE_VAL(uint8_t),
        CUJ_GENERATE_IMMEDIATE_VAL(uint16_t),
        CUJ_GENERATE_IMMEDIATE_VAL(uint32_t),
        CUJ_GENERATE_IMMEDIATE_VAL(uint64_t),
        CUJ_GENERATE_IMMEDIATE_VAL(int8_t),
        CUJ_GENERATE_IMMEDIATE_VAL(int16_t),
        CUJ_GENERATE_IMMEDIATE_VAL(int32_t),
        CUJ_GENERATE_IMMEDIATE_VAL(int64_t),
        CUJ_GENERATE_IMMEDIATE_VAL(float),
        CUJ_GENERATE_IMMEDIATE_VAL(double),
        CUJ_GENERATE_IMMEDIATE_VAL(char),
        [&](bool v)
    {
        str_.append(v ? 1 : 0);
    });
#undef CUJ_GENERATE_IMMEDIATE_VAL
}

void CGenerator::generate_intrinsic(const ir::IntrinsicOp &op)
{
#define CUJ_GEN(intrinsic_name, generated)                                      \
    if(op.name == intrinsic_name)                                               \
    {                                                                           \
        str_.append(generated, "(");                                            \
        if(!op.args.empty())                                                    \
            generate_value(op.args[0]);                                         \
        for(size_t i = 1; i < op.args.size(); ++i)                              \
        {                                                                       \
            str_.append(", ");                                                  \
            generate_value(op.args[i]);                                         \
        }                                                                       \
        str_.append(")");                                                       \
        return;                                                                 \
    }

#define CUJ_GEN_VAR(intrinsic, generated)                                       \
    if(op.name == intrinsic)                                                    \
    {                                                                           \
        str_.append(generated);                                                 \
        return;                                                                 \
    }

    CUJ_GEN("system.assertfail", "CUJAssertFail")
    CUJ_GEN("system.print", "CUJPrint")
    CUJ_GEN("cuda.thread_block_barrier", "__syncthreads()")

    CUJ_GEN_VAR("cuda.thread_index_x", "threadIdx.x")
    CUJ_GEN_VAR("cuda.thread_index_y", "threadIdx.y")
    CUJ_GEN_VAR("cuda.thread_index_z", "threadIdx.z")
    CUJ_GEN_VAR("cuda.block_index_x", "blockIdx.x")
    CUJ_GEN_VAR("cuda.block_index_y", "blockIdx.y")
    CUJ_GEN_VAR("cuda.block_index_z", "blockIdx.z")
    CUJ_GEN_VAR("cuda.block_dim_x", "blockDim.x")
    CUJ_GEN_VAR("cuda.block_dim_y", "blockDim.y")
    CUJ_GEN_VAR("cuda.block_dim_z", "blockDim.z")

#define CUJ_GEN_MATH_CALL(NAME, RESULT)                                         \
    CUJ_GEN("math." #NAME ".f32", #RESULT "f")                                  \
    CUJ_GEN("math." #NAME ".f64", #RESULT)                                      \

    CUJ_GEN_MATH_CALL(abs, fabs)
    CUJ_GEN_MATH_CALL(mod, fmod)
    CUJ_GEN_MATH_CALL(remainder, remainder)
    CUJ_GEN_MATH_CALL(exp, exp)
    CUJ_GEN_MATH_CALL(exp2, exp2)
    CUJ_GEN_MATH_CALL(log, log)
    CUJ_GEN_MATH_CALL(log2, log2)
    CUJ_GEN_MATH_CALL(log10, log10)
    CUJ_GEN_MATH_CALL(pow, pow)
    CUJ_GEN_MATH_CALL(sqrt, sqrt)
    CUJ_GEN_MATH_CALL(sin, sin)
    CUJ_GEN_MATH_CALL(cos, cos)
    CUJ_GEN_MATH_CALL(tan, tan)
    CUJ_GEN_MATH_CALL(asin, asin)
    CUJ_GEN_MATH_CALL(acos, acos)
    CUJ_GEN_MATH_CALL(atan, atan)
    CUJ_GEN_MATH_CALL(atan2, atan2)
    CUJ_GEN_MATH_CALL(floor, floor)
    CUJ_GEN_MATH_CALL(ceil, ceil)
    CUJ_GEN_MATH_CALL(trunc, trunc)
    CUJ_GEN_MATH_CALL(round, round)

    CUJ_GEN("math.rsqrt.f32", "CUJRSqrtF32")
    CUJ_GEN("math.rsqrt.f64", "CUJRSqrtF64")

    CUJ_GEN("math.exp10.f32", "CUJExp10F32")
    CUJ_GEN("math.exp10.f64", "CUJExp10F64")

    CUJ_GEN("math.isfinite.f32", "isfinite")
    CUJ_GEN("math.isfinite.f64", "isfinite")

    CUJ_GEN("math.isinf.f32", "isinf")
    CUJ_GEN("math.isinf.f64", "isinf")

    CUJ_GEN("math.isnan.f32", "isnan")
    CUJ_GEN("math.isnan.f64", "isnan")

    throw CUJException("unknown intrinsic: " + op.name);
}

void CGenerator::generate_value(const ir::Value &val)
{
    val.match(
        [&](const ir::BasicValue &v)
    {
        generate_value(v);
    },
        [&](const ir::BinaryOp &v)
    {
        str_.append("(");
        generate_value(v.lhs);
        str_.append(") ", binary_op_name(v.type), " (");
        generate_value(v.rhs);
        str_.append(")");
    },
        [&](const ir::UnaryOp &v)
    {
        str_.append(unary_op_name(v.type), "(");
        generate_value(v.input);
        str_.append(")");
    },
        [&](const ir::LoadOp &v)
    {
        str_.append("*(");
        generate_value(v.src_ptr);
        str_.append(")");
    },
        [&](const ir::CallOp &v)
    {
        str_.append(v.name, "(");
        if(!v.args.empty())
            generate_value(v.args[0]);
        for(size_t i = 1; i < v.args.size(); ++i)
        {
            str_.append(", ");
            generate_value(v.args[i]);
        }
        str_.append(")");
    },
        [&](const ir::CastBuiltinOp &v)
    {
#define CUJ_CAST_BUILTIN_TYPE(TYPE) \
    case ir::BuiltinType::TYPE: str_.append("(CUJ" #TYPE ")("); break
        switch(v.to_type)
        {
            CUJ_CAST_BUILTIN_TYPE(Void);
            CUJ_CAST_BUILTIN_TYPE(Char);
            CUJ_CAST_BUILTIN_TYPE(U8);
            CUJ_CAST_BUILTIN_TYPE(U16);
            CUJ_CAST_BUILTIN_TYPE(U32);
            CUJ_CAST_BUILTIN_TYPE(U64);
            CUJ_CAST_BUILTIN_TYPE(S8);
            CUJ_CAST_BUILTIN_TYPE(S16);
            CUJ_CAST_BUILTIN_TYPE(S32);
            CUJ_CAST_BUILTIN_TYPE(S64);
            CUJ_CAST_BUILTIN_TYPE(F32);
            CUJ_CAST_BUILTIN_TYPE(F64);
            CUJ_CAST_BUILTIN_TYPE(Bool);
        }
#undef CUJ_CAST_BUILTIN_TYPE
        generate_value(v.val);
        str_.append(")");
    },
        [&](const ir::CastPointerOp &v)
    {
        str_.append("(", types_.at(v.to_type).generated_name, ")(");
        generate_value(v.from_val);
        str_.append(")");
    },
        [&](const ir::ArrayElemAddrOp &v)
    {
        CUJ_INTERNAL_ASSERT(v.arr_alloc.is<ir::AllocAddress>());
        str_.append(
            "&var", v.arr_alloc.as<ir::AllocAddress>().alloc_index, "[0]");
    },
        [&](const ir::IntrinsicOp &v)
    {
        generate_intrinsic(v);
    },
        [&](const ir::MemberPtrOp &v)
    {
        str_.append("&((");
        generate_value(v.ptr);
        str_.append(")->m", v.member_index, ")");
    },
        [&](const ir::PointerOffsetOp &v)
    {
        generate_value(v.ptr);
        str_.append(" + (");
        generate_value(v.index);
        str_.append(")");
    },
        [&](const ir::EmptyPointerOp &v)
    {
        str_.append("NULL");
    },
        [&](const ir::PointerDiffOp &v)
    {
        str_.append("(");
        generate_value(v.lhs);
        str_.append(") - (");
        generate_value(v.rhs);
        str_.append(")");
    },
        [&](const ir::PointerToUIntOp &v)
    {
        str_.append(sizeof(void*) == 4 ? "(CUJU32)" : "(CUJU64)");
        str_.append("(");
        generate_value(v.ptr_val);
        str_.append(")");
    },
        [&](const ir::UintToPointerOp &v)
    {
        str_.append("(", types_.at(v.ptr_type).generated_name, ")(");
        generate_value(v.uint_val);
        str_.append(")");
    });
}

const ir::Type *CGenerator::get_type(const ir::BasicValue &val)
{
    return val.match(
        [](const ir::BasicTempValue &v)
    {
        return v.type;
    },
        [](const ir::BasicImmediateValue &) -> const ir::Type *
    {
        throw CUJException("get_type(BasicImmediateValue) is not supported");
    },
        [&](const ir::AllocAddress &v)
    {
        CUJ_INTERNAL_ASSERT(curr_func_);
        return curr_func_->index_to_allocs.at(v.alloc_index)->type;
    },
        [](const ir::ConstData &) -> const ir::Type *
    {
        throw CUJException("get_type(ConstData) is not supported");
    });
}

CUJ_NAMESPACE_END(cuj::gen)
