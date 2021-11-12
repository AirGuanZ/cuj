#pragma once

#include <cuj/ir/prog.h>
#include <cuj/util/indented_string_builder.h>
#include <cuj/util/uncopyable.h>

CUJ_NAMESPACE_BEGIN(cuj::gen)

class CGenerator : public Uncopyable
{
public:

    void set_cuda();

    std::string get_string() const;

    void print(const ir::Program &prog);

private:

    struct TypeInfo
    {
        std::string generated_name;
        bool        is_defined  = false;
        bool        is_complete = false;
    };

    struct FunctionInfo
    {
        std::string generated_name;
    };

    std::string generate_type_name(const ir::Type *type);

    void define_types(const ir::Program &prog);

    void define_type(const ir::Type *type);

    void define_builtin_type(ir::BuiltinType type, TypeInfo &info);

    void define_array_type(const ir::ArrayType *type, TypeInfo &info);

    void define_pointer_type(const ir::PointerType *type, TypeInfo &info);

    void define_struct_type(const ir::StructType *type, TypeInfo &info);

    void ensure_complete(const ir::Type *type);

    void declare_function(const ir::Function *func);

    void define_function(const ir::Function *func);

    void generate_statement(const ir::Statement &stat);

    void generate_block(const ir::Block &stat, bool bound = true);

    void generate_value(const ir::BasicValue &val);

    void generate_value(const ir::Value &val);

    void generate_value(const ir::BasicImmediateValue &val);

    void generate_intrinsic(const ir::IntrinsicOp &op);

    const ir::Type *get_type(const ir::BasicValue &val);

    std::string generate_global_consts() const;

    bool is_cuda_ = false;

    IndentedStringBuilder str_;
    std::string           result_;

    int generated_array_type_count_   = 0;
    int generated_pointer_type_count_ = 0;
    std::map<const ir::Type *, TypeInfo> types_;

    const ir::Function *curr_func_ = nullptr;
    std::map<std::vector<unsigned char>, std::string> global_const_data_;
};

CUJ_NAMESPACE_END(cuj::gen)
