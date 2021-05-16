#pragma once

#include <cuj/ir/prog.h>
#include <cuj/util/uncopyable.h>

namespace llvm
{

    class DataLayout;
    class Function;
    class FunctionType;
    class Module;
    class Type;
    class Value;

} // namespace llvm

CUJ_NAMESPACE_BEGIN(cuj::gen)

enum class OptLevel
{
    O0,
    O1,
    O2,
    O3,
    Default = O2,
};

class LLVMIRGenerator : public Uncopyable
{
public:

    enum class Target
    {
        Host,
#if CUJ_ENABLE_CUDA
        PTX,
#endif
    };

    struct Data;

    ~LLVMIRGenerator();

    void set_target(Target target);

    void generate(const ir::Program &prog, llvm::DataLayout *dl = nullptr);

    llvm::Module *get_module() const;

    std::unique_ptr<llvm::Module> get_module_ownership();

    std::string get_string() const;

private:

    static llvm::Type *find_llvm_type(const ir::Type *type);
    
    static llvm::Type *create_llvm_type_record(      ir::BuiltinType    type);
    static llvm::Type *create_llvm_type_record(const ir::ArrayType     &type);
    static llvm::Type *create_llvm_type_record(const ir::IntrinsicType &type);
    static llvm::Type *create_llvm_type_record(const ir::PointerType   &type);
    static llvm::Type *create_llvm_type_record(const ir::StructType    &type);

    void construct_struct_type_body(const ir::Type *type);

    void generate_func_decl(const ir::Function &func);

    llvm::Function *generate_func(const ir::Function &func);

    llvm::FunctionType *generate_func_type(const ir::Function &func);

    void mark_func_type(const ir::Function &func, llvm::Function *llvm_func);

    void generate_func_allocs(const ir::Function &func);

    void copy_func_args(const ir::Function &func);

    void generate(const ir::Statement &s);

    void generate(const ir::Store &store);

    void generate(const ir::Assign &assign);

    void generate(const ir::Break &);

    void generate(const ir::Continue &);

    void generate(const ir::Block &block);

    void generate(const ir::If &if_s);

    void generate(const ir::While &while_s);

    void generate(const ir::Return &return_s);

    void generate(const ir::ReturnClass &return_class);

    void generate(const ir::ReturnArray &return_array);

    void generate(const ir::Call &call);

    void generate(const ir::IntrinsicCall &call);

    llvm::Value *get_value(const ir::Value &v);

    llvm::Value *get_value(const ir::BasicValue &v);

    llvm::Value *get_value(const ir::BinaryOp &v);

    llvm::Value *get_value(const ir::UnaryOp &v);

    llvm::Value *get_value(const ir::LoadOp &v);

    llvm::Value *get_value(const ir::CallOp &v);

    llvm::Value *get_value(const ir::CastBuiltinOp &v);

    llvm::Value *get_value(const ir::CastPointerOp &v);

    llvm::Value *get_value(const ir::ArrayElemAddrOp &v);

    llvm::Value *get_value(const ir::IntrinsicOp &v);

    llvm::Value *get_value(const ir::MemberPtrOp &v);

    llvm::Value *get_value(const ir::PointerOffsetOp &v);

    llvm::Value *get_value(const ir::EmptyPointerOp &v);

    llvm::Value *get_value(const ir::PointerToUIntOp &v);

    llvm::Value *get_value(const ir::PointerDiffOp &v);

    llvm::Value *get_value(const ir::BasicTempValue &v);

    llvm::Value *get_value(const ir::BasicImmediateValue &v);

    llvm::Value *get_value(const ir::AllocAddress &v);

    llvm::Value *get_value(const ir::ConstData &v);

    llvm::Value *convert_to_bool(llvm::Value *from, ir::BuiltinType from_type);

    llvm::Value *convert_from_bool(llvm::Value *from, ir::BuiltinType to_type);

    llvm::Value *convert_arithmetic(
        llvm::Value *from, ir::BuiltinType from_type, ir::BuiltinType to_type);

    ir::BuiltinType get_arithmetic_type(const ir::BasicValue &v);

    llvm::Value *i1_to_bool(llvm::Value *val);
    
    Target target_ = Target::Host;

    llvm::DataLayout *dl_ = nullptr;
    
    Data *data_ = nullptr;
};

CUJ_NAMESPACE_END(cuj::gen)
