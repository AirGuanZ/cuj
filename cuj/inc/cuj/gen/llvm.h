#pragma once

#if CUJ_ENABLE_LLVM

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

    void set_machine(
        const llvm::DataLayout *data_layout,
        const char             *target_triple);

    void generate(const ir::Program &prog);

    llvm::Module *get_module() const;

    std::string get_string() const;

private:

    llvm::Type *find_llvm_type(const ir::Type *type);
    
    llvm::Type *create_llvm_type_record(      ir::BuiltinType    type);
    llvm::Type *create_llvm_type_record(const ir::ArrayType     &type);
    llvm::Type *create_llvm_type_record(const ir::IntrinsicType &type);
    llvm::Type *create_llvm_type_record(const ir::PointerType   &type);
    llvm::Type *create_llvm_type_record(const ir::StructType    &type);

    void construct_struct_type_body(const ir::Type *type);

    void generate_func(const ir::Function &func);

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

    llvm::Value *get_value(const ir::Value &v);

    llvm::Value *get_value(const ir::BasicValue &v);

    llvm::Value *get_value(const ir::BinaryOp &v);

    llvm::Value *get_value(const ir::UnaryOp &v);

    llvm::Value *get_value(const ir::LoadOp &v);

    llvm::Value *get_value(const ir::CallOp &v);

    llvm::Value *get_value(const ir::CastOp &v);

    llvm::Value *get_value(const ir::ArrayElemAddrOp &v);

    llvm::Value *get_value(const ir::IntrinsicOp &v);

    llvm::Value *get_value(const ir::MemberPtrOp &v);

    llvm::Value *get_value(const ir::PointerOffsetOp &v);

    llvm::Value *get_value(const ir::BasicTempValue &v);

    llvm::Value *get_value(const ir::BasicImmediateValue &v);

    llvm::Value *get_value(const ir::AllocAddress &v);

    llvm::Value *convert_to_bool(llvm::Value *from, ir::BuiltinType from_type);

    llvm::Value *convert_from_bool(llvm::Value *from, ir::BuiltinType to_type);

    llvm::Value *convert_arithmetic(
        llvm::Value *from, ir::BuiltinType from_type, ir::BuiltinType to_type);

    ir::BuiltinType get_arithmetic_type(const ir::BasicValue &v);

    Target target_ = Target::Host;

    const llvm::DataLayout *data_layout_ = nullptr;
    const char             *target_triple_ = nullptr;

    Data *data_ = nullptr;
};

CUJ_NAMESPACE_END(cuj::gen)

#endif // #if CUJ_ENABLE_LLVM
