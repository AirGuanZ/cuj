#pragma once

#include <optional>
#include <set>

#include <llvm/IR/IRBuilder.h>

#include "helper.h"

CUJ_NAMESPACE_BEGIN(cuj::gen::llvm_helper)

class TypeManager
{
public:

    void initialize(
        llvm::LLVMContext                           *llvm_context,
        const llvm::DataLayout                      *data_layout,
        std::map<const core::Type*, std::type_index> types);

    llvm::Type *get_llvm_type(const core::Type *type) const;

    size_t get_custom_alignment(const core::Type *type) const;

    int get_struct_member_index(
        const core::Type *struct_type, int raw_member_index) const;

private:

    struct Record
    {
        llvm::Type               *type;
        std::optional<size_t>     alignment;
        std::optional<size_t>     size;
        std::vector<int>          member_indices;
    };

    llvm::Type *create_record(const core::Type *type);

    void fill_layout(const core::Type *type);

    const Record &find_record(const core::Type *type) const;

    llvm::LLVMContext                            *llvm_context_ = nullptr;
    const llvm::DataLayout                       *data_layout_ = nullptr;
    std::map<std::type_index, Record>             records_;
    std::map<const core::Type *, std::type_index> type_indices_;
};

CUJ_NAMESPACE_END(cuj::gen::llvm_helper)
