#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable: 4141)
#pragma warning(disable: 4244)
#pragma warning(disable: 4624)
#pragma warning(disable: 4626)
#pragma warning(disable: 4996)
#endif

#include "type_manager.h"

CUJ_NAMESPACE_BEGIN(cuj::gen::llvm_helper)

void TypeManager::initialize(
    llvm::LLVMContext *llvm_context,
    const llvm::DataLayout *data_layout,
    std::map<const core::Type *, std::type_index> types)
{
    llvm_context_ = llvm_context;
    data_layout_ = data_layout;
    type_indices_ = std::move(types);

    std::map<std::type_index, const core::Type *> index_to_types;
    for(auto &[type, index] : type_indices_)
        index_to_types[index] = type;

    for(auto &[_, type] : index_to_types)
        create_record(type);

    for(auto &[_, type] : index_to_types)
        fill_layout(type);
}

llvm::Type *TypeManager::get_llvm_type(const core::Type *type) const
{
    const auto index = type_indices_.at(type);
    return records_.at(index).type;
}

size_t TypeManager::get_custom_alignment(const core::Type *type) const
{
    const auto index = type_indices_.at(type);
    return records_.at(index).alignment.value();
}

int TypeManager::get_struct_member_index(
    const core::Type *struct_type, int raw_member_index) const
{
    auto &record = find_record(struct_type);
    return record.member_indices[raw_member_index];
}

llvm::Type *TypeManager::create_record(const core::Type *type)
{
    const auto index = type_indices_.at(type);
    if(auto it = records_.find(index); it != records_.end())
        return it->second.type;

    auto llvm_type = type->match(
        [&](core::Builtin t) -> llvm::Type *
    {
        return builtin_to_llvm_type(llvm_context_, t);
    },
        [&](const core::Struct &t)->llvm::Type *
    {
        return llvm::StructType::create(*llvm_context_);
    },
        [&](const core::Array &t) -> llvm::Type *
    {
        auto elem = create_record(t.element);
        return llvm::ArrayType::get(elem, t.size);
    },
        [&](const core::Pointer &t) -> llvm::Type *
    {
        auto pt = t.pointed->as_if<core::Builtin>();
        if(pt && *pt == core::Builtin::Void)
        {
            return llvm::PointerType::get(
                llvm::Type::getInt8Ty(*llvm_context_), 0);
        }
        auto pointed = create_record(t.pointed);
        return llvm::PointerType::get(pointed, 0);
    });

    assert(!records_.contains(index));
    records_.insert({ index, Record{ .type = llvm_type } });
    return llvm_type;
}

void TypeManager::fill_layout(const core::Type *type)
{
    auto index = type_indices_.at(type);
    auto &record = records_.at(index);
    if(record.alignment)
        return;
    type->match(
        [&](core::Builtin t)
    {
        if(data_layout_ && t != core::Builtin::Void)
        {
            record.alignment = data_layout_->getABITypeAlign(record.type).value();
            record.size = data_layout_->getTypeAllocSize(record.type).getFixedSize();
        }
        else
        {
            using enum core::Builtin;
            switch(t)
            {
            case S8:
            case U8:
            case Char:
            case Bool:
            case Void:
                record.alignment = 1;
                record.size = 1;
                break;
            case S16:
            case U16:
                record.alignment = 2;
                record.size = 2;
                break;
            case S32:
            case U32:
            case F32:
                record.alignment = 4;
                record.size = 4;
                break;
            case S64:
            case U64:
            case F64:
                record.alignment = 8;
                record.size = 8;
                break;
            }
            assert(record.alignment);
        }
    },
        [&](const core::Struct &t)
    {
        struct LocalRecord
        {
            Record *record;
            size_t alignment;
            size_t size;
        };

        std::vector<LocalRecord> member_layouts(t.members.size());
        for(size_t i = 0; i < t.members.size(); ++i)
        {
            fill_layout(t.members[i]);
            auto &member_record = records_.at(type_indices_.at(t.members[i]));
            member_layouts[i].record = &member_record;
            member_layouts[i].alignment = *member_record.alignment;
            member_layouts[i].size = *member_record.size;
        }

        size_t alignment = t.custom_alignment;
        for(auto &ml : member_layouts)
            alignment = (std::max)(alignment, ml.alignment);

        auto llvm_u8 = llvm::IntegerType::getInt8Ty(*llvm_context_);

        size_t current_offset = 0;
        std::vector<llvm::Type *> member_llvm_types;
        for(size_t i = 0; i < t.members.size(); ++i)
        {
            auto &ml = member_layouts[i];
            while(current_offset % ml.alignment)
            {
                member_llvm_types.push_back(llvm_u8);
                ++current_offset;
            }
            record.member_indices.push_back(static_cast<int>(member_llvm_types.size()));
            member_llvm_types.push_back(ml.record->type);
            current_offset += ml.size;
        }

        size_t size = (current_offset + alignment - 1) / alignment * alignment;
        const size_t post_padding_bytes = size - current_offset;
        for(size_t i = 0; i < post_padding_bytes; ++i)
            member_llvm_types.push_back(llvm_u8);

        auto record_struct_type = llvm::dyn_cast<llvm::StructType>(record.type);
        assert(record_struct_type->isOpaque());
        record_struct_type->setBody(member_llvm_types);

        if(data_layout_)
        {
            size = (std::max)(size, data_layout_->getTypeAllocSize(record.type).getFixedSize());
            alignment = (std::max)(alignment, data_layout_->getABITypeAlign(record.type).value());
        }

        record.alignment = alignment;
        record.size = size;
    },
        [&](const core::Array &t)
    {
        fill_layout(t.element);
        auto &elem_record = find_record(t.element);
        record.alignment = elem_record.alignment;
        record.size = *elem_record.size * t.size;

        if(data_layout_)
        {
            record.alignment = (std::max)(
                *record.alignment, data_layout_->getABITypeAlign(record.type).value());
            record.size = (std::max)(
                *record.size, data_layout_->getTypeAllocSize(record.type).getFixedSize());
        }
    },
        [&](const core::Pointer &)
    {
        if(data_layout_)
        {
            record.alignment = data_layout_->getABITypeAlign(record.type).value();
            record.size = data_layout_->getTypeAllocSize(record.type).getFixedSize();
        }
        else
        {
            record.alignment = alignof(void *);
            record.size = sizeof(void *);
        }
    });
}

const TypeManager::Record &TypeManager::find_record(const core::Type *type) const
{
    const auto index = type_indices_.at(type);
    return records_.at(index);
}

CUJ_NAMESPACE_END(cuj::gen::llvm_helper)

#ifdef _MSC_VER
#pragma warning(pop)
#endif
