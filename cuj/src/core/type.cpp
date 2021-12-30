#include <cuj/core/type.h>

CUJ_NAMESPACE_BEGIN(cuj::core)

std::strong_ordering Struct::operator<=>(const Struct &rhs) const
{
    const size_t min_size = (std::min)(members.size(), rhs.members.size());
    for(size_t i = 0; i < min_size; ++i)
    {
        const auto mem_comp = *members[i] <=> *rhs.members[i];
        if(mem_comp != std::strong_ordering::equal)
            return mem_comp;
    }
    return members.size() <=> rhs.members.size();
}

bool Struct::operator==(const Struct &rhs) const
{
    if(members.size() != rhs.members.size())
        return false;
    for(size_t i = 0; i < members.size(); ++i)
    {
        if(*members[i] != *rhs.members[i])
            return false;
    }
    return true;
}

std::strong_ordering Array::operator<=>(const Array &rhs) const
{
    const std::strong_ordering elem_comp = *element <=> *rhs.element;
    if(elem_comp != std::strong_ordering::equal)
        return elem_comp;
    return size <=> rhs.size;
}

bool Array::operator==(const Array &rhs) const
{
    return *element == *rhs.element && size == rhs.size;
}

std::strong_ordering Pointer::operator<=>(const Pointer &rhs) const
{
    return *pointed <=> *rhs.pointed;
}

bool Pointer::operator==(const Pointer &rhs) const
{
    return *pointed == *rhs.pointed;
}

CUJ_NAMESPACE_END(cuj::core)
