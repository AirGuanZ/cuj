#include <cassert>
#include <stack>

#include <cuj/dsl/pointer_temp_var.h>

CUJ_NAMESPACE_BEGIN(cuj::dsl)

namespace
{

    std::stack<PointerTempVarContext*> &get_thread_local_contexts()
    {
        static thread_local std::stack<PointerTempVarContext *> ret;
        return ret;
    }

} // namespace anonymous

void PointerTempVarContext::push_context(PointerTempVarContext *context)
{
    get_thread_local_contexts().push(context);
}

void PointerTempVarContext::pop_context()
{
    assert(!get_thread_local_contexts().empty());
    get_thread_local_contexts().pop();
}

PointerTempVarContext *PointerTempVarContext::get_context()
{
    assert(!get_thread_local_contexts().empty());
    return get_thread_local_contexts().top();
}

void PointerTempVarContext::add_var(std::any var)
{
    vars_.push_back(var);
}

CUJ_NAMESPACE_END(cuj::dsl)
