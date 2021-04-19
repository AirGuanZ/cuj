#pragma once

#include <map>
#include <typeindex>

#include <cuj/ast/func.h>
#include <cuj/ir/type.h>
#include <cuj/util/scope_guard.h>
#include <cuj/util/uncopyable.h>

CUJ_NAMESPACE_BEGIN(cuj::ast)

class Context : public Uncopyable
{
public:

    template<typename T>
    const ir::Type *get_type();

    Function *get_current_function();

    void begin_function(
        std::string        name,
        ir::Function::Type type = ir::Function::Type::Default);

    void end_function();

    void gen_ir(ir::IRBuilder &builder) const;

private:

    std::map<std::type_index, RC<ir::Type>> types_;

    std::vector<Box<Function>> completed_funcs_;
    Box<Function>              current_func_;
};

inline void push_context(Context *context);

inline void pop_context();

inline Context *get_current_context();

inline Function *get_current_function();

#define CUJ_FUNC (::cuj::ast::get_current_function())

#define CUJ_SCOPED_CONTEXT(CTX_PTR)                                             \
    ::cuj::ast::push_context(CTX_PTR);                                          \
    CUJ_SCOPE_GUARD({ ::cuj::ast::pop_context(); })

CUJ_NAMESPACE_END(cuj::ast)
