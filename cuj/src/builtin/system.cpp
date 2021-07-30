#include <cuj/builtin/system.h>

CUJ_NAMESPACE_BEGIN(cuj::builtin::system)

namespace
{

    class PrintStatement : public ast::Statement
    {
    public:

        RC<ast::InternalPointerValue<char>> msg;

        void gen_ir(ir::IRBuilder &builder) const override
        {
            auto msg_ptr = msg->gen_ir(builder);
            auto call = ir::IntrinsicCall{ { "system.print", { msg_ptr } } };
            builder.append_statement(newRC<ir::Statement>(call));
        }
    };

    class MallocCall : public ast::InternalPointerValue<void>
    {
    public:

        RC<ast::InternalArithmeticValue<size_t>> size;

        ir::BasicValue gen_ir(ir::IRBuilder &builder) const override
        {
            auto size_val = size->gen_ir(builder);

            auto ret_type = get_current_context()->get_type<Pointer<void>>();
            auto ret = builder.gen_temp_value(ret_type);

            auto op = ir::IntrinsicOp{ "system.malloc", { size_val } };
            builder.append_assign(ret, op);

            return ret;
        }
    };

    class FreeStatement : public ast::Statement
    {
    public:

        RC<ast::InternalPointerValue<void>> ptr;

        void gen_ir(ir::IRBuilder &builder) const override
        {
            auto ptr_val = ptr->gen_ir(builder);
            auto call = ir::IntrinsicCall{ { "system.free", { ptr_val } } };
            builder.append_statement(newRC<ir::Statement>(call));
        }
    };

} // namespace anonymous

void print(const Pointer<char> &msg)
{
    auto stat = newRC<PrintStatement>();
    stat->msg = msg.get_impl();
    get_current_function()->append_statement(std::move(stat));
}

Pointer<void> malloc(const ArithmeticValue<size_t> &bytes)
{
    auto impl = newRC<MallocCall>();
    impl->size = bytes.get_impl();

    Pointer<void> old(std::move(impl));
    Pointer<void> ret = old;
    return ret;
}

void free(const ast::PointerImpl<void> &ptr)
{
    auto stat = newRC<FreeStatement>();
    stat->ptr = ptr.get_impl();
    get_current_function()->append_statement(std::move(stat));
}

CUJ_NAMESPACE_END(cuj::builtin::system)
