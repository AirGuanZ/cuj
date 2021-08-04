#include <cuj/builtin/assert.h>

CUJ_NAMESPACE_BEGIN(cuj::builtin)

namespace
{

    class AssertStatement : public ast::Statement
    {
    public:

        RC<ast::Block>                         cond_block;
        RC<ast::InternalArithmeticValue<bool>> cond;

        RC<ast::InternalPointerValue<char>>        message;
        RC<ast::InternalPointerValue<char>>        file;
        RC<ast::InternalArithmeticValue<uint32_t>> line;
        RC<ast::InternalPointerValue<char>>        function;

        void gen_ir(ir::IRBuilder &builder) const override
        {
            if(!builder.is_assertion_enabled())
                return;
            
            auto cond_block_gen = newRC<ir::Block>();
            builder.push_block(cond_block_gen);
            cond_block->gen_ir(builder);
            builder.pop_block();

            auto cond_val = cond->gen_ir(builder);
            auto if_body = newRC<ir::Block>();
            auto if_stat = ir::If{ cond_val, if_body, { } };

            builder.push_block(if_body);
            
            auto message_val  = message->gen_ir(builder);
            auto file_val     = file->gen_ir(builder);
            auto line_val     = line->gen_ir(builder);
            auto function_val = function->gen_ir(builder);

            auto call = ir::IntrinsicCall{ {
                "system.assertfail",
                { message_val, file_val, line_val, function_val }
            }};

            builder.append_statement(newRC<ir::Statement>(call));
            builder.pop_block();

            builder.append_statement(newRC<ir::Statement>(std::move(if_stat)));
        }
    };

} // namespace anonymous

void assert_impl(
    RC<ast::Block>                         cond_block,
    RC<ast::InternalArithmeticValue<bool>> cond,
    const Pointer<char>                   &message,
    const Pointer<char>                   &file,
    const u32                             &line,
    const Pointer<char>                   &function)
{
    auto stat = newRC<AssertStatement>();
    stat->cond_block = std::move(cond_block);
    stat->cond       = std::move(cond);
    stat->message    = message.get_impl();
    stat->file       = file.get_impl();
    stat->line       = line.get_impl();
    stat->function   = function.get_impl();
    get_current_function()->append_statement(std::move(stat));
}

CUJ_NAMESPACE_END(cuj::builtin)
