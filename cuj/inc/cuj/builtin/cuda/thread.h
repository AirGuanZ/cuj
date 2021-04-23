#pragma once

#if CUJ_ENABLE_CUDA

#include <cuj/ast/ast.h>

CUJ_NAMESPACE_BEGIN(cuj::builtin::cuda)

enum class IntrinsicValueType
{
    thread_index_x,
    thread_index_y,
    thread_index_z,
    block_index_x,
    block_index_y,
    block_index_z,
    block_dim_x,
    block_dim_y,
    block_dim_z
};

namespace detail
{

    class InternalIntrinsicIntValue : public ast::InternalArithmeticValue<int>
    {
    public:

        IntrinsicValueType type;

        ir::BasicValue gen_ir(ir::IRBuilder &builder) const override;
    };

} // namespace detail

class Dim3 : public ast::ClassBase<Dim3>
{
public:

    $mem(int, x);
    $mem(int, y);
    $mem(int, z);

    using ClassBase::ClassBase;
};

ast::Value<int> thread_index_x();
ast::Value<int> thread_index_y();
ast::Value<int> thread_index_z();

ast::Value<int> block_index_x();
ast::Value<int> block_index_y();
ast::Value<int> block_index_z();

ast::Value<int> block_dim_x();
ast::Value<int> block_dim_y();
ast::Value<int> block_dim_z();

ast::Value<Dim3> thread_index();
ast::Value<Dim3> block_index();
ast::Value<Dim3> block_dim();

CUJ_NAMESPACE_END(cuj::builtin::cuda)

#endif // #if CUJ_ENABLE_CUDA
