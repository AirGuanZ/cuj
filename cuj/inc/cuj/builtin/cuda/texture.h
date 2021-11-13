#pragma once

#if CUJ_ENABLE_CUDA

#include <cuj/builtin/math/math.h>

CUJ_NAMESPACE_BEGIN(cuj::builtin::cuda)

using TextureObject = u64;

math::Vec1f sample_texture2d_1f(TextureObject tex, f32 u, f32 v);
math::Vec2f sample_texture2d_2f(TextureObject tex, f32 u, f32 v);
math::Vec3f sample_texture2d_3f(TextureObject tex, f32 u, f32 v);
math::Vec4f sample_texture2d_4f(TextureObject tex, f32 u, f32 v);

math::Vec1i sample_texture2d_1i(TextureObject tex, f32 u, f32 v);
math::Vec2i sample_texture2d_2i(TextureObject tex, f32 u, f32 v);
math::Vec3i sample_texture2d_3i(TextureObject tex, f32 u, f32 v);
math::Vec4i sample_texture2d_4i(TextureObject tex, f32 u, f32 v);

CUJ_NAMESPACE_END(cuj::builtin::cuda)

#endif // #if CUJ_ENABLE_CUDA
