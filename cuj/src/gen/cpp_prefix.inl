R"___(
CUJ_FUNCTION_PREFIX inline constexpr size_t _cuj_constexpr_max(size_t a, size_t b)
{
    return a > b ? a : b;
}

CUJ_FUNCTION_PREFIX inline float _cuj_f32_abs(float f)
{
    return CUJ_STD fabsf(f);
}

CUJ_FUNCTION_PREFIX inline float _cuj_f32_mod(float a, float b)
{
    return CUJ_STD fmodf(a, b);
}

CUJ_FUNCTION_PREFIX inline float _cuj_f32_rem(float a, float b)
{
    return CUJ_STD remainderf(a, b);
}

CUJ_FUNCTION_PREFIX inline float _cuj_f32_exp(float a)
{
    return CUJ_STD expf(a);
}

CUJ_FUNCTION_PREFIX inline float _cuj_f32_exp2(float a)
{
    return CUJ_STD exp2f(a);
}

CUJ_FUNCTION_PREFIX inline float _cuj_f32_exp10(float a)
{
#ifdef CUJ_IS_CUDA
    return exp10f(a);
#else
    return CUJ_STD powf(10, a);
#endif
}

CUJ_FUNCTION_PREFIX inline float _cuj_f32_log(float a)
{
    return CUJ_STD logf(a);
}

CUJ_FUNCTION_PREFIX inline float _cuj_f32_log2(float a)
{
    return CUJ_STD log2f(a);
}

CUJ_FUNCTION_PREFIX inline float _cuj_f32_log10(float a)
{
    return CUJ_STD log10f(a);
}

CUJ_FUNCTION_PREFIX inline float _cuj_f32_pow(float a, float b)
{
    return CUJ_STD powf(a, b);
}

CUJ_FUNCTION_PREFIX inline float _cuj_f32_sqrt(float a)
{
    return CUJ_STD sqrtf(a);
}

CUJ_FUNCTION_PREFIX inline float _cuj_f32_rsqrt(float a)
{
#ifdef CUJ_IS_CUDA
    return rsqrtf(a);
#else
    return 1.0f / CUJ_STD sqrtf(a);
#endif
}

CUJ_FUNCTION_PREFIX inline float _cuj_f32_sin(float a)
{
    return CUJ_STD sinf(a);
}

CUJ_FUNCTION_PREFIX inline float _cuj_f32_cos(float a)
{
    return CUJ_STD cosf(a);
}

CUJ_FUNCTION_PREFIX inline float _cuj_f32_tan(float a)
{
    return CUJ_STD tanf(a);
}

CUJ_FUNCTION_PREFIX inline float _cuj_f32_asin(float a)
{
    return CUJ_STD asinf(a);
}

CUJ_FUNCTION_PREFIX inline float _cuj_f32_acos(float a)
{
    return CUJ_STD acosf(a);
}

CUJ_FUNCTION_PREFIX inline float _cuj_f32_atan(float a)
{
    return CUJ_STD atanf(a);
}

CUJ_FUNCTION_PREFIX inline float _cuj_f32_atan2(float a, float b)
{
    return CUJ_STD atan2f(a, b);
}

CUJ_FUNCTION_PREFIX inline float _cuj_f32_ceil(float a)
{
    return CUJ_STD ceilf(a);
}

CUJ_FUNCTION_PREFIX inline float _cuj_f32_floor(float a)
{
    return CUJ_STD floorf(a);
}

CUJ_FUNCTION_PREFIX inline float _cuj_f32_trunc(float a)
{
    return CUJ_STD truncf(a);
}

CUJ_FUNCTION_PREFIX inline float _cuj_f32_round(float a)
{
    return CUJ_STD roundf(a);
}

CUJ_FUNCTION_PREFIX inline int _cuj_f32_isfinite(float a)
{
    return int(CUJ_STD isfinite(a));
}

CUJ_FUNCTION_PREFIX inline int _cuj_f32_isinf(float a)
{
    return int(CUJ_STD isinf(a));
}

CUJ_FUNCTION_PREFIX inline int _cuj_f32_isnan(float a)
{
    return int(CUJ_STD isnan(a));
}

CUJ_FUNCTION_PREFIX inline float _cuj_f32_min(float a, float b)
{
    return CUJ_STD fminf(a, b);
}

CUJ_FUNCTION_PREFIX inline float _cuj_f32_max(float a, float b)
{
    return CUJ_STD fmaxf(a, b);
}

CUJ_FUNCTION_PREFIX inline float _cuj_f32_saturate(float a)
{
    return _cuj_f32_max(0, _cuj_f32_min(a, 1));
}

CUJ_FUNCTION_PREFIX inline double _cuj_f64_abs(double f)
{
    return CUJ_STD fabs(f);
}

CUJ_FUNCTION_PREFIX inline double _cuj_f64_mod(double a, double b)
{
    return CUJ_STD fmod(a, b);
}

CUJ_FUNCTION_PREFIX inline double _cuj_f64_rem(double a, double b)
{
    return CUJ_STD remainder(a, b);
}

CUJ_FUNCTION_PREFIX inline double _cuj_f64_exp(double a)
{
    return CUJ_STD exp(a);
}

CUJ_FUNCTION_PREFIX inline double _cuj_f64_exp2(double a)
{
    return CUJ_STD exp2(a);
}

CUJ_FUNCTION_PREFIX inline double _cuj_f64_exp10(double a)
{
#ifdef CUJ_IS_CUDA
    return exp10(a);
#else
    return CUJ_STD pow(10, a);
#endif
}

CUJ_FUNCTION_PREFIX inline double _cuj_f64_log(double a)
{
    return CUJ_STD log(a);
}

CUJ_FUNCTION_PREFIX inline double _cuj_f64_log2(double a)
{
    return CUJ_STD log2(a);
}

CUJ_FUNCTION_PREFIX inline double _cuj_f64_log10(double a)
{
    return CUJ_STD log10(a);
}

CUJ_FUNCTION_PREFIX inline double _cuj_f64_pow(double a, double b)
{
    return CUJ_STD pow(a, b);
}

CUJ_FUNCTION_PREFIX inline double _cuj_f64_sqrt(double a)
{
    return CUJ_STD sqrt(a);
}

CUJ_FUNCTION_PREFIX inline double _cuj_f64_rsqrt(double a)
{
#ifdef CUJ_IS_CUDA
    return rsqrt(a);
#else
    return 1.0f / CUJ_STD sqrt(a);
#endif
}

CUJ_FUNCTION_PREFIX inline double _cuj_f64_sin(double a)
{
    return CUJ_STD sin(a);
}

CUJ_FUNCTION_PREFIX inline double _cuj_f64_cos(double a)
{
    return CUJ_STD cos(a);
}

CUJ_FUNCTION_PREFIX inline double _cuj_f64_tan(double a)
{
    return CUJ_STD tan(a);
}

CUJ_FUNCTION_PREFIX inline double _cuj_f64_asin(double a)
{
    return CUJ_STD asin(a);
}

CUJ_FUNCTION_PREFIX inline double _cuj_f64_acos(double a)
{
    return CUJ_STD acos(a);
}

CUJ_FUNCTION_PREFIX inline double _cuj_f64_atan(double a)
{
    return CUJ_STD atan(a);
}

CUJ_FUNCTION_PREFIX inline double _cuj_f64_atan2(double a, double b)
{
    return CUJ_STD atan2(a, b);
}

CUJ_FUNCTION_PREFIX inline double _cuj_f64_ceil(double a)
{
    return CUJ_STD ceil(a);
}

CUJ_FUNCTION_PREFIX inline double _cuj_f64_floor(double a)
{
    return CUJ_STD floor(a);
}

CUJ_FUNCTION_PREFIX inline double _cuj_f64_trunc(double a)
{
    return CUJ_STD trunc(a);
}

CUJ_FUNCTION_PREFIX inline double _cuj_f64_round(double a)
{
    return CUJ_STD round(a);
}

CUJ_FUNCTION_PREFIX inline int _cuj_f64_isfinite(double a)
{
    return int(CUJ_STD isfinite(a));
}

CUJ_FUNCTION_PREFIX inline int _cuj_f64_isinf(double a)
{
    return int(CUJ_STD isinf(a));
}

CUJ_FUNCTION_PREFIX inline int _cuj_f64_isnan(double a)
{
    return int(CUJ_STD isnan(a));
}

CUJ_FUNCTION_PREFIX inline double _cuj_f64_min(double a, double b)
{
    return CUJ_STD fmin(a, b);
}

CUJ_FUNCTION_PREFIX inline double _cuj_f64_max(double a, double b)
{
    return CUJ_STD fmax(a, b);
}

CUJ_FUNCTION_PREFIX inline double _cuj_f64_saturate(double a)
{
    return _cuj_f64_max(0, _cuj_f64_min(a, 1));
}

CUJ_FUNCTION_PREFIX inline int _cuj_i32_min(int a, int b)
{
    return a < b ? a : b;
}

CUJ_FUNCTION_PREFIX inline int _cuj_i32_max(int a, int b)
{
    return a > b ? a : b;
}

CUJ_FUNCTION_PREFIX inline int _cuj_u32_min(unsigned a, unsigned b)
{
    return a < b ? a : b;
}

CUJ_FUNCTION_PREFIX inline int _cuj_u32_max(unsigned a, unsigned b)
{
    return a > b ? a : b;
}

CUJ_FUNCTION_PREFIX inline int _cuj_i64_min(long long a, long long b)
{
    return a < b ? a : b;
}

CUJ_FUNCTION_PREFIX inline int _cuj_i64_max(long long a, long long b)
{
    return a > b ? a : b;
}

CUJ_FUNCTION_PREFIX inline int _cuj_u64_min(unsigned long long a, unsigned long long b)
{
    return a < b ? a : b;
}

CUJ_FUNCTION_PREFIX inline int _cuj_u64_max(unsigned long long a, unsigned long long b)
{
    return a > b ? a : b;
}

#ifdef CUJ_IS_CUDA

CUJ_FUNCTION_PREFIX inline int _cuj_thread_idx_x()
{
    return threadIdx.x;
}

CUJ_FUNCTION_PREFIX inline int _cuj_thread_idx_y()
{
    return threadIdx.y;
}

CUJ_FUNCTION_PREFIX inline int _cuj_thread_idx_z()
{
    return threadIdx.z;
}

CUJ_FUNCTION_PREFIX inline int _cuj_block_idx_x()
{
    return blockIdx.x;
}

CUJ_FUNCTION_PREFIX inline int _cuj_block_idx_y()
{
    return blockIdx.y;
}

CUJ_FUNCTION_PREFIX inline int _cuj_block_idx_z()
{
    return blockIdx.z;
}

CUJ_FUNCTION_PREFIX inline int _cuj_block_dim_x()
{
    return blockDim.x;
}

CUJ_FUNCTION_PREFIX inline int _cuj_block_dim_y()
{
    return blockDim.y;
}

CUJ_FUNCTION_PREFIX inline int _cuj_block_dim_z()
{
    return blockDim.z;
}

#endif

CUJ_FUNCTION_PREFIX inline void _cuj_store_f32x4(float *p, float a, float b, float c, float d)
{
#ifdef CUJ_IS_CUDA
    *(float4 *)(p) = make_float4(a, b, c, d);
#else
    p[0] = a;
    p[1] = b;
    p[2] = c;
    p[3] = d;
#endif
}

CUJ_FUNCTION_PREFIX inline void _cuj_store_u32x4(unsigned *p, unsigned a, unsigned b, unsigned c, unsigned d)
{
#ifdef CUJ_IS_CUDA
    *(uint4 *)(p) = make_uint4(a, b, c, d);
#else
    p[0] = a;
    p[1] = b;
    p[2] = c;
    p[3] = d;
#endif
}

CUJ_FUNCTION_PREFIX inline void _cuj_store_i32x4(int *p, int a, int b, int c, int d)
{
#ifdef CUJ_IS_CUDA
    *(int4 *)(p) = make_int4(a, b, c, d);
#else
    p[0] = a;
    p[1] = b;
    p[2] = c;
    p[3] = d;
#endif
}

CUJ_FUNCTION_PREFIX inline void _cuj_store_f32x3(float *p, float a, float b, float c)
{
#ifdef CUJ_IS_CUDA
    *(float3 *)(p) = make_float3(a, b, c);
#else
    p[0] = a;
    p[1] = b;
    p[2] = c;
#endif
}

CUJ_FUNCTION_PREFIX inline void _cuj_store_u32x3(unsigned *p, unsigned a, unsigned b, unsigned c)
{
#ifdef CUJ_IS_CUDA
    *(uint3 *)(p) = make_uint3(a, b, c);
#else
    p[0] = a;
    p[1] = b;
    p[2] = c;
#endif
}

CUJ_FUNCTION_PREFIX inline void _cuj_store_i32x3(int *p, int a, int b, int c)
{
#ifdef CUJ_IS_CUDA
    *(int3 *)(p) = make_int3(a, b, c);
#else
    p[0] = a;
    p[1] = b;
    p[2] = c;
#endif
}

CUJ_FUNCTION_PREFIX inline void _cuj_store_f32x2(float *p, float a, float b)
{
#ifdef CUJ_IS_CUDA
    *(float2 *)(p) = make_float2(a, b);
#else
    p[0] = a;
    p[1] = b;
#endif
}

CUJ_FUNCTION_PREFIX inline void _cuj_store_u32x2(unsigned *p, unsigned a, unsigned b)
{
#ifdef CUJ_IS_CUDA
    *(uint2 *)(p) = make_uint2(a, b);
#else
    p[0] = a;
    p[1] = b;
#endif
}

CUJ_FUNCTION_PREFIX inline void _cuj_store_i32x2(int *p, int a, int b)
{
#ifdef CUJ_IS_CUDA
    *(int2 *)(p) = make_int2(a, b);
#else
    p[0] = a;
    p[1] = b;
#endif
}

CUJ_FUNCTION_PREFIX inline void _cuj_load_f32x4(float *p, float *a, float *b, float *c, float *d)
{
#ifdef CUJ_IS_CUDA
    auto v = *(float4 *)(p);
    *a = v.x;
    *b = v.y;
    *c = v.z;
    *d = v.w;
#else
    *a = p[0];
    *b = p[1];
    *c = p[2];
    *d = p[3];
#endif
}

CUJ_FUNCTION_PREFIX inline void _cuj_load_u32x4(unsigned *p, unsigned *a, unsigned *b, unsigned *c, unsigned *d)
{
#ifdef CUJ_IS_CUDA
    auto v = *(uint4 *)(p);
    *a = v.x;
    *b = v.y;
    *c = v.z;
    *d = v.w;
#else
    *a = p[0];
    *b = p[1];
    *c = p[2];
    *d = p[3];
#endif
}

CUJ_FUNCTION_PREFIX inline void _cuj_load_i32x4(int *p, int *a, int *b, int *c, int *d)
{
#ifdef CUJ_IS_CUDA
    auto v = *(int4 *)(p);
    *a = v.x;
    *b = v.y;
    *c = v.z;
    *d = v.w;
#else
    *a = p[0];
    *b = p[1];
    *c = p[2];
    *d = p[3];
#endif
}

CUJ_FUNCTION_PREFIX inline void _cuj_load_f32x3(float *p, float *a, float *b, float *c)
{
#ifdef CUJ_IS_CUDA
    auto v = *(float3 *)(p);
    *a = v.x;
    *b = v.y;
    *c = v.z;
#else
    *a = p[0];
    *b = p[1];
    *c = p[2];
#endif
}

CUJ_FUNCTION_PREFIX inline void _cuj_load_u32x3(unsigned *p, unsigned *a, unsigned *b, unsigned *c)
{
#ifdef CUJ_IS_CUDA
    auto v = *(uint3 *)(p);
    *a = v.x;
    *b = v.y;
    *c = v.z;
#else
    *a = p[0];
    *b = p[1];
    *c = p[2];
#endif
}

CUJ_FUNCTION_PREFIX inline void _cuj_load_i32x3(int *p, int *a, int *b, int *c)
{
#ifdef CUJ_IS_CUDA
    auto v = *(int3 *)(p);
    *a = v.x;
    *b = v.y;
    *c = v.z;
#else
    *a = p[0];
    *b = p[1];
    *c = p[2];
#endif
}

CUJ_FUNCTION_PREFIX inline void _cuj_load_f32x2(float *p, float *a, float *b)
{
#ifdef CUJ_IS_CUDA
    auto v = *(float2 *)(p);
    *a = v.x;
    *b = v.y;
#else
    *a = p[0];
    *b = p[1];
#endif
}

CUJ_FUNCTION_PREFIX inline void _cuj_load_u32x2(unsigned *p, unsigned *a, unsigned *b)
{
#ifdef CUJ_IS_CUDA
    auto v = *(uint2 *)(p);
    *a = v.x;
    *b = v.y;
#else
    *a = p[0];
    *b = p[1];
#endif
}

CUJ_FUNCTION_PREFIX inline void _cuj_load_i32x2(int *p, int *a, int *b)
{
#ifdef CUJ_IS_CUDA
    auto v = *(int2 *)(p);
    *a = v.x;
    *b = v.y;
#else
    *a = p[0];
    *b = p[1];
#endif
}

CUJ_FUNCTION_PREFIX inline int _cuj_u32_atomic_add(int *p, int v)
{
#ifdef CUJ_IS_CUDA
    return atomicAdd(p, v);
#else
    return CUJ_STD atomic_ref(*p).fetch_add(v);
#endif
}

CUJ_FUNCTION_PREFIX inline float _cuj_f32_atomic_add(float *p, float v)
{
#ifdef CUJ_IS_CUDA
    return atomicAdd(p, v);
#else
    return CUJ_STD atomic_ref(*p).fetch_add(v);
#endif
}

CUJ_FUNCTION_PREFIX inline unsigned _cuj_u32_atomic_add(unsigned *p, unsigned v)
{
#ifdef CUJ_IS_CUDA
    return atomicAdd(p, v);
#else
    return CUJ_STD atomic_ref(*p).fetch_add(v);
#endif
}

#define _cuj_print printf

#ifdef CUJ_IS_CUDA
__device__ void __assertfail(
    const char *message,
    const char *file,
    unsigned    line,
    const char *function,
    size_t      charSize);
#endif

CUJ_FUNCTION_PREFIX inline void _cuj_assertfail(
    const char *message,
    const char *file,
    int         line,
    const char *function)
{
#ifdef CUJ_IS_CUDA
    __assertfail(message, file, unsigned(line), function, 1);
#else
    _cuj_print(
        "assertion failed. file: %s, line: %d, func: %s, message: %s",
        file, line, function, message);
    std::abort();
#endif
}

CUJ_FUNCTION_PREFIX inline void _cuj_unreachable()
{
    // IMPROVE
}

#ifdef CUJ_IS_CUDA

CUJ_FUNCTION_PREFIX inline void _cuj_sample_tex2d_f32(unsigned long long tex, float u, float v, float *r, float *g, float *b, float *a)
{
    auto t = tex2D<float4>(tex, u, v);
    *r = t.x;
    *g = t.y;
    *b = t.z;
    *a = t.w;
}

CUJ_FUNCTION_PREFIX inline void _cuj_sample_tex2d_i32(unsigned long long tex, float u, float v, int *r, int *g, int *b, int *a)
{
    auto t = tex2D<int4>(tex, u, v);
    *r = t.x;
    *g = t.y;
    *b = t.z;
    *a = t.w;
}

CUJ_FUNCTION_PREFIX inline void _cuj_sample_tex3d_f32(unsigned long long tex, float u, float v, float w, float *r, float *g, float *b, float *a)
{
    auto t = tex3D<float4>(tex, u, v, w);
    *r = t.x;
    *g = t.y;
    *b = t.z;
    *a = t.w;
}

CUJ_FUNCTION_PREFIX inline void _cuj_sample_tex3d_i32(unsigned long long tex, float u, float v, float w, int *r, int *g, int *b, int *a)
{
    auto t = tex3D<int4>(tex, u, v, w);
    *r = t.x;
    *g = t.y;
    *b = t.z;
    *a = t.w;
}

#endif
)___"