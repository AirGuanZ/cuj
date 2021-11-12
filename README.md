# CUJ

Run-time program generator embedded in C++

## Building

### Requirements

* LLVM 11.1.0 and (optional) CUDA 11.1 (other versions may work but haven't been tested)
* A C++17-compatible compiler

### Building with CMake

```powershell
git clone https://github.com/AirGuanZ/cuj.git
cd cuj
mkdir build
cd build
cmake -DLLVM_DIR="llvm_cmake_config_dir" ..
```

To add CUJ into a CMake project, simply use `ADD_SUBDIRECTORY` and link against `cuj`.

## Usage

### A Quick Example

[Exponentiation by squaring](https://en.wikipedia.org/wiki/Exponentiation_by_squaring) is a fast algorithm for computing positive integer powers:

```cpp
int64_t pow(int32_t x, uint32_t n)
{
    int64_t result = 1;
    int64_t base = x;
    while(n)
    {
        if(n & 1)
            result *= base;
        base *= base;
        n >>= 1;
    }
    return result;
}
```

However, we always have a faster method `pow_n` for a fixed n. For example, the following function is better for computing `pow(x, 5)` than general `pow`:

```cpp
int64_t pow5(int32_t x)
{
    int64_t b1 = x;
    int64_t b2 = b1 * b1;
    int64_t b4 = b2 * b2;
    return b4 * b1;
}
```

The program may need to read `n` from a configuration file or user input, then evaluate `pow(x, n)` for millions of different `x`. The problem is: how can we efficiently generate `pow_n` after reading `n`? Here are some solutions:

* generate source code (in C, LLVM IR, etc) for computing `pow_n`, then compile it into executable machine code. When the algorithm becomes more complicated than `pow`, the generator itself also becomes hard to code.
* use existing `partial evaluation (PE)` or `multi-stage programming (MSP)` tools. However, to my knowledge, there is no practical PE/MSP implementation for C/C++.

Now let's try to implement the generator with CUJ. Firstly create a CUJ context for holding everything:

```cpp
ScopedContext context;
```

Then we create a CUJ function that computes `pow_n`, where n is read from user input:

```cpp
uint32_t n = 0;
std::cout << "Enter n: ";
std::cin >> n;

auto pow_n = to_callable<int64_t>(
    [n](i32 x) mutable
{
    i64 result = 1;
    i64 base = x;
    while(n)
    {
        if(n & 1)
            result = result * base;
        base = base * base;
        n >>= 1;
    }
    $return(result);
});
```

Note that the variable `x` has type `i32`, which is a CUJ type representing  `int32_t`. And the `return` statement is replaced with `$return(...)`, which tolds CUJ to generate a return instruction. The `pow_n` algorithm almost have the same form of the above `pow`, except some of its unknown parts are replaced with their corresponding CUJ types, like `int32_t x -> i32 x`. CUJ will trace the execution of the lambda in `to_callable`, and reconstruct the algorithm with the given `n`.

Now we can generate machine code for `pow_n` and query its function pointer:

```cpp
// pow_n_func is a raw function pointer
auto codegen_result = context.gen_native_jit();
auto pow_n_func = codegen_result.get_function(pow_n);

// test output
std::cout << "n = " << n << std::endl;
for(int i = 0; i <= 9; ++i)
    std::cout << i << " ^ n = " << pow_n_func(i) << std::endl;
```

Enter `5`, and we will get:

```
n = 5
0 ^ n = 0
1 ^ n = 1
2 ^ n = 32
...
8 ^ n = 32768
9 ^ n = 59049
```

Full source code of this example can be found in `example/native_codegen/main.cpp`.

#### CUJ Context

Any CUJ operation must be done with a context. There is a global context stack
in CUJ, and the context on the top of the stack will be used by current CUJ operations. We can use `push_context` and `pop_context` to manipulate this stack.

```cpp
{
    cuj::Context context;
    cuj::push_context(&context);
    // creating functions, generating machine code, ...
    cuj::pop_context();
}
```

We can also use scoped guard provided by CUJ. The following two pieces of codes are equivalent to the above.

```cpp
{
    cuj::Context context;
    CUJ_SCOPED_CONTEXT(&context);
    // creating functions, generating machine code, ...
}
```

```cpp
{
    cuj::ScopedContext context;
    // creating functions, generating machine code, ...
}
```

### Define Functions

#### Declare Regular Functions

```cpp
auto func = ctx.declare_function<i32(i32, f32)>(/* optional name */ "func_name");

// define func's body later
func.define([&](i32 x, f32 y) { ... });

// or:
to_callable<i32>(func.get_name(), [&](i32 x, f32 y) { ... });

// or any syntax that can define a regular function:
ctx.begin_function<i32(i32, i32)>(func.get_name());
...
ctx.end_function();
```

#### Define Regular Functions

```cpp
auto my_add_float = to_callable<float>(
    "add_float", // (optional) symbol name
    [](f32 a, f32 b) { $return(a + b); });
```
#### Recursive Function

```cpp
// just declare it before use
auto fib = declare<i32(i32)>();
fib.define([&](i32 i)
{
    $if(i <= 0)
    {
        $return(0);
    }
    $elif(i == 1)
    {
        $return(1);
    }
    $else
    {
        $return(fib(i - 1) + fib(i - 2));
    };
});
```

#### Define CUDA Kernels

```cpp
auto vec_add_float = to_kernel(
    "vec_add",
    [&](Pointer<float> A,
        Pointer<float> B,
        Pointer<float> C,
        i32            N)
    {
        i32 i = cuda::thread_index_x() + cuda::block_index_x() * cuda::block_dim_x();
        $if(i < N)
        {
            C[i] = math::sqrt(A[i] + B[i]);
        };
    });
```

#### Import Host Functions

```cpp
// convert a host callable object to a cuj function
// cuj will automatically handle the object's lifetime
auto add_float = import_function([](float a, float b) { return a + b; });

// capture is allowed
int n = 4;
auto set_n = import_function([&n](int value) { n = value; });
```

**Note**. after importing a host function, native-jit becomes the only valid backend.

#### Call Functions

```cpp
auto add_float = to_callable<float>([](f32 a, f32 b) { $return(a + b); });
auto another_function = to_callable<float>([&]
{
    f32 x = 1;
    f32 z = add_float(x, 2.0f);
    $return(z);
});
```

### Define Variables

**Note** Avoid using keyword `auto` to define CUJ variable, which may cause copy elision and break the recorded operation order.

#### Define Arithmetic Variables

```cpp
Var<int32_t> x = 0;
i32            y = x + 1;
```

You can use `cuj::Var<T>` for converting a C++ arithmetic type `T` to its corresponding CUJ type. You can also use following aliases for convenience

```cpp
f32/f64        -> Var<float/double>
i8/i16/i32/i64 -> Var<int8_t/int16_t/int32_t/int64_t>
u8/u16/u32/u64 -> Var<uint8_t/uint16_t/uint32_t/uint64_t>
boolean        -> Var<bool>
```

#### Define Pointers

```cpp
Var<float*>  px = nullptr;
Pointer<float> py = px;

f32 z = 0.0f;
Pointer<f32> pz = z.address(); // use x.address() to get address of a CUJ variable
*pz = 4.0f;                    // modify z via pz
pz.deref() = 5.0f;             // pz.deref() is equivalent to *pz

Pointer<Pointer<f32>> ppz = pz.address(); // nested pointer type is supported
```

`px, py, pz` have the same CUJ type.

#### Define Arrays

```cpp
Array<int, 5> arr0;
Array<i32, 6> arr1;
Array<Array<i32, 4>, 3> arr2;
```

#### Define Classes

```cpp
struct Float3 { float x, y, z; };

// Variable<Float3> is mapped to CFloat3,
// Variable<Float3*> is mapped to Pointer<CFloat3>, etc
CUJ_PROXY_CLASS(CFloat3, Float3, x, y, z)
{
    // define additional member functions
        
    using CUJBase::CUJBase;
    
    CFloat3(f32 _x, f32 _y, f32 _z)
    {
        x = _x;
        y = _y;
        z = _z;
    }
    
    auto length() const
    {
        return builtin::math::sqrt(x * x + y * y + z * z);
    }
};

void foo()
{
    // ...
    
    ScopedContext ctx;
    
    auto sum = to_callable<f32>([](const CFloat3 &v)
    {
        $return(v.x + v.y + v.z);
    });
    
    // ...
}
```

You can also use `CUJ_CLASS(Float3, x, y, z)` to make the proxy class anonymous.

It's recommanded to use `CUJ_CLASS` and `CUJ_PROXY_CLASS` in the same namespace that defining the original class. Otherwise, you should add `CUJ_REGISTER_GLOBAL_CLASS(Float3, CFloat3)` to the global namespace. For example:

```cpp
namespace N
{
    struct Float3 { float x, y, z; };
}

namespace M
{
	CUJ_PROXY_CLASS(CFloat3, Float3, x, y, z) { using CUJBase::CUJBase; };
}

// this line must be added to the global namespace
// as CFloat3 is not defined in the same namespace of Float3
CUJ_REGISTER_GLOBAL_CLASS(N::Float3, M::CFloat3);
```

CUJ array/class objects can be passed as function arguments or returned by a CUJ function. For example:

```cpp
auto array_sum = to_callable<int>(
    [](const Array<i32, 3> &arr)
{
    $return(arr[0] + arr[1] + arr[2]);
});

auto make_vector3 = to_callable<Vector3>(
	[](f32 x, f32 y, f32 z)
{
    Vector3 result(x, y, z);
    $return(result);
});
```

#### Cast Values

```cpp
i32 i = 2;
f32 f = cast<f32>(i); // f is 2.0f

Pointer<void> p = ...;
Pointer<f32>  pf = ptr_cast<f32>(p);

Variable<size_t> ip = ptr_to_uint(p);
Pointer<f32> pip = uint_to_ptr<f32>(ip);

float float_value = 1.0f;
Pointer<f32> pl = ptr_literal(&float_value); // pl can only be dereferenced when using host backend
```

#### Define Constant Data

You can convert C++ native data to CUJ pointer. Converted data and be accessed when using any backends and its lifetime is maintained by CUJ backends.

```cpp
int data[5] = { 1, 2, 3, 4, 5 };
Pointer<int> cuj_data_pointer = const_data(data);

int *p_data = ...;
Pointer<float> cuj_data_pointer2 = const_data<float>(p_data, total_bytes_of_data);

Pointer<char> cuj_string = "hello, world!"_cuj;
```

### Control Flow

#### Branch

```cpp
auto func = to_callable<int32_t>(
    [](i32 x, i32 y)
{
    $if(x == 100 && y == 200)
    {
        $return(999);
    };
    
    $if(x > 0)
    {
        $return(x + y);
    }
    $else // optional
    {
        $return(x - y);
    };
});
```

```cpp
auto min_func = to_callable<int32_t>(
	[](i32 x, i32 y)
{
    $return(select(x < y, x, y));
});
```

#### Loop

```cpp
auto sum = to_callable<int32_t>(
    [](Pointer<i32> p, i32 N)
{
	i32 result = 0, i = 0;
    $while(i < N)
    {
        result = result + p[i];
        i = i + 1;
    };
    $return(result);
});
```

#### Switch

```cpp
auto entry1 = to_callable<void>(
    [](i32 i, Pointer<i32> a, Pointer<i32> b)
{
    $switch(i)
    {
        $case(0) { *a = 1; $fallthrough; }; // 'break' became the default behavior
        $case(1) { *b = 1; };
        $case(2) { *a = 1; *b = 1; };
    };
});

auto entry2 = to_callable<int32_t>(
    [](i32 i)
{
    $switch(i)
    {
        for(int j = 1; j <= 4; ++j)
            $case(j) { $return(j + 1); };
        $default{ $return(0); };
    };
});
```

### Backends

#### Native JIT

```cpp
ScopedContext context;
auto add_float_handle = to_callable<float>(
    "add_float", [](f32 a, f32 b) { $return(a + b); });
auto native_jit = context.gen_native_jit();

// query c function pointer via symbol name
auto add_float_func_pointer1 = native_jit.get_function_by_name<float(float, float)>("add_float");
// query c function pointer via CUJ function handle
auto add_float_func_pointer2 = native_jit.get_function(add_float_handle);

assert(add_float_func_pointer1 == add_float_func_pointer2);
assert(add_float_func_pointer1(1.0f, 2.0f) == 3.0f);
```

**Note**

* class/array-typed function arguments are translated to `void*` in corresponding function pointer.
* class/array-typed function return value is translated to an additional `void*` argument at the first position.

#### PTX

```cpp
ScopedContext context;
// ...define functions/kernels

gen::Options options;
options.fast_math = true;

// enable O3 optimization; enable fast math
const std::string ptx = context.gen_ptx(options);

// another ptx backend using nvrtc
const std::string ptx2 = context.gen_ptx_nvrtc(options);
```

#### C

```cpp
ScopedContext context;
// ...define functions/kernels

// generate c source code
const std::string c_src = context.gen_c(false);

// generate cuda source code
const std::string cu_src = context.gen_c(true);
```

### Builtin Functions

#### Math

```cpp
// in namespace cuj::builtin::math
```

##### Basic

```cpp
f32 atomic_add(const Pointer<f32> &dst, const f32 &val);
f64 atomic_add(const Pointer<f64> &dst, const f64 &val);

f32 abs      (const f32 &x);
f32 mod      (const f32 &x, const f32 &y);
f32 remainder(const f32 &x, const f32 &y);
f32 exp      (const f32 &x);
f32 exp2     (const f32 &x);
f32 exp10    (const f32 &x);
f32 log      (const f32 &x);
f32 log2     (const f32 &x);
f32 log10    (const f32 &x);
f32 pow      (const f32 &x, const f32 &y);
f32 sqrt     (const f32 &x);
f32 rsqrt    (const f32 &x);
f32 sin      (const f32 &x);
f32 cos      (const f32 &x);
f32 tan      (const f32 &x);
f32 asin     (const f32 &x);
f32 acos     (const f32 &x);
f32 atan     (const f32 &x);
f32 atan2    (const f32 &y, const f32 &x);
f32 ceil     (const f32 &x);
f32 floor    (const f32 &x);
f32 trunc    (const f32 &x);
f32 round    (const f32 &x);
i32 isfinite (const f32 &x);
i32 isinf    (const f32 &x);
i32 isnan    (const f32 &x);

f64 abs      (const f64 &x);
f64 mod      (const f64 &x, const f64 &y);
f64 remainder(const f64 &x, const f64 &y);
f64 exp      (const f64 &x);
f64 exp2     (const f64 &x);
f64 exp10    (const f64 &x);
f64 log      (const f64 &x);
f64 log2     (const f64 &x);
f64 log10    (const f64 &x);
f64 pow      (const f64 &x, const f64 &y);
f64 sqrt     (const f64 &x);
f64 rsqrt    (const f64 &x);
f64 sin      (const f64 &x);
f64 cos      (const f64 &x);
f64 tan      (const f64 &x);
f64 asin     (const f64 &x);
f64 acos     (const f64 &x);
f64 atan     (const f64 &x);
f64 atan2    (const f64 &y, const f64 &x);
f64 ceil     (const f64 &x);
f64 floor    (const f64 &x);
f64 trunc    (const f64 &x);
f64 round    (const f64 &x);
i32 isfinite (const f64 &x);
i32 isinf    (const f64 &x);
i32 isnan    (const f64 &x);

// static_assert(std::is_arithmetic_v<T>);
template<typename T>
Var<T> min(const Var<T> &lhs, const Var<T> &rhs);
template<typename T>
Var<T> max(const Var<T> &lhs, const Var<T> &rhs);
template<typename T>
Var<T> clamp(const Var<T> &x, const Var<T> &min_x, const Var<T> &max_x);
```

##### Vec1/2/3/4

```cpp
// this is only for interface illustration.
// actual implementation can be more complex.
template<typename T>
class Vec3
{
public:
    
    Var<T> x;
    Var<T> y;
    Var<T> z;
    
    Vec3();
    Vec3(Var<T> value);
    Vec3(Var<T> x, Var<T> y, Var<T> z);
    Value<T> length_square() const;

    Value<T> length() const;
    Value<T> min_elem() const;
    Value<T> max_elem() const;
    
    // returned value is a reference
    Value<T> operator[](const Var<size_t> &i) const;
    
    Vec3<T> normalize() const;
};

using Vec3f = Vec3<float>;
using Vec3d = Vec3<double>;
using Vec3i = Vec3<int>;

template<typename T>
Vec3<T> make_vec3();
template<typename T>
Vec3<T> make_vec3(const Value<T> &v);
template<typename T>
Vec3<T> make_vec3(const Value<T> &x, const Value<T> &y, const Value<T> &z);

inline Vec3f make_vec3f();
inline Vec3f make_vec3f(const f32 &v);
inline Vec3f make_vec3f(const f32 &x, const f32 &y, const f32 &z);

inline Vec3d make_vec3d();
inline Vec3d make_vec3d(const f64 &v);
inline Vec3d make_vec3d(const f64 &x, const f64 &y, const f64 &z);

inline Vec3i make_vec3i();
inline Vec3i make_vec3i(const i32 &v);
inline Vec3i make_vec3i(const i32 &x, const i32 &y, const i32 &z);

template<typename T>
Vec3<T> operator-(const Vec3<T> &v);

template<typename T>
Vec3<T> operator+(const Vec3<T> &lhs, const Vec3<T> &rhs);
template<typename T>
Vec3<T> operator-(const Vec3<T> &lhs, const Vec3<T> &rhs);
template<typename T>
Vec3<T> operator*(const Vec3<T> &lhs, const Vec3<T> &rhs);
template<typename T>
Vec3<T> operator/(const Vec3<T> &lhs, const Vec3<T> &rhs);

template<typename T>
Vec3<T> operator+(const Vec3<T> &lhs, const Value<T> &rhs);
template<typename T>
Vec3<T> operator-(const Vec3<T> &lhs, const Value<T> &rhs);
template<typename T>
Vec3<T> operator*(const Vec3<T> &lhs, const Value<T> &rhs);
template<typename T>
Vec3<T> operator/(const Vec3<T> &lhs, const Value<T> &rhs);

template<typename T>
Vec3<T> operator+(const Value<T> &lhs, const Vec3<T> &rhs);
template<typename T>
Vec3<T> operator-(const Value<T> &lhs, const Vec3<T> &rhs);
template<typename T>
Vec3<T> operator*(const Value<T> &lhs, const Vec3<T> &rhs);
template<typename T>
Vec3<T> operator/(const Value<T> &lhs, const Vec3<T> &rhs);

template<typename T>
Value<T> dot(const Vec3<T> &a, const Vec3<T> &b);
template<typename T>
Value<T> cos(const Vec3<T> &a, const Vec3<T> &b);
template<typename T>
Vec3<T> cross(const Vec3<T> &a, const Vec3<T> &b);
```

#### System

```cpp
// in namespace cuj::builtin::system

void print(const Pointer<char> &msg);

Pointer<void> malloc(const Value<size_t> &bytes);
void free(const ast::Pointer<void> &ptr);
```

**Note**: `malloc` and `free` are only available on NativeJIT/C backends.

#### Assertion

```cpp
// in CUJ function
CUJ_ASSERT(i >= 0);
```

#### CUDA Specific Functions

```cpp
// in namespace cuj::builtin::cuda

using TextureObject = i64;

math::Vec1f sample_texture2d_1f(TextureObject tex, f32 u, f32 v);
math::Vec2f sample_texture2d_2f(TextureObject tex, f32 u, f32 v);
math::Vec3f sample_texture2d_3f(TextureObject tex, f32 u, f32 v);
math::Vec4f sample_texture2d_4f(TextureObject tex, f32 u, f32 v);

math::Vec1i sample_texture2d_1i(TextureObject tex, f32 u, f32 v);
math::Vec2i sample_texture2d_2i(TextureObject tex, f32 u, f32 v);
math::Vec3i sample_texture2d_3i(TextureObject tex, f32 u, f32 v);
math::Vec4i sample_texture2d_4i(TextureObject tex, f32 u, f32 v);

Value<int> thread_index_x();
Value<int> thread_index_y();
Value<int> thread_index_z();

Value<int> block_index_x();
Value<int> block_index_y();
Value<int> block_index_z();

Value<int> block_dim_x();
Value<int> block_dim_y();
Value<int> block_dim_z();

void sync_block_threads();
```

