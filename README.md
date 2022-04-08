# Cuj

Runtime program generator embedded in C++

[TOC]

## Building

**Requirements**

* LLVM 11.1.0 and (optional) CUDA 11.5 (other versions may work but haven't been tested)
* A C++20-compatible compiler

**Building with CMake**

```powershell
git clone https://github.com/AirGuanZ/cuj.git
cd cuj
mkdir build
cd build
cmake -DLLVM_DIR="llvm_cmake_config_dir" ..
```

To add CUJ into a CMake project, simply use `ADD_SUBDIRECTORY` and link against `cuj`.

## A Quick Example

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
* use existing `multi-stage programming ` tools/languages. However, to my knowledge, there is no practical MSP implementation for C/C++.

Now let's try to implement the generator with Cuj. Firstly create a Cuj context for holding everything:

```cpp
ScopedModule cuj_module;
```

Then we create a Cuj function that computes `pow_n`, where n is read from user input:

```cpp
uint32_t n = 0;
std::cout << "enter n: ";
std::cin >> n;
Function pow_n = [n](i32 x) mutable
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
    return result;
};
```

Note that the variable `x` has type `i32`, which is a Cuj type representing  `int32_t`.  The `pow_n` algorithm almost have the same form of the above `pow`, except some of its unknown parts are replaced with their corresponding Cuj types, like `int32_t x -> i32 x`. Cuj will trace the execution of the lambda , and reconstruct the algorithm with the given `n`.

Now we can generate machine code for `pow_n` and query its function pointer:

```cpp
// pow_n_func is a raw function pointer
MCJIT mcjit;
mcjit.generate(cuj_module);
auto pow_n_func = mcjit.get_function(pow_n);
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

Full source code of this example can be found in `example/pow/main.cpp`.

## Function

### Regular Function

Cuj provides various methods for defining functions. For example,

```cpp
Function add_int32 = [&](i32 a, i32 b) { return a + b; };
```

This defines a Cuj function returning sum of two 32-bit signed integers. Note that `decltype(add_int32)` is actually `Function<i32(i32)>`, whose template argument is automatically deduced by the C++ compiler.

We can also define a Cuj function with custom symbol name.

```cpp
auto add_int32 = function("add", [&](i32 a, i32 b) { return a + b; });
```

The custom symbol name `"add"` can be used for retrieving function pointer after the whole Cuj module is compiled to machine code using `MCJIT` backend, or used for finding kernel in PTX code generated by `PTXGenerator` backend.

We can directly call Cuj functions in other Cuj functions,

```cpp
Function pow2 = [&](i32 x) { return x * x; };
Function pow4 = [&](i32 x) { return pow2(x) * pow2(x); };
```

### Forward Declaration

To define a recursive function in Cuj, we need to declare it before using.

```cpp
auto fib = declare<i32(i32)>("fib"); // declare fib with full signature
fib.define([&](i32 i)                // define fib's function body
           {
               i32 result;
               $if(i <= 1)
               {
                   result = i;
               }
               $else
               {
                   result = fib(i - 1) + fib(i - 2);
               };
               return result;
           });
```

### CUDA Kernel

```cpp
auto my_cuda_kernel = kernel(
    "optional_kernel_symbol_name",
	[](ptr<f32> a, ptr<f32> b, ptr<f32> c, i32 n)
    {
        i32 idx = cstd::thread_idx_x() + cstd::block_dim_x() * cstd::block_idx_x();
        $if(idx < n)
        {
            c[idx] = a[idx] + b[idx];
        };
    });
```

## Module

Any compiled Cuj function must be registered in a `cuj::Module`. There is a global thread-local module pointer in Cuj, which will be used by current Cuj operations. We can use `Module::set_current_module` to manipulate this pointer.

```cpp
Module my_cuj_module;
Module::set_current_module(&my_cuj_module);
...
Module::set_current_module(nullptr);
```

We can also use scoped guard provided by Cuj. The following code is equivalent to the above.

```cpp
{
    ScopedModule my_cuj_module;
    ...
}
```

## Variable

### Arithmetic

```cpp
i32 x = 0;
i32 y = x + 1;
f32 fx = f32(x * 4) - 3.0f;
```

Cuj provides following arithmetic types:

* `i8/i16/i32/i64`: 8/16/32/64-bit signed integer
* `u8/u16/u32/u64`: 8/16/32/64-bit unsigned integer
* `f32/f64`: 32/64-bit floating-point number
* `boolean`: binary type

Note that Cuj doesn't allow implicit conversion between values of different arithmetic types. Use `dst_type(src_var)` to perform the explicit cast.

### Array

```cpp
arr<i32, 4> a;
for(size_t i = 0; i < a.size(); ++i)
    a[i] = i * i;
// a becomes { 0, 1, 4, 9 }
```

### Class

We can map a C++ class to a Cuj class using `CUJ_CLASS` macro.

```cpp
struct Vec3 { float x, y, z; };
CUJ_CLASS(Vec3, x, y, z);
```

Now `cxx<Vec3>` is a Cuj class that can be used in any Cuj function. It has the same non-static members as `Vec3`, except that they are all replaced with their corresponding Cuj types.

```cpp
Function make_cuj_vec3 = [](f32 x, f32 y, f32 z)
{
    cxx<Vec3> v;
    v.x = x;
    v.y = y;
    v.z = z;
    return v;
};
```

We can also add custom member functions to Cuj class.

```cpp
struct Vec3 { float x, y, z; };
CUJ_CLASS_EX(Vec3, x, y, z)
{
    CUJ_BASE_CONSTRUCTORS
    explicit Vec3(f32 v) : Vec3(v, v, v) { }
    Vec3(f32 _x, f32 _y, f32 _z) { x = _x; y = _y; z = _z; }
    f32 length() const { return cstd::sqrt(x * x + y * y + z * z); }
};
```

**Note**. In default, all Cuj classes are trivially-copyable. Cuj use this assumption optimize class object copying. If custom copy constructor or assignment operator are provided, we need to use macro `CUJ_NONE_TRIVIALLY_COPYABLE` to indicate that this class is not trivially-copyable.

```cpp
struct A { ... };
CUJ_PROXY_CLASS_EX(AProxy, A, ...)
{
    CUJ_BASE_CONSTRUCTORS
    CUJ_NONE_TRIVIALLY_COPYABLE
    AProxy(const AProxy &other) { ... }
    AProxy &operator=(const AProxy &other) { ... }
};
```

Custom class alignment can be specified with `CUJ_CLASS_ALIGNMENT`:

```cpp
CUJ_CLASS_EX(...)
{
    CUJ_CLASS_ALIGNMENT(128) // alignas(128)
    ...
};
```

We can also define a new Cuj class without relying on existing C++ class:

```cpp
CUJ_CLASS_BEGIN(MyCujClass)
    CUJ_CLASS_ALIGNMENT(64) // optional
    CUJ_MEMBER_VARIABLE(i32, x)
    CUJ_MEMBER_VARIABLE(f32, y)
CUJ_CLASS_END
```

### Pointer

```cpp
// ptr<i32> can be written as ptr
// as 'i32' can be automatically deduced
i32 x = 0;
ptr<i32> px = x.address();
*px = 1; // x becomes 1
```

We can also use `->` to access members of pointed Cuj class object.

```cpp
cxx<Vec3> v(1.0f, 2.0f, 3.0f);
ptr pv = v.address();
f32 x = (*pv).x;
f32 y = pv->y;
f32 len = pv->length();
```

### Reference

References in Cuj can be viewed as immutable pointers.

```cpp
i32 a = 0;
ref<i32> ra = a; // we can write 'ref' here as '<i32>' can be automatically deduced
ra = 1; // a becomes 1

cxx<Vec3> v;
ref rv = v;
rv.x = 1; // v.x becomes 1

ptr pv = v.address();
ref rv2 = *pv; // refer to dereferenced pointer
```

### General

Use `var` to define Cuj variables or `ref` for references when actual types can be automacially deduced by the C++ compiler.

```cpp
var a = 1;           // a: var<i32>
var b = 2.0f;        // b: var<f32>
var c = f32(a) * b;  // c: var<f32>
ref d = a;           // d: ref<i32>
var e = a.address(); // e: var<ptr<i32>>
ref f = *e;          // f: ref<i32>
```

`var<T>` can simply be treated like `T`.

## Global Variable

```cpp
ScopedModule mod;

// global variable can only be allocated within a module
auto global_i32_arr = allocate_global_memory<arr<i32, 16>>("my_symbol_name");

// constant memory in PTX
auto constant_f32 = allocate_constant_memory<f32>("constant_params");
```

Note that `allocate_constant_memory` allocates constant memory when using PTX backend and allocates normal global memory when using MC backend.

## Const Data

```cpp
// global const data
Function f = [](i32 x)
{
    ptr<i32> lut = const_data<i32>(std::array{ 1, 4, 6, 3 });
    return lut[x];
};

// string literial
Function f2 = []
{
    cstd::print("%s\n", string_literial("hello, cuj!"));
};
```

## Import Pointer

```cpp
int32_t i;
// equvialent to bitcast<ptr<i32>>(u64(&i))
var p = import_pointer(&i);
```

## Bitwise Cast

```cpp
u64 x0 = ...;
var y0 = bitcast<ptr<f32>>(x0); // num to ptr

f32 x1 = ...;
var y1 = bitcast<i32>(x1); // num to num

ptr<f32> x2 = ...;
var y2 = bitcast<i64>(x2); // ptr to num
```

## Control Flow

### If

```cpp
// a, b, c, d are Cuj variables
$if(0 <= a & a < 10)
{
    ...
}
$elif(b > 0 | c < 0)
{
    ...
}
$elif(!d)
{
    ...
}
$else
{
    ...
};
```

Note that Cuj doesn't provide `&&` and `||` operator since short-circuit evaluation cannot be implemented by operator overloading.

### Loop

```cpp
$loop
{
    $if(...)
    {
        $continue;
    }
    $if(...)
    {
        $break;
    };
    ...
};

$while(...)
{
    ...
};
```

### Switch

```cpp
$switch(i) // i must be a Cuj integer
{
$case(0)
{
    ...
    $fallthrough;
};
$case(1)
{
    ...
};
$case(2)
{
    ...
};
$default // optional
{
    ...
};
};
```

Note that Cuj will automatically insert a `break switch` after each case body. We can use `$fallthrough` to avoid that.

### Return

There are two methods to return a value in a Cuj function.

**Native C++ Return**

```cpp
auto f = function([](i32 a, i32 b)
{
    i32 ret;
    $if(a < b) { ret = 1; }
    $else      { ret = 2 };
    return ret;
});
```

There should be only one `return` statement that exits the callable object defining the function body. Cuj will use return type of the callable object to infer return type of the Cuj function. Note that Cuj always treats reference types as non-reference ones in return type inference. Therefore, we need to specify the return type as reference in that case.

```cpp
// returns i32 even through integers[index] is a reference
auto f1 = function([](ptr<i32> integers, i32 index)
{
    return integers[index];
});

// returns ref<i32>
auto f2 = function<ref<i32>>([](ptr<i32> integers, i32 index)
{
    return integers[index];
});
```

**Cuj Return**

We can also use `$return(...)` to generate a return statement in Cuj function. Cuj will not be able to infer the return type at compile time, so we need to specify the return type manually.

```cpp
auto f = function([](i32 a, i32 b)
{
    $if(a < b) { $return(1); }
    $else      { $return(2); };
});
```

### ExitScope

```cpp
i32 inlined_native_func(i32 x)
{
    i32 ret;
    $scope
    {
        $if(x <= 0)
        {
            ret = 4;
            $exit_scope;
        };
        $if(x == 9)
        {
            ret = 99;
            $exit_scope;
        }
        ret = 100;
    };
    return ret;
}

auto f = function<i32>([&](i32 x)
{
    $if(x == 1000)
    {
        $return(1999);
    };
    $return(inlined_native_func(x));
});
```

## Inline Asm

```cpp
i32 x, y, z, w;
inline_asm_volatile(
	"...", // asm code
    { { "=r", x }, { "=r", y } }, // output constraints
    { { "r", z }, { "r", w } },   // input constraints
    { "cc" }                      // clobber constraints
);
```

## Backend

### MC

```cpp
ScopedModule mod;
Function func = ...

MCJIT mcjit;
mcjit.generate(mod);
auto c_func_ptr = mcjit.get_function(func);
```

The type of `c_func_ptr` is automatically deduced by `MCJIT::get_function`. We can also specify a compatible type manually:

```cpp
auto c_func_ptr = mcjit.get_function<i32(i32)>(func);
```

Or by using function symbol name, if we haven't store the `Function` object:

```cpp
auto c_func_ptr = mcjit.get_function<i32(i32)>("func_symbol_name");
```

Note that when using `Function` object to query the C function pointer with manually-specified function type, Cuj will check whether the given type is compatible with the `Function` object. For example:

```cpp
Function func = [](ref<i32> a, ptr<cxx<Vec3>> b, f32 c)
{
    ptr<f32> ret = ...
    return ret;
};
...
// decltype(c_func_ptr1) is float*(*)(int32_t*, Vec3*, float)
auto c_func_ptr1 = mcjit.get_function(func);
// decltype(c_func_ptr2) is void*(const int32_t*, const Vec3*, float)
auto c_func_ptr2 = mcjit.get_function<void*(const int32_t*, const Vec3*, float)>(func);
// compile error
auto c_func_ptr3 = mcjit.get_function<int32_t(int32_t*, float, float)>(func);
```

All reference types are converted to corresponding pointers by MCJIT. The compatibility rules are:

```void
T* <-> const T*
T* <-> void*
T* <-> char*
T* <-> signed char*
T* <-> unsigned char*
```

### PTX

```cpp
ScopedModule mod;
...
PTXGenerator ptx_gen;
ptx_gen.generate(mod);
const std::string ptx = ptx_gen.get_ptx();
```

## Library

### Math

```cpp
// in namespace cuj::cstd

f32 abs(f32 x);
f32 mod(f32 x, f32 y);
f32 rem(f32 x, f32 y);
f32 exp(f32 x);
f32 exp2(f32 x);
f32 exp10(f32 x);
f32 log(f32 x);
f32 log2(f32 x);
f32 log10(f32 x);
f32 pow(f32 x, f32 y);
f32 sqrt(f32 x);
f32 rsqrt(f32 x);
f32 sin(f32 x);
f32 cos(f32 x);
f32 tan(f32 x);
f32 asin(f32 x);
f32 acos(f32 x);
f32 atan(f32 x);
f32 atan2(f32 y, f32 x);
f32 ceil(f32 x);
f32 floor(f32 x);
f32 trunc(f32 x);
f32 round(f32 x);
boolean isfinite(f32 x);
boolean isinf(f32 x);
boolean isnan(f32 x);

f64 abs(f64 x);
f64 mod(f64 x, f64 y);
f64 rem(f64 x, f64 y);
f64 exp(f64 x);
f64 exp2(f64 x);
f64 exp10(f64 x);
f64 log(f64 x);
f64 log2(f64 x);
f64 log10(f64 x);
f64 pow(f64 x, f64 y);
f64 sqrt(f64 x);
f64 rsqrt(f64 x);
f64 sin(f64 x);
f64 cos(f64 x);
f64 tan(f64 x);
f64 asin(f64 x);
f64 acos(f64 x);
f64 atan(f64 x);
f64 atan2(f64 y, f64 x);
f64 ceil(f64 x);
f64 floor(f64 x);
f64 trunc(f64 x);
f64 round(f64 x);
boolean isfinite(f64 x);
boolean isinf(f64 x);
boolean isnan(f64 x);

f32 min(f32 a, f32 b);
f32 max(f32 a, f32 b);

f64 min(f64 a, f64 b);
f64 max(f64 a, f64 b);

// returns a when cond is true, otherwise returns b
template<typename T>
T select(
    const boolean &cond,
    const T       &a,
    const T       &b);
```

### Atomic

```cpp
// in namespace cuj::cstd

i32 atomic_add(ptr<i32> dst, i32 val);
u32 atomic_add(ptr<u32> dst, u32 val);
f32 atomic_add(ptr<f32> dst, f32 val);
```

### CUDA

```cpp
// in namespace cuj::cstd
// available only when using PTX backend

i32 thread_idx_x();
i32 thread_idx_y();
i32 thread_idx_z();

i32 block_idx_x();
i32 block_idx_y();
i32 block_idx_z();

i32 block_dim_x();
i32 block_dim_y();
i32 block_dim_z();

void sample_texture2d_1f(u64 texture_object, f32 u, f32 v, ref<f32> r);
void sample_texture2d_3f(u64 texture_object, f32 u, f32 v, ref<f32> r, ref<f32> g, ref<f32> b);
void sample_texture2d_4f(u64 texture_object, f32 u, f32 v, ref<f32> r, ref<f32> g, ref<f32> b, ref<f32> a);

void sample_texture2d_1i(u64 texture_object, f32 u, f32 v, ref<i32> r);
void sample_texture2d_3i(u64 texture_object, f32 u, f32 v, ref<i32> r, ref<i32> g, ref<i32> b);
void sample_texture2d_4i(u64 texture_object, f32 u, f32 v, ref<i32> r, ref<i32> g, ref<i32> b, ref<i32> a);
```

### System

```cpp
// in namespace cuj::cstd

// e.g. print("%d\n%s", i32(42), string_literial("hello, cuj!"));
i32 print(const std::string &fmt_str, Args...args);

// in any cuj function
// assertion generation is controled by cuj::gen::Options::enable_assert
CUJ_ASSERT(cuj_bool_expr);

void unreachable();
```

## Example

### example/pow

Full source code of `A Quick Example`.

### example/sdf

A simple CUDA path tracer based on SDF marching. The scene comes from [taichi sdf renderer](https://github.com/taichi-dev/taichi/blob/master/examples/rendering/sdf_renderer.py).

![](./example/sdf/result.png)

### example/tex

Example of sampling cuda texture object.

```cpp
std::vector<float> tex_data =
{
    1, 0, 0, 1,
    0, 1, 0, 1,
    0, 0, 1, 1,
    
    0, 1, 0, 1,
    0, 0, 1, 1,
    1, 0, 0, 1,
    
    0, 0, 1, 1,
    1, 0, 0, 1,
    0, 1, 0, 1,
};
```

![](./example/tex/result.png)
