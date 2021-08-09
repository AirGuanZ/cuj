# CUJ

Run-time program generator embedded in C++

## Building

## Requirements

* LLVM 11.1.0 and (optional) CUDA 10.2/11.1 (other versions may work but haven't been tested)
* A C++17 compiler

### Building with CMake

```
git clone https://github.com/AirGuanZ/cuj.git
cd cuj
mkdir build
cd build
cmake -DLLVM_DIR="llvm_cmake_config_dir" ..
```

To add CUJ into a CMake project, simply use `ADD_SUBDIRECTORY` and link with target `cuj`.

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

#### Define Regular Functions

```cpp
auto my_add_float = to_callable<float>(
    "add_float", // (optional) symbol name
    [](f32 a, f32 b) { $return(a + b); });
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
Value<int32_t> x = 0;
i32            y = x + 1;
```

You can use `cuj::Value<T>` for converting a C++ arithmetic type `T` to its corresponding CUJ type. You can also use following aliases for convenience

```cpp
f32/f64        -> Value<float/double>
i8/i16/i32/i64 -> Value<int8_t/int16_t/int32_t/int64_t>
u8/u16/u32/u64 -> Value<uint8_t/uint16_t/uint32_t/uint64_t>
boolean        -> Value<bool>
```

#### Define Pointers

```cpp
Value<float*>  px = nullptr;
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
class Vector3Impl : public ClassBase<Vector3Impl>
{
public:
    
    CUJ_DEFINE_CLASS(Vector3Impl)
    
    $mem(float, x);
    $mem(float, y);
    $mem(float, z);
    
    using ClassBase<Vector3Impl>::ClassBase;
    
    // (optional) custom constructors
    // first parameter must be of type 'ClassAddress'
    explicit Vector3Impl(ClassAddress addr) : Vector3Impl(addr, 0) { }
    
    Vector3Impl(ClassAddress addr, f32 _x, f32 _y, f32 _z)
        : ClassBase<Vector3>(addr)
    {
        x = _x;
        y = _y;
        z = _z;
    }
};

using Vector3 = ClassValue<Vector3Impl>;
```

We can then use `Vector3` like other normal types:

```cpp
auto add_vector3 = to_callable<Vector3>(
    [](const Vector3 &lhs, const Vector3 &rhs)
{
    Vector3 ret;
    ret->x = lhs->x + rhs->x;
    ret->y = lhs->y + rhs->y;
    ret->z = lhs->z + rhs->z;
    $return(ret);
});
```

Note that member variables are accessed with overloaded `->`, which is conflict with the native `->` operator of pointers. That means we cannot use `->` to access members of pointed class object in CUJ. Instead, explicit dereference must be done first --

```cpp
Value<Vector3*> p = ...;
p->x         = 5; // compile error
(*p)->x      = 5; // ok
p.deref()->x = 5; // ok
```

#### Misc

CUJ array/class objects can be freely passed as function arguments or return values. For example:

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
const std::string ptx = context.gen_ptx();
```

### Builtin Functions

#### Math

#### System

#### Assertion

#### Special Registers

