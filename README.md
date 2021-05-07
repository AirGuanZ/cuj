# CUJ

[TOC]

Run-time program generator embedded in C++

## Requirements

* LLVM 11.1.0 and (optional) CUDA 10.2/11.1 (other versions may work but haven't been tested)
* A C++17 compiler

## Usage

### Building

```
git clone https://github.com/AirGuanZ/cuj.git
cd cuj
mkdir build
cd build
cmake -DLLVM_DIR="llvm_cmake_config_dir" ..
```

To add CUJ into a CMake project, simply use `ADD_SUBDIRECTORY` and link with target `cuj`.

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

However, we always have a faster method `pow_n` for computing `pow(x, n)` for a fixed n. For example, the following function is better for computing `pow(x, 5)` than general `pow`:

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

* generate source code (in C, LLVM IR, etc) for computing `pow_n`, then compile it into executable machine code. When the algorithm becomes more complicated than `pow`, the generator may become harder to code.
* use existing `partial evaluation` (PE) tools. However, there is no practical PE implementation for C/C++ or can be easily integrated into this context.

Now lets try to implement the generator with CUJ. Firstly create a CUJ context for holding everything about the generated code:

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

Note that the variable `x` is typed with `i32`, which is a CUJ variable representing a `int32_t` variable. And the `return` statement is replaced with `$return(...)`, which tolds CUJ to generate a return instruction. The `pow_n` algorithm almost have the same form of the above `pow`, except some of its unknown parts are replaced with their corresponding CUJ types, like `int32_t x -> i32 x`. CUJ will trace the execution of the lambda in `to_callable`, and reconstruct the algorithm with a fixed `n`.

Now we can generate exeutable machine code for `pow_n` and query its function pointer:

```cpp
// pow_n_func is a raw function pointer
auto codegen_result = context.gen_native_jit();
auto pow_n_func = codegen_result.get_symbol(pow_n);

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

### CUJ Context

Any CUJ operation must be done with a context. There is a global context stack
in CUJ, and the context on the top of the stack will be used by current CUJ operations. We can use `push_context` and `pop_context` to manipulate this stack.

```cpp
{
    cuj::Context context;
    cuj::push_context(&context);
    // operations like creating functions and generating machine code
    cuj::pop_context();
}
```

We can also use scoped guard provided by CUJ. The following two pieces of codes are equivalent to the above.

```cpp
{
    cuj::Context context;
    CUJ_SCOPED_CONTEXT(&context);
    // operations like creating functions and generating machine code
}
```

```cpp
{
    cuj::ScopedContext context;
    // operations like creating functions and generating machine code
}
```

### Define Functions

The simplest way to define a CUJ function is calling `to_callable` with a callable object describing how to build the function body. For example:

```cpp
{
    using namespace cuj;
    ScopedContext context;

    auto my_add_float = to_callable<float>(
        "add_float", [](f32 a, f32 b) { $return(a + b); });

    // other operations
}
```

The above codes define a CUJ function `add_float`, which returns the sum of two given floating point numbers.

`f32 a, f32 b` means this CUJ function receives two arguments of `float` type, where `f32` is a CUJ representation of `float`, allowing CUJ to trace what happened to these two arguments.

Note that the functor returns `void`, even if it defines a CUJ function which returns `float`. Instead, the return value type of CUJ function is specified by the template argument of `to_callable`, and the real `return` statement is replaced with CUJ's `$return`.

All variables, operators and statements contained in the defined CUJ function are replaced with their corresponding CUJ representations (`float` -> f32, for example). CUJ constructs the function body by tracing operations on these representations. We will see more examples later.

There are some variants of `to_callable`:

* `to_callable<ReturnType>(Functor)` omits the function name. CUJ will automaticly assign an unique name to it.
* `to_callable<ReturnType>(Type, Functor)` specifies the function type (see `cuj::ir::FunctionType`). This can be used for defining special functions like device functions in NVIDIA PTX.
* `to_callable<ReturnType>(Name, Type, Functor)` specifies both of the name and type.

We can also define a CUJ function by explicitly calling `Context::begin_function` and `Context::end_function`. For example:

```cpp
{
    using namespace cuj;
    ScopedContext context;

    auto add_float = context.begin_function<float(float, float)>("add_float");
    {
        $arg(float, a);
        $arg(float, b);
        $return(a + b);
    }
    context.end_function();

    // other operations
}
```

These codes define a CUJ function that is exactly the same as the previous one. When using `Context::begin_function`, we must specify the whole function signature, and declare arguments with CUJ macro `$arg`.

### Call Functions

A CUJ function can not be called within another CUJ function. The calling syntax is very similar to C++ —— just use the return value of `to_callable` or `Context::begin_function` as a functor. For example, to call `my_add_float` in another function:

```cpp
{
    using namespace cuj;
    ScopedContext context;

    auto my_add_float = to_callable<float>(
        [](f32 a, f32 b) { $return(a + b); });

    auto another_func = to_callable<float>([&]
    {
        f32 x = 1;
        f32 z = my_add_float(x, 2.0f);
        $return(z);
    });
    
    // other operations
}
```

### Variables

Variables in CUJ can be of the following types:

* basic
* pointer
* array
* class

We can use `cuj::Value` to convert a C++ type to its corresponding CUJ type. To define a CUJ variable, simply use `Value<CPPType> var_name`. For example:

```cpp
{
    using namespace cuj;
    ScopedContext context;
    
    auto my_func = to_callable<int32_t>(
        [](i32 x, i32 y)
    {
        i32 x2 = x * x;
        i32 y2 = y * y;
        $return(x2 * y + x * y2);
    });

    // other operations
}
```

Function `my_func` receives two integers `x, y`, and returns `x * x * y + x * y * y`.

CUJ predefines some basic types for convenience:

```cpp
f32/f64        -> Value<float/double>
i8/i16/i32/i64 -> Value<int8_t/int16_t/int32_t/int64_t>
u8/u16/u32/u64 -> Value<uint8_t/uint16_t/uint32_t/uint64_t>
boolean        -> Value<bool>
```

For any "left value" `val` in CUJ, use `val.address()` to get a pointer to it. Pointer-type values can be used like in C++. For example:

```cpp
Value<int32_t>  x  = 5;
Value<int32_t*> px = x.address();
*px = 0; // modify x to 0
```

Class type in CUJ is very similar to struct type in C. To define a CUJ class `C`, we need to derive it from `cuj::ClassBase<C>`, define member variables using macro `$mem`, and export necessary constructors. Take 3D vector as an example:

```cpp
class Float3 : public ClassBase<Float3>
{
public:

    $mem(float, x);
    $mem(float, y);
    $mem(float, z);

    using ClassBase::ClassBase;
};
```

We can then use `Float3` like other normal types:

```cpp
auto add_float3 = to_callable<Float3>(
    [](const Value<Float3> &lhs, const Value<Float3> &rhs)
{
    Value<Float3> ret;
    ret->x = lhs->x + rhs->x;
    ret->y = lhs->y + rhs->y;
    ret->z = lhs->z + rhs->z;
    $return(ret);
});
```

Note that member variables are accessed with overloaded `->`, which is conflict with the native `->` operator of pointers. That means we cannot use `->` to access members of pointed class object in CUJ. Instead, explicit dereference must be done first --

```cpp
Value<Float3*> p_float3 = ...;
p_float3->x         = 5; // error
(*p_float3)->x      = 5; // ok
p_float3.deref()->x = 5; // ok
```

**Note** CUJ Class doesn't support inheritance. 

*TODO: USER-DEFINED CONSTRUCTOR, OPERATOR OVERLOADING AND CLASS TEMPLATE*

### Control Flow Statements

Control flow statements in CUJ:

```cpp
$if(...)
{
    // do somthing
};

$if(...)
{
    // do somthing
}
$else
{
    // do somthing else
};

$while(...)
{
    // do somthing

    $if(...)
    {
        $continue;
    };

    $if(...)
    {
        $break;
    };
};

$return(...);
```

Branches in CUJ are not branches in C++ -- they just aid CUJ to construct the control flow of generated code. For example, in `$if(x > 0) { A } $else { B }`, both `A` and `B` will be executed in C++. This allows CUJ to record statements in all branch destinations and create real branches in its generated code.

*TODO: MORE DETAILS*

### Backends

There are two backends in CUJ currently: NativeJIT and PTX. Both backends are based on LLVM.

The NativeJIT backend convert a CUJ context to a LLVM JIT module, and we can simply get function pointer in this module. To create a JIT module, use:

```cpp
auto jit = context.gen_native_jit();
```

Function pointers can be queried by its CUJ function name or CUJ function handle, for example:

```cpp
{
    ScopedContext context;
    auto add_float_handle = to_callable<float>(
        "add_float", [](f32 a, f32 b) { $return(a + b); });
    auto native_jit = context.gen_native_jit();
    
    auto add_float_func_pointer1 = native_jit.get_symbol<float(float, float)>("add_float");
    auto add_float_func_pointer2 = native_jit.get_symbol(add_float_handle);

    assert(add_float_func_pointer1 == add_float_func_pointer2);
    assert(add_float_func_pointer1(1.0f, 2.0f) == 3.0f);
}
```

When querying function pointer by its name, we need to specify the function signature; when querying function pointer by its handle, NativeJIT can automatically infer the function signature from given handle.

**Note**

* class/array-typed function arguments are translated to `void*` in corresponding function pointer.
* class/array-typed function return value is translated to an additional `void*` argument at the first position.

The PTX backend convert a CUJ context to a NVIDIA PTX string:

```cpp
{
    ScopedContext context;
    // ...create CUJ functions
    const std::string ptx = context.gen_ptx();
    std::cout << ptx << std::endl;
}
```

We can use CUDA driver API to manipulate the generated PTX string. See `./example/vec_sqrt_add` for a simple example.

**Note** Shared memory and thread sync haven't been supported yet.

### Builtin Functions

CUJ has some builtin functions, providing basic math operations like `log` and `exp`. See `inc/cuj/builtin/math/basic.h` for a full function list.

Besides, we can use following functions to read special registers need by CUDA programming:

```cpp
namespace cuj::builtin::cuda
{
    Value<int> thread_index_x();
    Value<int> thread_index_y();
    Value<int> thread_index_z();
    Value<int> block_index_x();
    Value<int> block_index_y();
    Value<int> block_index_z();
    Value<int> block_dim_x();
    Value<int> block_dim_y();
    Value<int> block_dim_z();
    Dim3 thread_index();
    Dim3 block_index();
    Dim3 block_dim();
}
```

See `./example/vec_sqrt_add` for an example usage.
