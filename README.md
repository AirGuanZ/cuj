# CUJ

[TOC]

Run-time program generator embedded in C++

## Requirements

* LLVM 11.1.0 and (optional) CUDA 10 (other versions may work but haven't been tested)
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

### A Quick Example

TODO

### CUJ Context

Any CUJ operation must be done with a context. There is a global context stack
in CUJ, and the context on the top of the stack will be used by CUJ operations. We can use `push_context` and `pop_context` to manipulate this stack.

```cpp
{
    cuj::Context context;
    cuj::push_context(&context);
    // operations like creating functions and generating machine code
    cuj::pop_context();
}
```

We can also use scoped guard provided by CUJ to avoid forgetting `pop_context`. The following two pieces of codes are equivalent to the above.

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

### Define CUJ Functions

The simplest way to define a CUJ function is calling `to_callable` with a callable object describing how to build the function body. For example:

```cpp
{
    cuj::ScopedContext context;

    auto my_add_float = cuj::to_callable<float>(
        "add_float", [](cuj::f32 a, cuj::f32 b) { $return(a + b); });

    // other operations
}
```

The above codes define a CUJ function `add_float`, which returns the sum of two given floating point numbers.

`$float a, $float b` means this CUJ function receives two arguments of `float` type, where `$float` is a CUJ representation of `float`, allowing CUJ to trace what happened to these two arguments.

Note that the functor returns `void`, even if it defines a CUJ function which returns `float`. Instead, the return value type of CUJ function is specified by the template argument of `to_callable`, and the real `return` statement is replaced with CUJ's `$return`.

All variables, operators and statements contained in the defined CUJ function are replaced with their CUJ representations. CUJ constructs the function body by tracing operations on these representations. We will see more examples later.

There are some variants of `to_callable`:

* `to_callable<ReturnType>(Functor)` omits the function name. CUJ will automaticly assign an unique name to it.
* `to_callable<ReturnType>(Type, Functor)` specifies the function type (see `cuj::ir::FunctionType`). This can be used for defining special function type like `device` function in NVIDIA PTX.
* `to_callable<ReturnType>(Name, Type, Functor)` specifies both of the name and type.

We can also define a CUJ function by explicitly calling `Context::begin_function` and `Context::end_function`. For example:

```cpp
{
    cuj::ScopedContext context;
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

### Call CUJ Functions

A CUJ function can not be called within another CUJ function. The calling syntax is very similar to C++ —— just use the return value of `to_callable` or `Context::begin_function` as a functor. For example, to call `my_add_float` in another function:

```cpp
{
    cuj::ScopedContext context;

    auto my_add_float = cuj::to_callable<float>(
        [](cuj::f32 a, cuj::f32 b) { $return(a + b); });

    auto another_func = cuj::to_callable<float>([&]
    {
        cuj::f32 x = 1;
        cuj::f32 z = my_add_float(x, 2.0f);
        $return(z);
    });
    
    // other operations
}
```

### CUJ Variables

Variables in CUJ can be of the following types:

* basic
* pointer
* array
* class

We can use `Value` to convert a C++ type to its corresponding CUJ type. To define a variable in CUJ function, simply use `Value<CPPType> var_name;`. For example:

```cpp
{
    cuj::ScopedContext context;
    
    auto my_func = cuj::to_callable<int>(
        [](cuj::Value<int> x, cuj::Value<int> y)
    {
        cuj::Value<int> x2 = x * x;
        cuj::Value<int> y2 = y * y;
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

For any "left value" in CUJ, use `val.address()` to get a pointer to it. Pointer-type values can be used like in C++. For example:

```cpp
Value<int32_t>  x  = 5;
Value<int32_t*> px = x.address();
*px = 0; // modify x to 0
```

TODO

### Control Flow Statements

### Backends
