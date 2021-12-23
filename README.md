# Cuj

Runtime program generator embedded in C++

## Building

### Requirements

* LLVM 11.1.0 and (optional) CUDA 11.5 (other versions may work but haven't been tested)
* A C++20-compatible compiler

### Building with CMake

```powershell
git clone https://github.com/AirGuanZ/cuj.git
cd cuj
mkdir build
cd build
cmake -DLLVM_DIR="llvm_cmake_config_dir" ..
```

To add CUJ into a CMake project, simply use `ADD_SUBDIRECTORY` and link against `cuj`.