#pragma once

#include <cuj/gen/cpp.h>
#include <cuj/gen/llvm.h>
#include <cuj/gen/mcjit.h>
#include <cuj/gen/nvrtc.h>
#include <cuj/gen/ptx.h>

CUJ_NAMESPACE_BEGIN(cuj)

using gen::Options;
using gen::OptimizationLevel;

using gen::CPPCodeGenerator;
using gen::LLVMIRGenerator;
using gen::MCJIT;
using gen::PTXGenerator;

#ifdef CUJ_ENABLE_CUDA
using gen::NVRTC;
#endif

CUJ_NAMESPACE_END(cuj)
