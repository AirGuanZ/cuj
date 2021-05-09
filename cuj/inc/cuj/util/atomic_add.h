#pragma once

#include <cuj/common.h>

CUJ_NAMESPACE_BEGIN(cuj)

float atomic_add_float(float *dst, float val);

double atomic_add_double(double *dst, double val);

CUJ_NAMESPACE_END(cuj)
