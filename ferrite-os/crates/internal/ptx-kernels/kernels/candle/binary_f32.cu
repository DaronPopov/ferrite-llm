#include "binary_op_macros.cuh"
#include<stdint.h>

// Generate F32 binary operations only
BINARY_OP(float, badd_f32, x + y)
BINARY_OP(float, bdiv_f32, x / y)
BINARY_OP(float, bmul_f32, x * y)
BINARY_OP(float, bsub_f32, x - y)
BINARY_OP(float, bminimum_f32, ming(x, y))
BINARY_OP(float, bmaximum_f32, maxg(x, y))
