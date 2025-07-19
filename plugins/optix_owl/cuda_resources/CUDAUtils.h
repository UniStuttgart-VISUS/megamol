#pragma once

#ifndef __CUDACC__
#define CU_CALLABLE
#else
#define CU_CALLABLE __host__ __device__
#endif
