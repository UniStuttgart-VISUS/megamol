/*
 * GPUAffinity.cpp
 *
 * Copyright (C) 2021 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#include "mmcore/view/GPUAffinity.h"

namespace megamol::core::view {
const GPUAffinity::GpuHandleType GPUAffinity::NO_GPU_AFFINITY = nullptr;


GPUAffinity::GPUAffinity() : gpuAffinity(NO_GPU_AFFINITY) {
}

} // end namespace
