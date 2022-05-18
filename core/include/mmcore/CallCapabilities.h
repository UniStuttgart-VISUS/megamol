/*
 * Call.h
 *
 * Copyright (C) 2021 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#pragma once
#include <cstdint>

namespace megamol {
namespace core {

class CallCapabilities {
public:
    enum caps {
        REQUIRES_OPENGL = 1 << 0,
        REQUIRES_CUDA = 1 << 1,
        REQUIRES_OPENCL = 1 << 2,
        REQUIRES_OPTIX = 1 << 3,
        REQUIRES_OSPRAY = 1 << 4,
        REQUIRES_VULKAN = 1 << 5
    };

    void RequireOpenGL();
    void RequireCUDA();
    void RequireOpenCL();
    void RequireOptiX();
    void RequireOSPRay();
    void RequireVulkan();

    bool OpenGLRequired() const;
    bool CUDARequired() const;
    bool OpenCLRequired() const;
    bool OptiXRequired() const;
    bool OSPRayRequired() const;
    bool VulkanRequired() const;

private:
    uint64_t cap_bits = 0;
};

} // namespace core
} // namespace megamol
