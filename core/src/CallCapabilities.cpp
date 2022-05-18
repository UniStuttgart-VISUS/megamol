#include "mmcore/CallCapabilities.h"

using namespace megamol::core;

void CallCapabilities::RequireOpenGL() {
    cap_bits |= REQUIRES_OPENGL;
}

void CallCapabilities::RequireCUDA() {
    cap_bits |= REQUIRES_CUDA;
}

void CallCapabilities::RequireOpenCL() {
    cap_bits |= REQUIRES_OPENCL;
}

void CallCapabilities::RequireOptiX() {
    cap_bits |= REQUIRES_OPTIX;
}

void CallCapabilities::RequireOSPRay() {
    cap_bits |= REQUIRES_OSPRAY;
}

void CallCapabilities::RequireVulkan() {
    cap_bits |= REQUIRES_VULKAN;
}

bool CallCapabilities::OpenGLRequired() const {
    return (cap_bits & REQUIRES_OPENGL) > 0;
}

bool CallCapabilities::CUDARequired() const {
    return (cap_bits & REQUIRES_CUDA) > 0;
}

bool CallCapabilities::OpenCLRequired() const {
    return (cap_bits & REQUIRES_OPENCL) > 0;
}

bool CallCapabilities::OptiXRequired() const {
    return (cap_bits & REQUIRES_OPTIX) > 0;
}

bool CallCapabilities::OSPRayRequired() const {
    return (cap_bits & REQUIRES_OSPRAY) > 0;
}

bool CallCapabilities::VulkanRequired() const {
    return (cap_bits & REQUIRES_VULKAN) > 0;
}
