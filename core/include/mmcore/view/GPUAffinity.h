/*
 * GPUAffinity.h
 *
 * Copyright (C) 2021 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#pragma once

namespace megamol::core::view {

class GPUAffinity {

public:
    /** Defines the type of a GPU handle specifying the GPU affinity. */
    typedef void* GpuHandleType;

    /**
     * Get the GPU affinity handle and convert it to its native type in
     * one step.
     *
     * This value is only meaningful, if IsGpuAffinity() is true.
     *
     * You must ensure that the handle type you request matches the GPU in
     * the system.
     *
     * @return The GPU affinity handle.
     */
    template<class T>
    T GpuAffinity(void) const {
        static_assert(sizeof(T) == sizeof(this->gpuAffinity), "The size of "
                                                              "the GPU handle is unexpected. You are probably doing "
                                                              "something very nasty.");
        return reinterpret_cast<T>(this->gpuAffinity);
    }

    /**
     * Answer whether GPU affinity was requested for the rendering this view.
     *
     * @return true in case GPU affinity was requested, false otherwise.
     */
    inline bool IsGpuAffinity(void) const {
        return (this->gpuAffinity != NO_GPU_AFFINITY);
    }

    /**
     * Sets the GPU that the renderer should use for the following frame.
     *
     * This parameter is set by the core and derived from the
     * mmcRenderViewContext. DO NOT USE THIS UNLESS YOU KNOW WHAT YOU ARE
     * DOING!
     *
     * @param gpuAffinity The handle for the GPU the renderer should use;
     *                    NO_GPU_AFFINITY in case affinity does not matter.
     */
    inline void SetGpuAffinity(const GpuHandleType gpuAffinity) {
        this->gpuAffinity = gpuAffinity;
    }
    
    /** Constant value for specifying no GPU affinity is requested. */
    static const GpuHandleType NO_GPU_AFFINITY;



    /** Some kind of GPU handle if GPU affinity is requested. */
    GpuHandleType gpuAffinity;

            /**
     * Assignment operator
     *
     * @param rhs The right hand side operand
     *
     * @return A reference to this
     */
    GPUAffinity& operator=(const GPUAffinity& rhs) {
        this->gpuAffinity = rhs.gpuAffinity;
        return *this;
    }


    /** Dtor. */
    virtual ~GPUAffinity(void) = default;

    
    /** Ctor. */
    GPUAffinity(void);
};



}
