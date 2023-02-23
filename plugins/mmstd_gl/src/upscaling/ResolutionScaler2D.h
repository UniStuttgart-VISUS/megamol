/**
 * MegaMol
 * Copyright (c) 2022, MegaMol Dev Team
 * All rights reserved.
 */

#pragma once

#include "ResolutionScalerBase.h"
#include "mmstd_gl/renderer/Renderer2DModuleGL.h"

namespace megamol::mmstd_gl {

/**
 * Module to interconnect between two modules that scales the incomming framebuffer from the caller,
 * passes the scaled framebuffer to the callee, re-scales the returned scaled framebuffer from the callee
 * and returns the re-scaled framebuffer to the initial caller.
 */

class ResolutionScaler2D : public ResolutionScalerBase<mmstd_gl::CallRender2DGL> {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static inline const char* ClassName() {
        return "ResolutionScaler2D";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static inline const char* Description() {
        return "Scale module that scales an incoming fbo, calls the rhs renderers with the reduced resolution fbo and "
               "re-scales the fbo to full resolution. The upscale should occur after proper AntiAliasing!";
    }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static inline bool IsAvailable() {
        return true;
    }

    /** ctor */
    ResolutionScaler2D() = default;

    /** dtor */
    ~ResolutionScaler2D() override {
        this->Release();
    };

protected:
    void releaseImpl() override {
        //this->Release();
    }

private:
}; /* end class ResolutionScaler2D */

} // namespace megamol::mmstd_gl
