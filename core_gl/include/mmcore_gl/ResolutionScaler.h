/**
 * MegaMol
 * Copyright (c) 2022, MegaMol Dev Team
 * All rights reserved.
 */

#pragma once

#include "glowl/glowl.h"
#include "glowl/BufferObject.hpp"
#include "glowl/Sampler.hpp"

#include "mmcore/Call.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/param/ParamSlot.h"
#include "mmcore_gl/view/Renderer3DModuleGL.h"

#include "glm/glm.hpp"

#include "mmcore_gl/3rd/au1_ah1_af1.h"


namespace megamol {
namespace core_gl {


/**
 * Module to interconnect between two modules that scales the incomming framebuffer from the caller,
 * passes the scaled framebuffer to the callee, re-scales the returned scaled framebuffer from the callee
 * and returns the re-scaled framebuffer to the initial caller.
 */
class ResolutionScaler : public core::view::RendererModule<view::CallRender3DGL, ModuleGL> {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName(void) {
        return "ResolutionScaler";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description(void) {
        return "Scale module that scales and re-scales a framebuffer. The upscale should occur after proper AntiAliasing!";
    }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable(void) {
        return true;
    }

    /** ctor */
    ResolutionScaler(void);

    /** dtor */
    virtual ~ResolutionScaler(void);

protected:
    /**
     * Implementation of 'Create'.
     *
     * @return 'true' on success, 'false' otherwise.
     */
    virtual bool create(void);

    /**
     * Implementation of 'Release'.
     */
    virtual void release(void);

    /**
     * The get extents callback. The module should set the members of
     * 'call' to tell the caller the extents of its data (bounding boxes
     * and times).
     *
     * @param call The calling call.
     *
     * @return The return value of the function.
     */
    bool GetExtents(core_gl::view::CallRender3DGL& call);

    /**
     * The render callback.
     *
     * @param call The calling call.
     *
     * @return The return value of the function.
     */
    bool Render(core_gl::view::CallRender3DGL& call);

private:
    struct FSRConstants {
        glm::uvec4 const0;
        glm::uvec4 const1;
        glm::uvec4 const2;
        glm::uvec4 const3;
        glm::uvec4 Sample;
    };

    // AMD FUNCTION
    inline unsigned int AU1_AF1(float a) {
        union {
            float f;
            unsigned int u;
        } bits;
        bits.f = a;
        return bits.u;
    }

    // AMD FUNCTION
    inline unsigned int AU1_AH2_AF2(float* a) {
        return AU1_AH1_AF1(a[0]) + (AU1_AH1_AF1(a[1]) << 16);
    }
    
    // AMD FUNCTION
    void calcConstants(
        glm::uvec4& con0, glm::uvec4& con1,
        glm::uvec4& con2, glm::uvec4& con3,
        float inputSizeX, float inputSizeY,     // rendered image resolution
        float outputSizeX, float outputSizeY);  // display resolution after being upscaled

    /*
    * AMD FUNCTION
    * Sharpness:
    * The scale is {0.0 := maximum, to N>0, where N is the number of stops (halving) of the reduction of sharpness}.
    */
    void FsrRcasCon(glm::uvec4& con, float sharpness);

    inline void resetGLStates() {
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, 0);
        glBindSampler(0, 0);
        glUseProgram(0);
    }

    std::shared_ptr<glowl::FramebufferObject> scaled_fbo_;

    std::shared_ptr<glowl::GLSLProgram> naive_downsample_prgm_;
    std::shared_ptr<glowl::GLSLProgram> naive_upsample_prgm_;
    std::shared_ptr<glowl::GLSLProgram> fsr_downsample_prgm_;
    std::shared_ptr<glowl::GLSLProgram> fsr_bilinear_upsample_prgm_;
    std::shared_ptr<glowl::GLSLProgram> fsr_easu_upsample_prgm_;
    std::shared_ptr<glowl::GLSLProgram> fsr_rcas_upsample_prgm_;

    std::unique_ptr<glowl::BufferObject> fsr_consts_ssbo_;

    // intermediary texture for fsr easu + rcas pass
    glowl::TextureLayout inter_tl_;
    std::unique_ptr<glowl::Texture2D> intermediary_rcas_tx2D_;

    std::unique_ptr<glowl::Sampler> fsr_input_sampler_;

    core::param::ParamSlot scale_mode_;
    core::param::ParamSlot rcas_sharpness_attenuation_;


}; /* end class StubModule */

} /* end namespace core */
} /* end namespace megamol */
