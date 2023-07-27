/*
 * ParticleWorker.h
 *
 * Copyright (C) 2013 by Universitaet Stuttgart (VISUS).
 * Alle Rechte vorbehalten.
 */

#pragma once

#include <memory>

#include <glad/gl.h>
#include <glowl/glowl.h>

#include "geometry_calls/MultiParticleDataCall.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/param/ParamSlot.h"
#include "mmstd_gl/ModuleGL.h"
#include "vislib/RawStorage.h"
#include "vislib/types.h"

namespace megamol::datatools_gl::misc {

/**
 * Module to filter calls with multiple particle lists by list index
 */
// TODO this module looks quite broken and will probably not work at all for particles having a direction
class ParticleWorker : public mmstd_gl::ModuleGL {
public:
    class VAOUnlocker : public core::AbstractGetDataCall::Unlocker {
    public:
        VAOUnlocker(){};
        ~VAOUnlocker() override{};
        void Unlock() override {
            glBindVertexArray(0);
            glBindBufferARB(GL_SHADER_STORAGE_BUFFER, 0);
        };
    };

    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName() {
        return "ParticleWorker";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description() {
        return "Modify incoming particles";
    }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable() {
        return true;
    }

    /** Ctor. */
    ParticleWorker();

    /** Dtor. */
    ~ParticleWorker() override;

protected:
    /**
     * Implementation of 'Create'.
     *
     * @return 'true' on success, 'false' otherwise.
     */
    bool create() override;

    /**
     * Implementation of 'Release'.
     */
    void release() override;

private:
    /**
     * Callback publishing the gridded data
     *
     * @param call The call requesting the gridded data
     *
     * @return 'true' on success, 'false' on failure
     */
    bool getDataCallback(core::Call& call);

    /**
     * Callback publishing the extend of the data
     *
     * @param call The call requesting the extend of the data
     *
     * @return 'true' on success, 'false' on failure
     */
    bool getExtentCallback(core::Call& call);

    core::CallerSlot inParticlesDataSlot;

    core::CalleeSlot outParticlesDataSlot;

    vislib::Array<GLuint> glVAO;
    vislib::Array<GLuint> glVB;
    vislib::Array<GLuint> glCB;

    GLuint glClusterInfos;
    std::unique_ptr<glowl::GLSLProgram> shaderOnClusterComputation;

    /*
    GLuint glParticleList;
    GLuint glPrefixIn;
    GLuint glPrefixOut;

    vislib_gl::graphics::gl::GLSLComputeShader shaderComputeInitParticleList;
    vislib_gl::graphics::gl::GLSLComputeShader shaderComputeMakeParticleList;
    vislib_gl::graphics::gl::GLSLComputeShader shaderComputeCompactToClusterList;
    vislib_gl::graphics::gl::GLSLComputeShader shaderComputeGrid;
    vislib_gl::graphics::gl::GLSLComputeShader shaderComputeGriddify;
    vislib_gl::graphics::gl::GLSLComputeShader shaderComputePrefixSum;
    */
};

} // namespace megamol::datatools_gl::misc
