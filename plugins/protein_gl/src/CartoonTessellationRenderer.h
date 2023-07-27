/**
 * MegaMol
 * Copyright (c) 2021, MegaMol Dev Team
 * All rights reserved.
 */

#pragma once

#include <map>
#include <utility>

#include <glowl/GLSLProgram.hpp>

#include "mmcore/Call.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/param/ParamSlot.h"
#include "mmstd_gl/renderer/Renderer3DModuleGL.h"
#include "protein_calls/MolecularDataCall.h"
#include "protein_gl/DeferredRenderingProvider.h"

namespace megamol::protein_gl {

using namespace megamol::core;
using namespace megamol::protein_calls;

/**
 * Renderer for simple sphere glyphs
 */
class CartoonTessellationRenderer : public megamol::mmstd_gl::Renderer3DModuleGL {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName() {
        return "CartoonTessellationRenderer";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description() {
        return "Offers cartoon renderings for biomolecules (uses Tessellation Shaders).";
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
    CartoonTessellationRenderer();

    /** Dtor. */
    ~CartoonTessellationRenderer() override;

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

    /**
     * The get extents callback. The module should set the members of
     * 'call' to tell the caller the extents of its data (bounding boxes
     * and times).
     *
     * @param call The calling call.
     *
     * @return The return value of the function.
     */
    bool GetExtents(mmstd_gl::CallRender3DGL& call) override;

    /**
     * TODO: Document
     */
    MolecularDataCall* getData(unsigned int t, float& outScaling);

    /**
     * The render callback.
     *
     * @param call The calling call.
     *
     * @return The return value of the function.
     */
    bool Render(mmstd_gl::CallRender3DGL& call) override;

private:
    struct CAlpha {
        float pos[4];
        float dir[3];
        int type;
    };

    /** The call for data */
    CallerSlot getDataSlot;

    /** The call for light sources */
    core::CallerSlot getLightsSlot;

    core::CallerSlot getFramebufferSlot;

    void getBytesAndStride(MolecularDataCall& mol, unsigned int& colBytes, unsigned int& vertBytes,
        unsigned int& colStride, unsigned int& vertStride);
    void getBytesAndStrideLines(MolecularDataCall& mol, unsigned int& colBytes, unsigned int& vertBytes,
        unsigned int& colStride, unsigned int& vertStride);

    void queueSignal(GLsync& syncObj);
    void waitSignal(GLsync& syncObj);

    GLuint vertArray;
    std::vector<GLsync> fences;
    GLuint theSingleBuffer;
    unsigned int currBuf;
    GLuint colIdxAttribLoc;
    GLsizeiptr bufSize;
    int numBuffers;
    void* theSingleMappedMem;
    GLuint singleBufferCreationBits;
    GLuint singleBufferMappingBits;
    core::param::ParamSlot scalingParam;
    core::param::ParamSlot lineParam;
    core::param::ParamSlot backboneParam;
    core::param::ParamSlot backboneWidthParam;
    core::param::ParamSlot materialParam;
    core::param::ParamSlot lineDebugParam;
    core::param::ParamSlot colorInterpolationParam;

    core::param::ParamSlot coilColorParam;
    core::param::ParamSlot turnColorParam;
    core::param::ParamSlot helixColorParam;
    core::param::ParamSlot sheetColorParam;
    core::param::ParamSlot lineColorParam;

    std::vector<std::vector<float>> positionsCa;
    std::vector<std::vector<float>> positionsO;

    std::shared_ptr<glowl::GLSLProgram> lineShader_;
    std::shared_ptr<glowl::GLSLProgram> cartoonShader_;

    std::vector<CAlpha> mainchain;

    DeferredRenderingProvider deferredProvider_;
};

} // namespace megamol::protein_gl
