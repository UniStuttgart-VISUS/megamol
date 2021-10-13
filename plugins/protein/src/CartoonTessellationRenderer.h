/**
 * MegaMol
 * Copyright (c) 2021, MegaMol Dev Team
 * All rights reserved.
 */

#ifndef MMPROTEINPLUGIN_CARTOONTESSELLATIONRENDERER_H_INCLUDED
#define MMPROTEINPLUGIN_CARTOONTESSELLATIONRENDERER_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include <map>
#include <utility>
#include "glowl/GLSLProgram.hpp"
#include "mmcore/Call.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/param/ParamSlot.h"
#include "mmcore/view/Renderer3DModuleGL.h"
#include "protein_calls/MolecularDataCall.h"

namespace megamol {
namespace protein {

    using namespace megamol::core;
    using namespace megamol::protein_calls;

    /**
     * Renderer for simple sphere glyphs
     */
    class CartoonTessellationRenderer : public megamol::core::view::Renderer3DModuleGL {
    public:
        /**
         * Answer the name of this module.
         *
         * @return The name of this module.
         */
        static const char* ClassName(void) {
            return "CartoonTessellationRenderer";
        }

        /**
         * Answer a human readable description of this module.
         *
         * @return A human readable description of this module.
         */
        static const char* Description(void) {
            return "Offers cartoon renderings for biomolecules (uses Tessellation Shaders).";
        }

        /**
         * Answers whether this module is available on the current system.
         *
         * @return 'true' if the module is available, 'false' otherwise.
         */
        static bool IsAvailable(void) {
            return isExtAvailable("GL_ARB_buffer_storage") && ogl_IsVersionGEQ(4, 3);
        }

        /** Ctor. */
        CartoonTessellationRenderer(void);

        /** Dtor. */
        virtual ~CartoonTessellationRenderer(void);

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
        virtual bool GetExtents(view::CallRender3DGL& call);

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
        virtual bool Render(view::CallRender3DGL& call);

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

        core::param::ParamSlot ambientColorParam;
        core::param::ParamSlot diffuseColorParam;
        core::param::ParamSlot specularColorParam;
        core::param::ParamSlot ambientFactorParam;
        core::param::ParamSlot diffuseFactorParam;
        core::param::ParamSlot specularFactorParam;
        core::param::ParamSlot specularExponentParam;
        core::param::ParamSlot useLambertParam;

        std::vector<std::vector<float>> positionsCa;
        std::vector<std::vector<float>> positionsO;

        std::shared_ptr<glowl::GLSLProgram> lineShader_;
        std::shared_ptr<glowl::GLSLProgram> cartoonShader_;
        std::shared_ptr<glowl::GLSLProgram> lightingShader_;

        std::vector<CAlpha> mainchain;
    };

} /* end namespace protein */
} /* end namespace megamol */

#endif /* MMPROTEINPLUGIN_CARTOONTESSELLATIONRENDERER_H_INCLUDED */
