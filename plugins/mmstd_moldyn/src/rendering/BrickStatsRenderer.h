/*
 * BrickStatsRenderer.h
 *
 * Copyright (C) 2016 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#ifndef MMSTD_MOLDYN_BRICKSTATSRENDERER_H_INCLUDED
#define MMSTD_MOLDYN_BRICKSTATSRENDERER_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "vislib/graphics/gl/IncludeAllGL.h"
#include "vislib/graphics/gl/GLSLShader.h"
#include "mmstd_moldyn/BrickStatsCall.h"
#include "mmcore/view/Renderer3DModule.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/param/ParamSlot.h"
#include "vislib/graphics/gl/ShaderSource.h"
//#include "TimeMeasure.h"

namespace megamol {
namespace stdplugin {
namespace moldyn {
namespace rendering {

    using namespace megamol::core;
    using namespace vislib::graphics::gl;

    /**
     * Renderer for simple sphere glyphs
     */
    class BrickStatsRenderer : public megamol::core::view::Renderer3DModule {
    public:

        /**
         * Answer the name of this module.
         *
         * @return The name of this module.
         */
        static const char *ClassName(void) {
            return "BrickStatsRenderer";
        }

        /**
         * Answer a human readable description of this module.
         *
         * @return A human readable description of this module.
         */
        static const char *Description(void) {
            return "Renderer that gives an overview of the distribution of particles in a large dataset";
        }

        /**
         * Answers whether this module is available on the current system.
         *
         * @return 'true' if the module is available, 'false' otherwise.
         */
        static bool IsAvailable(void) {
#ifdef _WIN32
#if defined(DEBUG) || defined(_DEBUG)
            HDC dc = ::wglGetCurrentDC();
            HGLRC rc = ::wglGetCurrentContext();
            ASSERT(dc != NULL);
            ASSERT(rc != NULL);
#endif // DEBUG || _DEBUG
#endif // _WIN32
            return vislib::graphics::gl::GLSLShader::AreExtensionsAvailable()
                && isExtAvailable("GL_ARB_buffer_storage")
                && ogl_IsVersionGEQ(4,4);
        }

        /** Ctor. */
        BrickStatsRenderer(void);

        /** Dtor. */
        virtual ~BrickStatsRenderer(void);

    protected:

        /**
         * Implementation of 'Create'.
         *
         * @return 'true' on success, 'false' otherwise.
         */
        virtual bool create(void);

        /**
        * The get extents callback. The module should set the members of
        * 'call' to tell the caller the extents of its data (bounding boxes
        * and times).
        *
        * @param call The calling call.
        *
        * @return The return value of the function.
        */
        virtual bool GetExtents(Call& call);


        /**
         * Implementation of 'Release'.
         */
        virtual void release(void);
        
        /**
         * The render callback.
         *
         * @param call The calling call.
         *
         * @return The return value of the function.
         */
        virtual bool Render(Call& call);

    private:

        bool assertData(Call& call);

        GLuint statsBuffer;

        vislib::graphics::gl::GLSLShader statsShader, boxesShader;

        bool makeProgram(std::string name, vislib::graphics::gl::GLSLShader& program,
            vislib::graphics::gl::ShaderSource& vert, vislib::graphics::gl::ShaderSource& frag);

        size_t numBricks;

        float scaling;

        CallerSlot statSlot;

        param::ParamSlot numBricksSlot;
        param::ParamSlot showBrickSlot;

        param::ParamSlot showStatistics;
        param::ParamSlot showBoxes;
    };

} /* end namespace rendering */
} /* end namespace moldyn */
} /* end namespace stdplugin */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_NGSPHERERENDERER_H_INCLUDED */
