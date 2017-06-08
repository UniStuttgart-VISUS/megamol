/*
 * NGSphereRenderer.h
 *
 * Copyright (C) 2014 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_NGSPHEREBUFFERARRAYRENDERER_H_INCLUDED
#define MEGAMOLCORE_NGSPHEREBUFFERARRAYRENDERER_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "mmcore/moldyn/AbstractSimpleSphereRenderer.h"
#include "vislib/graphics/gl/GLSLShader.h"
#include "vislib/graphics/gl/IncludeAllGL.h"

namespace megamol {
namespace stdplugin {
namespace moldyn {
namespace rendering {

	using namespace megamol::core;
	using namespace megamol::core::moldyn;

    /**
     * Renderer for simple sphere glyphs
     */
    class NGSphereBufferArrayRenderer : public megamol::core::moldyn::AbstractSimpleSphereRenderer {
    public:

        /**
         * Answer the name of this module.
         *
         * @return The name of this module.
         */
        static const char *ClassName(void) {
            return "NGSphereBufferArrayRenderer";
        }

        /**
         * Answer a human readable description of this module.
         *
         * @return A human readable description of this module.
         */
        static const char *Description(void) {
            return "Renderer for sphere glyphs with a bit of bleeding-edge features";
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
		NGSphereBufferArrayRenderer(void);

        /** Dtor. */
		virtual ~NGSphereBufferArrayRenderer(void);

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
         * The render callback.
         *
         * @param call The calling call.
         *
         * @return The return value of the function.
         */
        virtual bool Render(Call& call);

    private:

        void setPointers(MultiParticleDataCall::Particles &parts, GLuint vertBuf, const void *vertPtr, GLuint colBuf, const void *colPtr);

		void lockSingle(GLsync& syncObj);
		void waitSingle(GLsync& syncObj);

        /** The sphere shader */
        vislib::graphics::gl::GLSLShader sphereShader;

        GLuint vertArray;
        std::vector<GLsync> fences;
		GLuint theSingleBuffer;
        unsigned int currBuf;
        GLuint colIdxAttribLoc;
        GLsizeiptr bufSize;
		int numBuffers;
		void *theSingleMappedMem;
		GLuint singleBufferCreationBits;
		GLuint singleBufferMappingBits;

    };

} /* end namespace rendering */
} /* end namespace moldyn */
} /* end namespace stdplugin */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_NGSPHEREBUFFERARRAYRENDERER_H_INCLUDED */
