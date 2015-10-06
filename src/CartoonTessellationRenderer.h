/*
 * CartoonTessellationRenderer.h
 *
 * Copyright (C) 2014 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#ifndef MMPROTEINPLUGIN_CARTOONTESSELLATIONRENDERER_H_INCLUDED
#define MMPROTEINPLUGIN_CARTOONTESSELLATIONRENDERER_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "mmcore/view/Renderer3DModule.h"
#include "vislib/graphics/gl/GLSLShader.h"
#include "vislib/graphics/gl/ShaderSource.h"
#include "vislib/graphics/gl/IncludeAllGL.h"
#include "mmcore/Call.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/param/ParamSlot.h"
#include "mmcore/moldyn/MolecularDataCall.h"
#include <map>
#include <utility>

namespace megamol {
namespace protein {

	using namespace megamol::core;
	using namespace megamol::core::moldyn;
	using namespace vislib::graphics::gl;

    /**
     * Renderer for simple sphere glyphs
     */
    class CartoonTessellationRenderer : public megamol::core::view::Renderer3DModule {
    public:

        /**
         * Answer the name of this module.
         *
         * @return The name of this module.
         */
        static const char *ClassName(void) {
            return "CartoonTessellationRenderer";
        }

        /**
         * Answer a human readable description of this module.
         *
         * @return A human readable description of this module.
         */
        static const char *Description(void) {
            return "Offers cartoon renderings for biomolecules (uses Tessellation Shaders).";
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
        * The get capabilities callback. The module should set the members
        * of 'call' to tell the caller its capabilities.
        *
        * @param call The calling call.
        *
        * @return The return value of the function.
        */
        virtual bool GetCapabilities(Call& call);

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
        * TODO: Document
        */
        MolecularDataCall *getData(unsigned int t, float& outScaling);

        /**
         * The render callback.
         *
         * @param call The calling call.
         *
         * @return The return value of the function.
         */
        virtual bool Render(Call& call);

    private:

        /** The call for data */
        CallerSlot getDataSlot;

        void setPointers(MolecularDataCall &mol, GLuint vertBuf, const void *vertPtr, GLuint colBuf, const void *colPtr);
		std::shared_ptr<GLSLShader> generateShader(MolecularDataCall &mol);
		std::shared_ptr<GLSLShader> makeShader(vislib::SmartPtr<ShaderSource> vert, vislib::SmartPtr<ShaderSource> frag);
		bool makeColorString(MolecularDataCall &mol, std::string &code, std::string &declaration);
		bool makeVertexString(MolecularDataCall &mol, std::string &code, std::string &declaration);
		void getBytesAndStride(MolecularDataCall &mol, unsigned int &colBytes, unsigned int &vertBytes,
			unsigned int &colStride, unsigned int &vertStride);

        void queueSignal(GLsync& syncObj);
		void waitSignal(GLsync& syncObj);

        /** The sphere shader */
        //vislib::graphics::gl::GLSLShader sphereShader;

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
		typedef std::map <std::pair<int, int>, std::shared_ptr<GLSLShader>> shaderMap;
		std::shared_ptr<GLSLShader> newShader;
		shaderMap theShaders;
		vislib::SmartPtr<ShaderSource> vert, frag;
        core::param::ParamSlot scalingParam;

        vislib::Array<vislib::Array<float> > positions;

        /** A simple black-to-white transfer function texture as fallback */
        unsigned int greyTF;
    };

} /* end namespace protein */
} /* end namespace megamol */

#endif /* MMPROTEINPLUGIN_CARTOONTESSELLATIONRENDERER_H_INCLUDED */
