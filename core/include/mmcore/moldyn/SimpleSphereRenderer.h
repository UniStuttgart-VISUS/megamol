/*
 * SimpleSphereRenderer.h
 *
 * Copyright (C) 2009 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_SIMPLESPHERERENDERER_H_INCLUDED
#define MEGAMOLCORE_SIMPLESPHERERENDERER_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */


#include "mmcore/moldyn/AbstractSimpleSphereRenderer.h"
#include "mmcore/moldyn/MultiParticleDataCall.h"
#include "mmcore/CoreInstance.h"
#include "mmcore/view/CallClipPlane.h"
#include "mmcore/view/CallGetTransferFunction.h"
#include "mmcore/view/CallRender3D.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/param/ButtonParam.h"
#include "mmcore/utility/SSBOStreamer.h"

#include "vislib/assert.h"
#include "vislib/graphics/gl/ShaderSource.h"
#include "vislib/graphics/gl/GLSLShader.h"
#include "vislib/graphics/gl/GLSLGeometryShader.h"
#include "vislib/graphics/gl/IncludeAllGL.h"
#include "vislib/math/mathfunctions.h"
#include "vislib/math/ShallowMatrix.h"
#include "vislib/math/Matrix.h"
#include "vislib/Map.h"

#include <map>
#include <tuple>
#include <utility>
#include <cmath>
#include <cinttypes>
#include <chrono>
#include <sstream>
#include <iterator>

//#include "TimeMeasure.h"


namespace megamol {
namespace core {
namespace moldyn {

    using namespace vislib::graphics::gl;

    /**
     * Renderer for simple sphere glyphs.
     */
    class SimpleSphereRenderer : public AbstractSimpleSphereRenderer {
    public:

        /**
         * Answer the name of this module.
         *
         * @return The name of this module.
         */
        static const char *ClassName(void) {
            return "SimpleSphereRenderer";
        }

        /**
         * Answer a human readable description of this module.
         *
         * @return A human readable description of this module.
         */
        static const char *Description(void) {
            return "Renderer for sphere glyphs providing different modes using e.g. a bit of bleeding-edge features or a geometry shader.";
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
                                                                                        /// Necessary for:
            return vislib::graphics::gl::GLSLShader::AreExtensionsAvailable()           // SimpleSphere, Clustered, NGSphere, NGBufferArray, NGSplat, SimpleGeo
                && isExtAvailable("GL_ARB_buffer_storage")                              // NGSphere, NGBufferArray, NGSplat
                && ogl_IsVersionGEQ(4, 4)                                               // NGSphere, NGBufferArray, NGSplat
                && vislib::graphics::gl::GLSLGeometryShader::AreExtensionsAvailable()   // SimpleGeo
                && ogl_IsVersionGEQ(3, 2)                                               // SimpleGeo
                && isExtAvailable("GL_EXT_geometry_shader4")                            // SimpleGeo
                && isExtAvailable("GL_EXT_gpu_shader4")                                 // SimpleGeo
                && isExtAvailable("GL_EXT_bindable_uniform")                            // SimpleGeo
                && isExtAvailable("GL_ARB_shader_objects");                             // SimpleGeo
        }

        /** Ctor. */
        SimpleSphereRenderer(void);

        /** Dtor. */
        virtual ~SimpleSphereRenderer(void);

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
        virtual bool Render(megamol::core::view::CallRender3D& call);

    private:

        /*********************************************************************/
        /* VARIABLES                                                         */
        /*********************************************************************/

        enum RenderMode {
            SIMPLE           = 0,     /// Simple sphere rendering.
            SIMPLE_CLUSTERED = 1,     /// Same as "Simple" - Clustered rendering is not yet implemented in SimpleSphericalParticles.
            SIMPLE_GEO       = 2,     /// Simple sphere rendering using geometry shader.
            NG               = 3,     /// Next generation (NG) sphere rendering using shader storage buffer object.
            NG_SPLAT         = 4,     /// NG sphere rendering using splats.
            NG_BUFFER_ARRAY  = 5,     /// NG sphere rendering using array buffers.
            __MODE_COUNT__   = 6
        };

        typedef std::map <std::tuple<int, int, bool>, std::shared_ptr<GLSLShader> > shaderMap;
        typedef std::map <std::pair<int, int>, std::shared_ptr<GLSLShader> >        shaderMap_splat;

        RenderMode                               renderMode;
        vislib::graphics::gl::GLSLShader         sphereShader;
        vislib::graphics::gl::GLSLGeometryShader sphereGeometryShader;
        vislib::SmartPtr<ShaderSource>           vertShader;
        vislib::SmartPtr<ShaderSource>           fragShader;
        vislib::SmartPtr<ShaderSource>           geoShader;

        GLuint                                   vertArray;
        SimpleSphericalParticles::ColourDataType colType;
        SimpleSphericalParticles::VertexDataType vertType;
        std::shared_ptr<GLSLShader>              newShader;
        shaderMap                                theShaders;
        shaderMap_splat                          theShaders_splat;

        megamol::core::utility::SSBOStreamer     streamer;
        megamol::core::utility::SSBOStreamer     colStreamer;

        std::vector<GLsync>                      fences;
        GLuint                                   theSingleBuffer;
        unsigned int                             currBuf;
        GLsizeiptr                               bufSize;
        int                                      numBuffers;
        void                                    *theSingleMappedMem;
        GLuint                                   singleBufferCreationBits;
        GLuint                                   singleBufferMappingBits;

        //TimeMeasure                            timer;

        /*********************************************************************/
        /* PARAMETERS                                                        */
        /*********************************************************************/

        core::param::ParamSlot renderModeParam;
        core::param::ParamSlot toggleModeParam;

        core::param::ParamSlot radiusScalingParam;

        // NGSplat ////////////////////////////////////////////////////////////
        core::param::ParamSlot alphaScalingParam;
        core::param::ParamSlot attenuateSubpixelParam;

        /*********************************************************************/
        /* FUNCTIONS                                                         */
        /*********************************************************************/

        /**
         * Toggle render mode on button press.
         *
         * @param slot The calling parameter slot.
         *
         * @return True if success, false otherwise.
         */
        bool toggleRenderMode(param::ParamSlot& slot);

        /**
         * Create shaders for given render mode.
         * 
         * @return True if success, false otherwise.
         */
        bool createShaders(void);

        /**
         * Release all OpenGL resources.
         *
         * @return True if success, false otherwise.
         */
        bool resetResources(void);

        /**
         * Render spheres in different render modes.
         *
         * @param cr3d       Pointer to the calling render call.
         * @param mpdc       Pointer to the multi particle data call.
         * @param vp         The current viewport.
         * @param clipDat    The current clip data.
         * @param clipCol    The current clip data.
         * @param scaling    The current scaling factor.
         * @param lp         The current light position.
         * @param MVinv      The inverse model view matrix.
         * @param MVP        The model view projection matrix.
         * @param MVPinv     The inverse model view projection matrix.
         * @param MVPtransp  The tranpose model view projection matrix.
         *
         * @return         True if success, false otherwise.
         */

        // SIMPLE
        bool renderSimple(view::CallRender3D* cr3d, MultiParticleDataCall* mpdc, 
            float vp[4], float clipDat[4], float clipCol[4], float scaling, float lp[4],
            vislib::math::Matrix<GLfloat, 4, vislib::math::COLUMN_MAJOR> &MVinv,
            vislib::math::Matrix<GLfloat, 4, vislib::math::COLUMN_MAJOR> &MVP,
            vislib::math::Matrix<GLfloat, 4, vislib::math::COLUMN_MAJOR> &MVPinv,
            vislib::math::Matrix<GLfloat, 4, vislib::math::COLUMN_MAJOR> &MVPtransp);

        // NG
        bool renderNG(view::CallRender3D* cr3d, MultiParticleDataCall* mpdc, 
            float vp[4], float clipDat[4], float clipCol[4], float scaling, float lp[4],
            vislib::math::Matrix<GLfloat, 4, vislib::math::COLUMN_MAJOR> &MVinv,
            vislib::math::Matrix<GLfloat, 4, vislib::math::COLUMN_MAJOR> &MVP,
            vislib::math::Matrix<GLfloat, 4, vislib::math::COLUMN_MAJOR> &MVPinv,
            vislib::math::Matrix<GLfloat, 4, vislib::math::COLUMN_MAJOR> &MVPtransp);

        // NGSPLAT
        bool renderNGSplat(view::CallRender3D* cr3d, MultiParticleDataCall* mpdc,
            float vp[4], float clipDat[4], float clipCol[4], float scaling, float lp[4],
            vislib::math::Matrix<GLfloat, 4, vislib::math::COLUMN_MAJOR> &MVinv,
            vislib::math::Matrix<GLfloat, 4, vislib::math::COLUMN_MAJOR> &MVP,
            vislib::math::Matrix<GLfloat, 4, vislib::math::COLUMN_MAJOR> &MVPinv,
            vislib::math::Matrix<GLfloat, 4, vislib::math::COLUMN_MAJOR> &MVPtransp);

        // NGBUFFERARRAY
        bool renderNGBufferArray(view::CallRender3D* cr3d, MultiParticleDataCall* mpdc,
            float vp[4], float clipDat[4], float clipCol[4], float scaling, float lp[4],
            vislib::math::Matrix<GLfloat, 4, vislib::math::COLUMN_MAJOR> &MVinv,
            vislib::math::Matrix<GLfloat, 4, vislib::math::COLUMN_MAJOR> &MVP,
            vislib::math::Matrix<GLfloat, 4, vislib::math::COLUMN_MAJOR> &MVPinv,
            vislib::math::Matrix<GLfloat, 4, vislib::math::COLUMN_MAJOR> &MVPtransp);

        // GEO
        bool renderGeo(view::CallRender3D* cr3d, MultiParticleDataCall* mpdc, 
            float vp[4], float clipDat[4], float clipCol[4], float scaling, float lp[4],
            vislib::math::Matrix<GLfloat, 4, vislib::math::COLUMN_MAJOR> &MVinv,
            vislib::math::Matrix<GLfloat, 4, vislib::math::COLUMN_MAJOR> &MVP,
            vislib::math::Matrix<GLfloat, 4, vislib::math::COLUMN_MAJOR> &MVPinv,
            vislib::math::Matrix<GLfloat, 4, vislib::math::COLUMN_MAJOR> &MVPtransp);

        template <typename T>
        void setPointers(MultiParticleDataCall::Particles &parts, T &shader,
            GLuint vertBuf, const void *vertPtr, GLuint vertAttribLoc,
            GLuint colBuf,  const void *colPtr,  GLuint colAttribLoc, GLuint colIdxAttribLoc);

        void getBytesAndStride(MultiParticleDataCall::Particles &parts, unsigned int &colBytes, unsigned int &vertBytes,
            unsigned int &colStride, unsigned int &vertStride, bool &interleaved);

        bool makeColorString(MultiParticleDataCall::Particles &parts, std::string &code, std::string &declaration, bool interleaved);

        bool makeVertexString(MultiParticleDataCall::Particles &parts, std::string &code, std::string &declaration, bool interleaved);

        std::shared_ptr<GLSLShader> makeShader(vislib::SmartPtr<ShaderSource> vert, vislib::SmartPtr<ShaderSource> frag);

        std::shared_ptr<GLSLShader> generateShader(MultiParticleDataCall::Particles &parts);

        void lockSingle(GLsync& syncObj);

        void waitSingle(GLsync& syncObj);
    };

} /* end namespace moldyn */
} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_SIMPLESPHERERENDERER_H_INCLUDED */
