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
#include "mmcore/utility/MDAO2ShaderUtilities.h"
#include "mmcore/utility/MDAO2VolumeGenerator.h"

#include "mmcore/CoreInstance.h"
#include "mmcore/view/CallClipPlane.h"
#include "mmcore/view/CallGetTransferFunction.h"
#include "mmcore/view/CallRender3D.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/BoolParam.h"
#include "mmcore/param/IntParam.h"
#include "mmcore/param/StringParam.h"
#include "mmcore/param/ButtonParam.h"
#include "mmcore/utility/SSBOStreamer.h"
#include "mmcore/utility/SSBOBufferArray.h"

#include "vislib/types.h"
#include "vislib/assert.h"
#include "vislib/graphics/gl/ShaderSource.h"
#include "vislib/graphics/gl/GLSLShader.h"
#include "vislib/graphics/gl/GLSLGeometryShader.h"
#include "vislib/graphics/gl/IncludeAllGL.h"
#include "vislib/graphics/gl/CameraOpenGL.h"
#include "vislib/graphics/CameraParameters.h"
#include "vislib/math/mathfunctions.h"
#include "vislib/math/ShallowMatrix.h"
#include "vislib/math/Vector.h"
#include "vislib/math/Matrix.h"
#include "vislib/math/Cuboid.h"
#include "vislib/Map.h"

#include <map>
#include <tuple>
#include <utility>
#include <cmath>
#include <cinttypes>
#include <chrono>
#include <sstream>
#include <iterator>

#include <GL/glu.h>
#include <algorithm>
#include <vector>
#include <string>
#include <sstream>
#include <deque>
#include <fstream>
#include <signal.h>

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
                && vislib::graphics::gl::GLSLGeometryShader::AreExtensionsAvailable()   // SimpleGeo
                && ogl_IsVersionGEQ(4, 4)                                               // NGSphere, NGBufferArray, NGSplat
                //&& ogl_IsVersionGEQ(2, 2)                                             // SimpleGeo
                //&& ogl_IsVersionGEQ(3, 3)                                             // AmbientOcclusion
                && isExtAvailable("GL_ARB_buffer_storage")                              // NGSphere, NGBufferArray, NGSplat
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
            SIMPLE            = 0,     /// Simple sphere rendering.
            SIMPLE_CLUSTERED  = 1,     /// Same as "Simple" - Clustered rendering is not yet implemented in SimpleSphericalParticles?
            SIMPLE_GEO        = 2,     /// Simple sphere rendering using geometry shader.
            NG                = 3,     /// Next generation (NG) sphere rendering using shader storage buffer object.
            NG_SPLAT          = 4,     /// NG sphere rendering using splats.
            NG_BUFFER_ARRAY   = 5,     /// NG sphere rendering using array buffers.
            AMBIENT_OCCLUSION = 6,     /// Sphere rendering with ambient occlusion
            __MODE_COUNT__    = 7
        };

        typedef std::map <std::tuple<int, int, bool>, std::shared_ptr<GLSLShader> > shaderMap;
        typedef std::map <std::pair<int, int>, std::shared_ptr<GLSLShader> >        shaderMap_splat;

        struct gpuParticleDataType {
            GLuint vertexVBO, colorVBO, vertexArray;
        };

        struct gBufferDataType {
            GLuint color, depth, normals;
            GLuint fbo;
        };

        // Current Render State -----------------------------------------------

        float curViewAttrib[4];
        float curClipDat[4];
        float oldClipDat[4];
        float curClipCol[4];
        float curLightPos[4];
        int   curVpWidth;
        int   curVpHeight;
        int   lastVpWidth;
        int   lastVpHeight;
        vislib::math::Matrix<GLfloat, 4, vislib::math::COLUMN_MAJOR> curMVinv;
        vislib::math::Matrix<GLfloat, 4, vislib::math::COLUMN_MAJOR> curMVP;
        vislib::math::Matrix<GLfloat, 4, vislib::math::COLUMN_MAJOR> curMVPinv;
        vislib::math::Matrix<GLfloat, 4, vislib::math::COLUMN_MAJOR> curMVPtransp;

        // --------------------------------------------------------------------

        RenderMode                               renderMode;

        vislib::graphics::gl::GLSLShader         sphereShader;
        vislib::graphics::gl::GLSLGeometryShader sphereGeometryShader;
        vislib::graphics::gl::GLSLShader         lightingShader;

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
        megamol::core::utility::SSBOBufferArray  bufArray;
        megamol::core::utility::SSBOBufferArray  colBufArray;



        std::vector<GLsync>                      fences;
        GLuint                                   theSingleBuffer;
        unsigned int                             currBuf;
        GLsizeiptr                               bufSize;
        int                                      numBuffers;
        void                                    *theSingleMappedMem;
        GLuint                                   singleBufferCreationBits;
        GLuint                                   singleBufferMappingBits;

        std::vector<gpuParticleDataType>         gpuData;
        gBufferDataType                          gBuffer;
        SIZE_T                                   oldHash;
        unsigned int                             oldFrameID;
        bool                                     stateInvalid;
        vislib::math::Vector<float, 2>           ambConeConstants;
        GLuint                                   tfFallbackHandle;
        core::utility::MDAO2VolumeGenerator     *volGen;

        //TimeMeasure                            timer;

        /*********************************************************************/
        /* PARAMETERS                                                        */
        /*********************************************************************/

        core::param::ParamSlot renderModeParam;
        core::param::ParamSlot toggleModeParam;

        core::param::ParamSlot radiusScalingParam;

        // NGSplat ------------------------------------------------------------

        core::param::ParamSlot alphaScalingParam;
        core::param::ParamSlot attenuateSubpixelParam;
        core::param::ParamSlot useStaticDataParam;

        // Ambient Occlusion --------------------------------------------------

        // Enable or disable lighting
        megamol::core::param::ParamSlot enableLightingSlot;
        // Enable Ambient Occlusion
        megamol::core::param::ParamSlot enableAOSlot;
        megamol::core::param::ParamSlot enableGeometryShader;
        // AO texture size 
        megamol::core::param::ParamSlot aoVolSizeSlot;
        // Cone Apex Angle 
        megamol::core::param::ParamSlot aoConeApexSlot;
        // AO offset from surface
        megamol::core::param::ParamSlot aoOffsetSlot;
        // AO strength
        megamol::core::param::ParamSlot aoStrengthSlot;
        // AO cone length
        megamol::core::param::ParamSlot aoConeLengthSlot;
        // High precision textures slot
        megamol::core::param::ParamSlot useHPTexturesSlot;


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
        bool createResources(void);

        /**
         * Reset all OpenGL resources.
         *
         * @return True if success, false otherwise.
         */
        bool resetResources(void);

        /**
         * Render spheres in different render modes.
         *
         * @param cr3d       Pointer to the current calling render call.
         * @param mpdc       Pointer to the current multi particle data call.
         *
         * @return           True if success, false otherwise.
         */
        bool renderSimple(view::CallRender3D* cr3d, MultiParticleDataCall* mpdc);
        bool renderNG(view::CallRender3D* cr3d, MultiParticleDataCall* mpdc);
        bool renderNGSplat(view::CallRender3D* cr3d, MultiParticleDataCall* mpdc);
        bool renderNGBufferArray(view::CallRender3D* cr3d, MultiParticleDataCall* mpdc);
        bool renderGeo(view::CallRender3D* cr3d, MultiParticleDataCall* mpdc);
        bool renderAmbientOcclusion(view::CallRender3D* cr3d, MultiParticleDataCall* mpdc);

        /**
         * Set pointers to vertex and color buffers and corresponding shader variables.
         *
         * @param parts            ...
         * @param shader           ...
         * @param vertBuf          ...
         * @param vertPtr          ...
         * @param vertAttribLoc    ...
         * @param colBuf           ...
         * @param colPtr           ...
         * @param colAttribLoc     ...
         * @param colIdxAttribLoc  ...
         */
        template <typename T>
        void setPointers(MultiParticleDataCall::Particles &parts, T &shader,
            GLuint vertBuf, const void *vertPtr, GLuint vertAttribLoc,
            GLuint colBuf,  const void *colPtr,  GLuint colAttribLoc, GLuint colIdxAttribLoc);

        /**
         * Get bytes and stride.
         *
         * @param parts        ...
         * @param colBytes     ...
         * @param vertBytes    ...
         * @param colStride    ...
         * @param vertStride   ...
         * @param interleaved  ...
         */
        void getBytesAndStride(MultiParticleDataCall::Particles &parts, unsigned int &colBytes, unsigned int &vertBytes,
            unsigned int &colStride, unsigned int &vertStride, bool &interleaved);

        /**
         * Make NG vertex shader color string.
         *
         * @param parts        ...
         * @param code         ...
         * @param declaration  ...
         * @param interleaved  ...
         *
         */
        bool makeColorString(MultiParticleDataCall::Particles &parts, std::string &code, std::string &declaration, bool interleaved);

        /**
         * Make NG vertex shader position string.
         *
         * @param parts        ...
         * @param code         ...
         * @param declaration  ...
         * @param interleaved  ...
         */
        bool makeVertexString(MultiParticleDataCall::Particles &parts, std::string &code, std::string &declaration, bool interleaved);

        /**
         * Make NG shaders.
         *
         * @param vert  ...
         * @param frag  ...
         */
        std::shared_ptr<GLSLShader> makeShader(vislib::SmartPtr<ShaderSource> vert, vislib::SmartPtr<ShaderSource> frag);

        /**
         * Generate NG shaders.
         *
         * @param parts  ...
         *
         */
        std::shared_ptr<GLSLShader> generateShader(MultiParticleDataCall::Particles &parts);

        /**
         * Lock single.
         *
         * @param syncObj  ...
         */
        void lockSingle(GLsync& syncObj);

        /**
         * Wait single.
         *
         * @param syncObj  ...
         */
        void waitSingle(GLsync& syncObj);

        // Ambient Occlusion --------------------------------------------------

        /**
         * Rebuild the ambient occlusion shaders.
         *
         * @return ...  
         */
        bool rebuildShader(void);

        /**
         * Rebuild the ambient occlusion gBuffer.
         * 
         * @return ...  
         */
        bool rebuildGBuffer(void);

        /**
         * Rebuild working data.
         *
         * @param cr3d  ...
         * @param dataCall    ...
         */
        void rebuildWorkingData(megamol::core::view::CallRender3D* cr3d, megamol::core::moldyn::MultiParticleDataCall* dataCall);

        /**
         * Render particles geometry.
         *
         * @param cr3d  ...
         * @param dataCall    ...
         */
        void renderParticlesGeometry(megamol::core::view::CallRender3D* cr3d, megamol::core::moldyn::MultiParticleDataCall* dataCall);

        /**
         * Render deferred pass.
         *
         * @param cr3d  ...
         */
        void renderDeferredPass(megamol::core::view::CallRender3D* cr3d);

        /**
         * Upload data to GPU.
         *
         * @param gpuData    ...
         * @param particles  ...
         */
        void uploadDataToGPU(const gpuParticleDataType &gpuData, megamol::core::moldyn::MultiParticleDataCall::Particles& particles);

        /**
         * Generate direction shader array string.
         *
         * @param directions      ...
         * @param directionsName  ...
         *
         * @return ...  
         */
        std::string generateDirectionShaderArrayString(const std::vector< vislib::math::Vector< float, int(4) > >& directions, const std::string& directionsName);

        /**
         * Generate 3 cone directions.
         *
         * @param directions  ...
         * @param apex        ...
         */
        void generate3ConeDirections(std::vector< vislib::math::Vector< float, int(4) > >& directions, float apex);

        /**
         * Get transfer function handle.
         *
         * @return ...  ...
         */
        GLuint getTransferFunctionHandle(void);

    };

} /* end namespace moldyn */
} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_SIMPLESPHERERENDERER_H_INCLUDED */
