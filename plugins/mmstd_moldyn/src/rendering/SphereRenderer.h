/*
 * SphereRenderer.h
 *
 * Copyright (C) 2009 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOL_MOLDYN_SPHERERENDERER_H_INCLUDED
#define MEGAMOL_MOLDYN_SPHERERENDERER_H_INCLUDED

#include "misc/MDAOVolumeGenerator.h"

#include "mmcore/Call.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/param/ParamSlot.h"
#include "mmcore/CoreInstance.h"
#include "mmcore/FlagStorage_GL.h"
#include "mmcore/FlagCall_GL.h"
#include "mmcore/view/CallClipPlane.h"
#include "mmcore/view/CallGetTransferFunction.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/param/ColorParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/BoolParam.h"
#include "mmcore/param/Vector2fParam.h"
#include "mmcore/param/IntParam.h"
#include "mmcore/param/StringParam.h"
#include "mmcore/param/ButtonParam.h"
#include "mmcore/utility/SSBOStreamer.h"
#include "mmcore/utility/SSBOBufferArray.h"
#include "mmcore/moldyn/MultiParticleDataCall.h"
#include "mmcore/view/CallRender3D_2.h"
#include "mmcore/view/Renderer3DModule_2.h"

#include "vislib/types.h"
#include "vislib/assert.h"
#include "vislib/graphics/gl/ShaderSource.h"
#include "vislib/graphics/gl/GLSLShader.h"
#include "vislib/graphics/gl/GLSLGeometryShader.h"
#include "vislib/graphics/gl/IncludeAllGL.h"
#include "vislib/graphics/gl/CameraOpenGL.h"
#include "vislib/graphics/CameraParameters.h"
#include "vislib/math/mathfunctions.h"
#include "vislib/math/Cuboid.h"

#define _USE_MATH_DEFINES
#include <math.h>

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


// Minimum GLSL version for all render modes
#define SPHERE_MIN_GLSL_MAJOR 1
#define SPHERE_MIN_GLSL_MINOR 3

// Minimum OpenGL version for different render modes
#ifdef GL_VERSION_1_4
#define SPHERE_MIN_OGL_SIMPLE
#define SPHERE_MIN_OGL_SIMPLE_CLUSTERED 
#define SPHERE_MIN_OGL_OUTLINE
#endif // GL_VERSION_1_4

#ifdef GL_VERSION_3_2 
#define SPHERE_MIN_OGL_GEOMETRY_SHADER
#endif // GL_VERSION_3_2

#ifdef GL_VERSION_4_2
#define SPHERE_MIN_OGL_SSBO_STREAM
#define SPHERE_MIN_OGL_AMBIENT_OCCLUSION
#endif // GL_VERSION_4_2

#ifdef GL_VERSION_4_5
#define SPHERE_MIN_OGL_SPLAT
#define SPHERE_MIN_OGL_BUFFER_ARRAY
#endif // GL_VERSION_4_5


namespace megamol {
namespace stdplugin {
namespace moldyn {
namespace rendering {

    using namespace megamol::core;
    using namespace megamol::core::moldyn;
    using namespace vislib::graphics::gl;


    /**
     * Renderer for simple sphere glyphs.
     */
    class SphereRenderer : public megamol::core::view::Renderer3DModule_2 {
    public:
       
        /**
         * Answer the name of this module.
         *
         * @return The name of this module.
         */
        static const char *ClassName(void) {
            return "SphereRenderer";
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
            if (dc == nullptr) {
                vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR, 
                    "[SphereRenderer] There is no OpenGL rendering context available.");
            }
            if (rc == nullptr) {
                vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR, 
                    "[SphereRenderer] There is no current OpenGL rendering context available from the calling thread.");
            }
            ASSERT(dc != nullptr);
            ASSERT(rc != nullptr);
#endif // DEBUG || _DEBUG
#endif // _WIN32

            bool retval = true;

            // Minimum requirements for all render modes
            if (!GLSLShader::AreExtensionsAvailable()) {
                vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR, 
                    "[SphereRenderer] No render mode is available. Shader extensions are not available.");
                retval = false;
            }
            // (OpenGL Version and GLSL Version might not correlate, see Mesa 3D on Stampede ...)
            if (ogl_IsVersionGEQ(1, 4) == 0) {
                vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR, 
                    "[SphereRenderer] No render mode available. OpenGL version 1.4 or greater is required.");
                retval = false;
            }
            std::string glslVerStr((char*)glGetString(GL_SHADING_LANGUAGE_VERSION));
            std::size_t found = glslVerStr.find(".");
            int major = -1;
            int minor = -1;
            if (found != std::string::npos) {
                major = std::atoi(glslVerStr.substr(0, 1).c_str());
                minor = std::atoi(glslVerStr.substr(found+1, 1).c_str());
            }
            else {
                vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
                    "[SphereRenderer] No valid GL_SHADING_LANGUAGE_VERSION string: %s", glslVerStr.c_str());
            }
            vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_INFO,
                "[SphereRenderer] Found GLSL version %d.%d (%s).", major, minor, glslVerStr.c_str());
            if ((major < (int)(SPHERE_MIN_GLSL_MAJOR)) || (major == (int)(SPHERE_MIN_GLSL_MAJOR) && minor < (int)(SPHERE_MIN_GLSL_MINOR))) {
                vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR, 
                    "[SphereRenderer] No render mode available. OpenGL Shading Language version 1.3 or greater is required.");
                retval = false; 
            }
            if (!isExtAvailable("GL_ARB_explicit_attrib_location")) {
                vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_WARN,
                    "[SphereRenderer] No render mode is available. Extension GL_ARB_explicit_attrib_location is not available.");
                retval = false;
            }
            if (!isExtAvailable("GL_ARB_conservative_depth")) {
                vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_WARN,
                    "[SphereRenderer] No render mode is available. Extension GL_ARB_conservative_depth is not available.");
                retval = false;
            }

            return retval;
        }

        /**
         * The get extents callback. The module should set the members of
         * 'call' to tell the caller the extents of its data (bounding boxes
         * and times).
         *
         * @param call The calling call.
         *
         * @return The return value of the function.
         */
        virtual bool GetExtents(megamol::core::view::CallRender3D_2& call);

        /** Ctor. */
        SphereRenderer(void);

        /** Dtor. */
        virtual ~SphereRenderer(void);

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
        virtual bool Render(megamol::core::view::CallRender3D_2& call);

    private:

        /*********************************************************************/
        /* VARIABLES                                                         */
        /*********************************************************************/

        enum RenderMode {              
            SIMPLE            = 0,
            SIMPLE_CLUSTERED  = 1,
            GEOMETRY_SHADER   = 2,
            SSBO_STREAM       = 3,
            BUFFER_ARRAY      = 4,
            SPLAT             = 5,
            AMBIENT_OCCLUSION = 6,
			OUTLINE           = 7
        };

        typedef std::map <std::tuple<int, int, bool>, std::shared_ptr<GLSLShader> > shaderMap;

        struct gpuParticleDataType {
            GLuint vertexVBO, colorVBO, vertexArray;
        };

        struct gBufferDataType {
            GLuint color, depth, normals;
            GLuint fbo;
        };

        // Current Render State -----------------------------------------------

        int                         curVpWidth;
        int                         curVpHeight;
        int                         lastVpWidth;
        int                         lastVpHeight;
        glm::vec4                   curViewAttrib;
        glm::vec4                   curClipDat;
        glm::vec4                   oldClipDat;
        glm::vec4                   curClipCol;
        glm::vec4                   curlightDir;
        glm::vec4                   curCamUp;
        float                       curCamNearClip;
        glm::vec4                   curCamView;
        glm::vec4                   curCamRight;
        glm::vec4                   curCamPos;
        glm::mat4                   curMVinv;
        glm::mat4                   curMVtransp;
        glm::mat4                   curMVP;
        glm::mat4                   curMVPinv;
        glm::mat4                   curMVPtransp;
        vislib::math::Cuboid<float> curClipBox;

        // --------------------------------------------------------------------

        RenderMode                               renderMode;
        GLuint                                   greyTF;
        std::array<float, 2>                     range;

        bool                                     flags_enabled; 
        bool                                     flags_available;

        GLSLShader                               sphereShader;
        GLSLGeometryShader                       sphereGeometryShader;
        GLSLShader                               lightingShader;

        std::shared_ptr<ShaderSource>            vertShader;
        std::shared_ptr<ShaderSource>            fragShader;
        std::shared_ptr<ShaderSource>            geoShader;

        GLuint                                   vertArray;
        SimpleSphericalParticles::ColourDataType colType;
        SimpleSphericalParticles::VertexDataType vertType;
        std::shared_ptr<GLSLShader>              newShader;
        shaderMap                                theShaders;

        GLuint                                   theSingleBuffer;
        unsigned int                             currBuf;
        GLsizeiptr                               bufSize;
        int                                      numBuffers;
        void                                    *theSingleMappedMem;

        std::vector<gpuParticleDataType>         gpuData;
        gBufferDataType                          gBuffer;
        SIZE_T                                   oldHash;
        unsigned int                             oldFrameID;
        bool                                     stateInvalid;
        glm::vec2                                ambConeConstants;
        misc::MDAOVolumeGenerator               *volGen;
        bool                                     triggerRebuildGBuffer;

        //TimeMeasure                            timer;

#if defined(SPHERE_MIN_OGL_BUFFER_ARRAY) || defined(SPHERE_MIN_OGL_SPLAT)
        GLuint                                   singleBufferCreationBits;
        GLuint                                   singleBufferMappingBits;
        std::vector<GLsync>                      fences;
#endif // defined(SPHERE_MIN_OGL_BUFFER_ARRAY) || defined(SPHERE_MIN_OGL_SPLAT)

#ifdef SPHERE_MIN_OGL_SSBO_STREAM
        megamol::core::utility::SSBOStreamer                  streamer;
        megamol::core::utility::SSBOStreamer                  colStreamer;
        std::vector<megamol::core::utility::SSBOBufferArray>  bufArray;
        std::vector<megamol::core::utility::SSBOBufferArray>  colBufArray;
#endif // SPHERE_MIN_OGL_SSBO_STREAM

        /*********************************************************************/
        /* SLOTS                                                             */
        /*********************************************************************/

        megamol::core::CallerSlot getDataSlot;
        megamol::core::CallerSlot getClipPlaneSlot;
        megamol::core::CallerSlot getTFSlot;
        megamol::core::CallerSlot readFlagsSlot;

        /*********************************************************************/
        /* PARAMETERS                                                        */
        /*********************************************************************/

        megamol::core::param::ParamSlot renderModeParam;
        megamol::core::param::ParamSlot radiusScalingParam;
        megamol::core::param::ParamSlot forceTimeSlot;
        megamol::core::param::ParamSlot useLocalBBoxParam;
        megamol::core::param::ParamSlot selectColorParam;
        megamol::core::param::ParamSlot softSelectColorParam;

        // Affects only Splat rendering ---------------------------------------

        core::param::ParamSlot alphaScalingParam;
        core::param::ParamSlot attenuateSubpixelParam;
        core::param::ParamSlot useStaticDataParam;

        // Affects only Ambient Occlusion rendering: --------------------------

        megamol::core::param::ParamSlot enableLightingSlot;
        megamol::core::param::ParamSlot enableGeometryShader;
        megamol::core::param::ParamSlot aoVolSizeSlot;
        megamol::core::param::ParamSlot aoConeApexSlot;
        megamol::core::param::ParamSlot aoOffsetSlot;
        megamol::core::param::ParamSlot aoStrengthSlot;
        megamol::core::param::ParamSlot aoConeLengthSlot;
        megamol::core::param::ParamSlot useHPTexturesSlot;

		// Affects only Outline rendering: --------------------------

		megamol::core::param::ParamSlot outlineSizeSlot;

        /*********************************************************************/
        /* FUNCTIONS                                                         */
        /*********************************************************************/

        /**
         * Return specified render mode as human readable string.
         */
        static std::string getRenderModeString(RenderMode rm);

        /**
         * TODO: Document
         *
         * @param t           ...
         * @param outScaling  ...
         *
         * @return Pointer to MultiParticleDataCall ...
         */
        MultiParticleDataCall *getData(unsigned int t, float& outScaling);

        /**
         * Return clipping information.
         *
         * @param clipDat  Points to four floats ...
         * @param clipCol  Points to four floats ....
         */
        void getClipData(glm::vec4& out_clipDat, glm::vec4& out_clipCol);

        /**
         * Check if specified render mode or all render mode are available.
         *
         * @param rm      ...
         * @param silent  ...
         *
         * @return 'True' on success, 'false' otherwise.
         */
        static bool isRenderModeAvailable(RenderMode rm, bool silent = false);

        /**
         * Check if specified render mode or all render mode are available.
         *
         * @param out_flag_snippet   The vertex shader snippet defining the usage of the flag storage depending on its availability.
         *
         * @return 'True' on success, 'false' otherwise.
         */
        bool isFlagStorageAvailable(vislib::SmartPtr<ShaderSource::Snippet>& out_flag_snippet);

        /**
         * Create shaders for given render mode.
         *
         * @return 'True' on success, 'false' otherwise.
         */
        bool createResources(void);

        /**
         * Reset all OpenGL resources.
         *
         * @return 'True' on success, 'false' otherwise.
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
        bool renderSimple(view::CallRender3D_2& cr3d, MultiParticleDataCall* mpdc);
        bool renderGeometryShader(view::CallRender3D_2& cr3d, MultiParticleDataCall* mpdc);
        bool renderSSBO(view::CallRender3D_2& cr3d, MultiParticleDataCall* mpdc);
        bool renderSplat(view::CallRender3D_2& cr3d, MultiParticleDataCall* mpdc);
        bool renderBufferArray(view::CallRender3D_2& cr3d, MultiParticleDataCall* mpdc);
        bool renderAmbientOcclusion(view::CallRender3D_2& cr3d, MultiParticleDataCall* mpdc);
		bool renderOutline(view::CallRender3D_2& cr3d, MultiParticleDataCall* mpdc);

        /**
         * Set pointers to vertex and color buffers and corresponding shader variables.
         *
         * @param shader           The current shader.
         * @param parts            The current particles of a list.
         * @param vertBuf          ...
         * @param vertPtr          ...
         * @param vertAttribLoc    ...
         * @param colBuf           ...
         * @param colPtr           ...
         * @param colAttribLoc     ...
         * @param colIdxAttribLoc  ...
         *
         * @return 'True' on success, 'false' otherwise.
         */
        bool enableBufferData(const GLSLShader& shader, const MultiParticleDataCall::Particles &parts, 
            GLuint vertBuf, const void *vertPtr, GLuint colBuf,  const void *colPtr, bool createBufferData = false);

        /**
         * Unset pointers to vertex and color buffers.
         *
         * @param shader  The current shader.
         *
         * @return 'True' on success, 'false' otherwise.
         */
        bool disableBufferData(const GLSLShader& shader);

        /**
         * Set pointers to vertex and color buffers and corresponding shader variables.
         *
         * @param shader           The current shader.
         * @param parts            The current particles of a list.
         *
         * @return 'True' on success, 'false' otherwise.
         */
        bool enableShaderData(GLSLShader& shader, const MultiParticleDataCall::Particles &parts);

        /**
         * Unset pointers to vertex and color buffers.
         *
         * @return 'True' on success, 'false' otherwise.
         */
        bool disableShaderData(void);

        /**
         * Enables the transfer function texture.
         *
         * @param shader    The current shader.
         *
         * @return 'True' on success, 'false' otherwise.
         */
        bool enableTransferFunctionTexture(GLSLShader& shader);

        /**
         * Disables the transfer function texture.
         *
         * @return 'True' on success, 'false' otherwise.
         */
        bool disableTransferFunctionTexture(void);

        /**
         * Enable flag storage.
         *
         * @param shader           The current shader.
         * @param parts            The current particles of a list.
         *
         * @return 'True' on success, 'false' otherwise.
         */
        bool enableFlagStorage(const GLSLShader& shader, MultiParticleDataCall* mpdc);

        /**
         * Enable flag storage.
         *
         * @param shader           The current shader.
         *
         * @return 'True' on success, 'false' otherwise.
         */
        bool disableFlagStorage(const GLSLShader& shader);

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
        void getBytesAndStride(const MultiParticleDataCall::Particles &parts, unsigned int &outColBytes, unsigned int &outVertBytes,
            unsigned int &outColStride, unsigned int &outVertStride, bool &outInterleaved);

        /**
         * Make SSBO vertex shader color string.
         *
         * @param parts        ...
         * @param code         ...
         * @param declaration  ...
         * @param interleaved  ...
         *
         * @return 'True' on success, 'false' otherwise.
         */
        bool makeColorString(const MultiParticleDataCall::Particles &parts, std::string &outCode, std::string &outDeclaration, bool interleaved);

        /**
         * Make SSBO vertex shader position string.
         *
         * @param parts        ...
         * @param code         ...
         * @param declaration  ...
         * @param interleaved  ...
         *
         * @return 'True' on success, 'false' otherwise.
         */
        bool makeVertexString(const MultiParticleDataCall::Particles &parts, std::string &outCode, std::string &outDeclaration, bool interleaved);

        /**
         * Make SSBO shaders.
         *
         * @param vert  ...
         * @param frag  ...
         *
         * @return ...
         */
        std::shared_ptr<GLSLShader> makeShader(const std::shared_ptr<ShaderSource> vert, const std::shared_ptr<ShaderSource> frag);

        /**
         * Generate SSBO shaders.
         *
         * @param parts  ...
         *
         * @return ...
         */
        std::shared_ptr<GLSLShader> generateShader(const MultiParticleDataCall::Particles &parts);

        /**
         * Returns GLSL minor and major version.
         *
         * @param major The major version of the currently available GLSL version.
         * @param minor The minor version of the currently available GLSL version.
         */
        void getGLSLVersion(int &outMajor, int &outMinor) const;

#if defined(SPHERE_MIN_OGL_BUFFER_ARRAY) || defined(SPHERE_MIN_OGL_SPLAT)

        /**
         * Lock single.
         *
         * @param syncObj  ...
         */
        void lockSingle(GLsync& outSyncObj);

        /**
         * Wait single.
         *
         * @param syncObj  ...
         */
        void waitSingle(const GLsync& syncObj);

#endif // defined(SPHERE_MIN_OGL_BUFFER_ARRAY) || defined(SPHERE_MIN_OGL_SPLAT)

        // ONLY used for Ambient Occlusion rendering: -------------------------

        /**
         * Rebuild the ambient occlusion gBuffer.
         * 
         * @return ...  
         */
        bool rebuildGBuffer(void);

        /**
         * Rebuild working data.
         *
         * @param cr3d    ...
         * @param mpdc    ...
         * @param shader  ...
         */
        void rebuildWorkingData(megamol::core::view::CallRender3D_2& cr3d, megamol::core::moldyn::MultiParticleDataCall* mpdc, const GLSLShader& shader);

        /**
         * Render deferred pass.
         *
         * @param cr3d  ...
         */
        void renderDeferredPass(megamol::core::view::CallRender3D_2& cr3d);

        /**
         * Generate direction shader array string.
         *
         * @param directions      ...
         * @param directionsName  ...
         *
         * @return ...  
         */
        std::string generateDirectionShaderArrayString(const std::vector<glm::vec4>& directions, const std::string& directionsName);

        /**
         * Generate 3 cone directions.
         *
         * @param directions  ...
         * @param apex        ...
         */
        void generate3ConeDirections(std::vector<glm::vec4>& outDirections, float apex);

    };

} /* end namespace rendering */
} /* end namespace moldyn */
} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOL_MOLDYN_SPHERERENDERER_H_INCLUDED */
