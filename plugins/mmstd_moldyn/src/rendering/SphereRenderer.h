/*
 * SphereRenderer.h
 *
 * Copyright (C) 2009 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOL_MOLDYN_SPHERERENDERER_H_INCLUDED
#define MEGAMOL_MOLDYN_SPHERERENDERER_H_INCLUDED


#include "mmcore/Call.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/param/ParamSlot.h"
#include "mmcore/CoreInstance.h"
#include "mmcore/UniFlagStorage.h"
#include "mmcore/UniFlagCalls.h"
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
#include "mmcore/view/CallRender3DGL.h"
#include "mmcore/view/Renderer3DModuleGL.h"
#include "mmcore/utility/Picking_gl.h"

#include "misc/MDAOVolumeGenerator.h"

#include "vislib/types.h"
#include "vislib/assert.h"
#include "vislib/graphics/gl/ShaderSource.h"
#include "vislib/graphics/gl/GLSLShader.h"
#include "vislib/graphics/gl/GLSLGeometryShader.h"
#include "vislib/graphics/gl/IncludeAllGL.h"
#include "vislib/math/mathfunctions.h"
#include "vislib/math/Cuboid.h"

#define _USE_MATH_DEFINES
#include <cmath>
#include <map>
#include <tuple>
#include <utility>
#include <cmath>
#include <cinttypes>
#include <chrono>
#include <sstream>
#include <iterator>
#include <algorithm>
#include <vector>
#include <string>
#include <sstream>
#include <fstream>


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
    class SphereRenderer : public megamol::core::view::Renderer3DModuleGL {
    public:
       
        /**
         * Answer the name of this module.
         *
         * @return The name of this module.
         */
        static const char *ClassName() {
            return "SphereRenderer";
        }

        /**
         * Answer a human readable description of this module.
         *
         * @return A human readable description of this module.
         */
        static const char *Description() {
            return "Renderer for sphere glyphs providing different modes using e.g. a bit of bleeding-edge features or a geometry shader.";
        }

        /**
         * Answers whether this module is available on the current system.
         *
         * @return 'true' if the module is available, 'false' otherwise.
         */
        static bool IsAvailable() {

#ifdef _WIN32
#if defined(DEBUG) || defined(_DEBUG)
            HDC dc = ::wglGetCurrentDC();
            HGLRC rc = ::wglGetCurrentContext();
            if (dc == nullptr) {
                megamol::core::utility::log::Log::DefaultLog.WriteMsg(megamol::core::utility::log::Log::LEVEL_ERROR, 
                    "[SphereRenderer] There is no OpenGL rendering context available.");
            }
            if (rc == nullptr) {
                megamol::core::utility::log::Log::DefaultLog.WriteMsg(megamol::core::utility::log::Log::LEVEL_ERROR, 
                    "[SphereRenderer] There is no current OpenGL rendering context available from the calling thread.");
            }
            ASSERT(dc != nullptr);
            ASSERT(rc != nullptr);
#endif // DEBUG || _DEBUG
#endif // _WIN32

            bool retval = true;

            // Minimum requirements for all render modes
            if (!GLSLShader::AreExtensionsAvailable()) {
                megamol::core::utility::log::Log::DefaultLog.WriteMsg(megamol::core::utility::log::Log::LEVEL_ERROR, 
                    "[SphereRenderer] No render mode is available. Shader extensions are not available.");
                retval = false;
            }
            // (OpenGL Version and GLSL Version might not correlate, see Mesa 3D on Stampede ...)
            if (ogl_IsVersionGEQ(1, 4) == 0) {
                megamol::core::utility::log::Log::DefaultLog.WriteMsg(megamol::core::utility::log::Log::LEVEL_ERROR, 
                    "[SphereRenderer] No render mode available. OpenGL version 1.4 or greater is required.");
                retval = false;
            }
            std::string glslVerStr((char*)glGetString(GL_SHADING_LANGUAGE_VERSION));
            std::size_t found = glslVerStr.find('.');
            int major = -1;
            int minor = -1;
            if (found != std::string::npos) {
                major = std::atoi(glslVerStr.substr(0, 1).c_str());
                minor = std::atoi(glslVerStr.substr(found+1, 1).c_str());
            }
            else {
                megamol::core::utility::log::Log::DefaultLog.WriteMsg(megamol::core::utility::log::Log::LEVEL_ERROR,
                    "[SphereRenderer] No valid GL_SHADING_LANGUAGE_VERSION string: %s", glslVerStr.c_str());
            }
            megamol::core::utility::log::Log::DefaultLog.WriteMsg(megamol::core::utility::log::Log::LEVEL_INFO,
                "[SphereRenderer] Found GLSL version %d.%d (%s).", major, minor, glslVerStr.c_str());
            if ((major < (int)(SPHERE_MIN_GLSL_MAJOR)) || (major == (int)(SPHERE_MIN_GLSL_MAJOR) && minor < (int)(SPHERE_MIN_GLSL_MINOR))) {
                megamol::core::utility::log::Log::DefaultLog.WriteMsg(megamol::core::utility::log::Log::LEVEL_ERROR, 
                    "[SphereRenderer] No render mode available. OpenGL Shading Language version 1.3 or greater is required.");
                retval = false; 
            }
            if (!isExtAvailable("GL_ARB_explicit_attrib_location")) {
                megamol::core::utility::log::Log::DefaultLog.WriteMsg(megamol::core::utility::log::Log::LEVEL_WARN,
                    "[SphereRenderer] No render mode is available. Extension GL_ARB_explicit_attrib_location is not available.");
                retval = false;
            }
            if (!isExtAvailable("GL_ARB_conservative_depth")) {
                megamol::core::utility::log::Log::DefaultLog.WriteMsg(megamol::core::utility::log::Log::LEVEL_WARN,
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
        bool GetExtents(megamol::core::view::CallRender3DGL& call) override;

        /* Required for picking. */
        bool OnMouseMove(double x, double y) override;
        bool OnMouseButton(megamol::core::view::MouseButton button, megamol::core::view::MouseButtonAction action, megamol::core::view::Modifiers mods) override;

        /** Ctor. */
        SphereRenderer();

        /** Dtor. */
        ~SphereRenderer() override;

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
         * The render callback.
         *
         * @param call The calling call.
         *
         * @return The return value of the function.
         */
        bool Render(megamol::core::view::CallRender3DGL& call) override;

    private:

        /*********************************************************************/
        /* SLOTS                                                             */
        /*********************************************************************/

        megamol::core::CallerSlot getDataSlot;
        megamol::core::CallerSlot getClipPlaneSlot;
        megamol::core::CallerSlot getTFSlot;
        megamol::core::CallerSlot readFlagsSlot;
        megamol::core::CallerSlot writeFlagsSlot;
        megamol::core::CallerSlot getLightsSlot;

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

        bool                                     init_resources;
        RenderMode                               renderMode;
        GLuint                                   greyTF;
        std::array<float, 2>                     range;

        megamol::core::utility::PickingBuffer    picking_buffer;

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
        /* PARAMETERS                                                        */
        /*********************************************************************/

        megamol::core::param::ParamSlot renderModeParam;
        megamol::core::param::ParamSlot radiusScalingParam;
        megamol::core::param::ParamSlot forceTimeSlot;
        megamol::core::param::ParamSlot useLocalBBoxParam;
        megamol::core::param::ParamSlot selectColorParam;
        megamol::core::param::ParamSlot softSelectColorParam;
        megamol::core::param::ParamSlot highlightedColorParam;

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
		megamol::core::param::ParamSlot outlineWidthSlot;

        /*********************************************************************/
        /* FUNCTIONS                                                         */
        /*********************************************************************/

        static std::string getRenderModeString(RenderMode rm);
        static bool isRenderModeAvailable(RenderMode rm, bool silent = false);

        void getGLSLVersion(int &outMajor, int &outMinor) const;

        MultiParticleDataCall *getData(unsigned int t, float& outScaling);
        void getClipData(glm::vec4& out_clipDat, glm::vec4& out_clipCol);

        bool createResources();
        bool resetResources();

        bool renderSimple(view::CallRender3DGL& cr3d, MultiParticleDataCall* mpdc);
        bool renderGeometryShader(view::CallRender3DGL& cr3d, MultiParticleDataCall* mpdc);
        bool renderSSBO(view::CallRender3DGL& cr3d, MultiParticleDataCall* mpdc);
        bool renderSplat(view::CallRender3DGL& cr3d, MultiParticleDataCall* mpdc);
        bool renderBufferArray(view::CallRender3DGL& cr3d, MultiParticleDataCall* mpdc);
        bool renderAmbientOcclusion(view::CallRender3DGL& cr3d, MultiParticleDataCall* mpdc);
		bool renderOutline(view::CallRender3DGL& cr3d, MultiParticleDataCall* mpdc);

        bool enableBufferData(const GLSLShader& shader, const MultiParticleDataCall::Particles &parts,
            GLuint vertBuf, const void *vertPtr, GLuint colBuf,  const void *colPtr, bool createBufferData = false) const;
        bool disableBufferData(const GLSLShader& shader) const;

        bool enableShaderData(GLSLShader& shader, const MultiParticleDataCall::Particles &parts);
        bool disableShaderData();

        bool enableTransferFunctionTexture(GLSLShader& shader);
        bool disableTransferFunctionTexture();

        void checkFlagStorageAvailability(vislib::SmartPtr<ShaderSource::Snippet>& out_flag_snippet);
        void enableFlagStorage(MultiParticleDataCall* mpdc);
        void disableFlagStorage();
        void setFlagStorageUniforms(GLSLShader& shader, unsigned int particle_offset);

        void getBytesAndStride(const MultiParticleDataCall::Particles &parts, unsigned int &outColBytes, unsigned int &outVertBytes,
            unsigned int &outColStride, unsigned int &outVertStride, bool &outInterleaved);

        bool makeColorString(const MultiParticleDataCall::Particles &parts, std::string &outCode, std::string &outDeclaration, bool interleaved);
        bool makeVertexString(const MultiParticleDataCall::Particles &parts, std::string &outCode, std::string &outDeclaration, bool interleaved);
        std::shared_ptr<GLSLShader> makeShader(const std::shared_ptr<ShaderSource>& vert, const std::shared_ptr<ShaderSource>& frag);
        std::shared_ptr<GLSLShader> generateShader(const MultiParticleDataCall::Particles &parts);

#if defined(SPHERE_MIN_OGL_BUFFER_ARRAY) || defined(SPHERE_MIN_OGL_SPLAT)

        void lockSingle(GLsync& outSyncObj);
        void waitSingle(const GLsync& syncObj);

#endif // defined(SPHERE_MIN_OGL_BUFFER_ARRAY) || defined(SPHERE_MIN_OGL_SPLAT)

        // ONLY used for Ambient Occlusion rendering: -------------------------
        bool rebuildGBuffer();
        void rebuildWorkingData(megamol::core::view::CallRender3DGL& cr3d, megamol::core::moldyn::MultiParticleDataCall* mpdc, const GLSLShader& shader);
        void renderDeferredPass(megamol::core::view::CallRender3DGL& cr3d);
        std::string generateDirectionShaderArrayString(const std::vector<glm::vec4>& directions, const std::string& directionsName);
        void generate3ConeDirections(std::vector<glm::vec4>& outDirections, float apex);

    };

} /* end namespace rendering */
} /* end namespace moldyn */
} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOL_MOLDYN_SPHERERENDERER_H_INCLUDED */
