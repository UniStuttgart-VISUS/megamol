/*
 * SphereRenderer.h
 *
 * Copyright (C) 2009 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#pragma once

#define _USE_MATH_DEFINES
#include <math.h>
#include <signal.h>

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cmath>
#include <deque>
#include <fstream>
#include <iterator>
#include <map>
#include <sstream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

// clang-format off
#include <glowl/glowl.h>
#include <vk_platform.h>
// clang-format on

#include "PerformanceManager.h"
#include "geometry_calls/MultiParticleDataCall.h"
#include "misc/MDAOVolumeGenerator.h"
#include "mmcore/Call.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/param/BoolParam.h"
#include "mmcore/param/ButtonParam.h"
#include "mmcore/param/ColorParam.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/IntParam.h"
#include "mmcore/param/ParamSlot.h"
#include "mmcore/param/StringParam.h"
#include "mmcore/param/Vector2fParam.h"
#include "mmcore_gl/utility/SSBOBufferArray.h"
#include "mmcore_gl/utility/SSBOStreamer.h"
#include "mmcore_gl/utility/ShaderFactory.h"
#include "mmstd/renderer/CallClipPlane.h"
#include "mmstd_gl/flags/FlagCallsGL.h"
#include "mmstd_gl/flags/UniFlagStorage.h"
#include "mmstd_gl/renderer/CallGetTransferFunctionGL.h"
#include "mmstd_gl/renderer/CallRender3DGL.h"
#include "mmstd_gl/renderer/Renderer3DModuleGL.h"
#include "vislib/assert.h"
#include "vislib/math/Cuboid.h"
#include "vislib/math/mathfunctions.h"
#include "vislib/types.h"


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


namespace megamol::moldyn_gl::rendering {

using namespace megamol::core;
using namespace megamol::geocalls;
using namespace vislib_gl::graphics::gl;


/**
 * Renderer for simple sphere glyphs.
 */
class SphereRenderer : public megamol::mmstd_gl::Renderer3DModuleGL {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName() {
        return "SphereRenderer";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description() {
        return "Renderer for sphere glyphs providing different modes using e.g. a bit of bleeding-edge features or a "
               "geometry shader.";
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
            megamol::core::utility::log::Log::DefaultLog.WriteError(
                "[SphereRenderer] There is no OpenGL rendering context available.");
        }
        if (rc == nullptr) {
            megamol::core::utility::log::Log::DefaultLog.WriteError(
                "[SphereRenderer] There is no current OpenGL rendering context available from the calling thread.");
        }
        ASSERT(dc != nullptr);
        ASSERT(rc != nullptr);
#endif // DEBUG || _DEBUG
#endif // _WIN32

        bool retval = true;

        // (OpenGL Version and GLSL Version might not correlate, see Mesa 3D on Stampede ...)

        std::string glsl_ver_str((char*) glGetString(GL_SHADING_LANGUAGE_VERSION));
        std::size_t found = glsl_ver_str.find(".");
        int major = -1;
        int minor = -1;
        if (found != std::string::npos) {
            major = std::atoi(glsl_ver_str.substr(0, 1).c_str());
            minor = std::atoi(glsl_ver_str.substr(found + 1, 1).c_str());
        } else {
            megamol::core::utility::log::Log::DefaultLog.WriteError(
                "[SphereRenderer] No valid GL_SHADING_LANGUAGE_VERSION string: %s", glsl_ver_str.c_str());
        }
        megamol::core::utility::log::Log::DefaultLog.WriteInfo(
            "[SphereRenderer] Found GLSL version %d.%d (%s).", major, minor, glsl_ver_str.c_str());
        if ((major < (int) (SPHERE_MIN_GLSL_MAJOR)) ||
            (major == (int) (SPHERE_MIN_GLSL_MAJOR) && minor < (int) (SPHERE_MIN_GLSL_MINOR))) {
            megamol::core::utility::log::Log::DefaultLog.WriteError(
                "[SphereRenderer] No render mode available. OpenGL "
                "Shading Language version 1.3 or greater is required.");
            retval = false;
        }

        return retval;
    }

#ifdef MEGAMOL_USE_PROFILING
    static void requested_lifetime_resources(frontend_resources::ResourceRequest& req) {
        ModuleGL::requested_lifetime_resources(req);
        req.require<frontend_resources::PerformanceManager>();
    }
#endif

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
    bool Render(megamol::mmstd_gl::CallRender3DGL& call) override;

    /**
     * The get extents callback. The module should set the members of
     * 'call' to tell the caller the extents of its data (bounding boxes
     * and times).
     *
     * @param call The calling call.
     *
     * @return The return value of the function.
     */
    bool GetExtents(megamol::mmstd_gl::CallRender3DGL& call) override;

private:
    /*********************************************************************/
    /* VARIABLES                                                         */
    /*********************************************************************/

    enum RenderMode {
        SIMPLE = 0,
        SIMPLE_CLUSTERED = 1,
        GEOMETRY_SHADER = 2,
        SSBO_STREAM = 3,
        BUFFER_ARRAY = 4,
        SPLAT = 5,
        AMBIENT_OCCLUSION = 6,
        OUTLINE = 7
    };

    enum ShadingMode { FORWARD = 0, DEFERRED = 1 };

    typedef std::map<std::tuple<int, int, bool>, std::shared_ptr<glowl::GLSLProgram>> shader_map;

    struct gpuParticleDataType {
        GLuint vertex_vbo, color_vbo, vertex_array;
    };

    struct gBufferDataType {
        GLuint color, depth, normals;
        GLuint fbo;
    };

    // Current Render State -----------------------------------------------

    int cur_vp_width_;
    int cur_vp_height_;
    int last_vp_width_;
    int last_vp_height_;
    glm::vec4 cur_view_attrib_;
    glm::vec4 cur_clip_dat_;
    glm::vec4 old_clip_dat_;
    glm::vec4 cur_clip_col_;
    glm::vec4 cur_light_dir_;
    glm::vec4 cur_cam_up_;
    glm::vec4 cur_cam_view_;
    glm::vec4 cur_cam_right_;
    glm::vec4 cur_cam_pos_;
    glm::mat4 cur_mv_inv_;
    glm::mat4 cur_mv_transp_;
    glm::mat4 cur_mvp_;
    glm::mat4 cur_mvp_inv_;
    glm::mat4 cur_mvp_transp_;
    vislib::math::Cuboid<float> cur_clip_box_;

    // --------------------------------------------------------------------
    std::unique_ptr<msf::ShaderFactoryOptionsOpenGL> shader_options_flags_;

    bool init_resources_;
    RenderMode render_mode_;
    ShadingMode shading_mode_;
    GLuint grey_tf_;
    std::array<float, 2> range_;

    bool flags_enabled_;
    bool flags_available_;

    std::shared_ptr<glowl::GLSLProgram> sphere_prgm_;
    std::shared_ptr<glowl::GLSLProgram> sphere_geometry_prgm_;
    std::shared_ptr<glowl::GLSLProgram> lighting_prgm_;

    std::unique_ptr<glowl::BufferObject> ao_dir_ubo_;

    GLuint vert_array_;
    SimpleSphericalParticles::ColourDataType col_type_;
    SimpleSphericalParticles::VertexDataType vert_type_;
    std::shared_ptr<glowl::GLSLProgram> new_shader_;
    shader_map the_shaders_;

    GLuint the_single_buffer_;
    unsigned int curr_buf_;
    GLsizeiptr buf_size_;
    int num_buffers_;
    void* the_single_mapped_mem_;

    std::vector<gpuParticleDataType> gpu_data_;
    gBufferDataType g_buffer_;
    SIZE_T old_hash_;
    unsigned int old_frame_id_;
    bool state_invalid_;
    glm::vec2 amb_cone_constants_;
    misc::MDAOVolumeGenerator* vol_gen_;
    bool trigger_rebuild_g_buffer_;

#ifdef MEGAMOL_USE_PROFILING
    frontend_resources::PerformanceManager::handle_vector timers_;
    frontend_resources::PerformanceManager* perf_manager_ = nullptr;
#endif

#if defined(SPHERE_MIN_OGL_BUFFER_ARRAY) || defined(SPHERE_MIN_OGL_SPLAT)
    GLuint single_buffer_creation_bits_;
    GLuint single_buffer_mapping_bits_;
    std::vector<GLsync> fences_;
#endif // defined(SPHERE_MIN_OGL_BUFFER_ARRAY) || defined(SPHERE_MIN_OGL_SPLAT)

#ifdef SPHERE_MIN_OGL_SSBO_STREAM
    megamol::core::utility::SSBOStreamer streamer_;
    megamol::core::utility::SSBOStreamer col_streamer_;
    std::vector<megamol::core::utility::SSBOBufferArray> buf_array_;
    std::vector<megamol::core::utility::SSBOBufferArray> col_buf_array_;
#endif // SPHERE_MIN_OGL_SSBO_STREAM

    /*********************************************************************/
    /* SLOTS                                                             */
    /*********************************************************************/

    megamol::core::CallerSlot get_data_slot_;
    megamol::core::CallerSlot get_clip_plane_slot_;
    megamol::core::CallerSlot get_tf_slot_;
    megamol::core::CallerSlot read_flags_slot_;
    megamol::core::CallerSlot get_lights_slot_;

    /*********************************************************************/
    /* PARAMETERS                                                        */
    /*********************************************************************/

    megamol::core::param::ParamSlot render_mode_param_;
    megamol::core::param::ParamSlot shading_mode_param_;
    megamol::core::param::ParamSlot radius_scaling_param_;
    megamol::core::param::ParamSlot force_time_slot_;
    megamol::core::param::ParamSlot use_local_bbox_param_;
    megamol::core::param::ParamSlot select_color_param_;
    megamol::core::param::ParamSlot soft_select_color_param_;

    // Affects only Splat rendering ---------------------------------------

    core::param::ParamSlot alpha_scaling_param_;
    core::param::ParamSlot attenuate_subpixel_param_;
    core::param::ParamSlot use_static_data_param_;

    // Affects only Ambient Occlusion rendering: --------------------------

    megamol::core::param::ParamSlot enable_lighting_slot_;
    megamol::core::param::ParamSlot enable_geometry_shader_;
    megamol::core::param::ParamSlot ao_vol_size_slot_;
    megamol::core::param::ParamSlot ao_cone_apex_slot_;
    megamol::core::param::ParamSlot ao_offset_slot_;
    megamol::core::param::ParamSlot ao_strength_slot_;
    megamol::core::param::ParamSlot ao_cone_length_slot_;
    megamol::core::param::ParamSlot use_hp_textures_slot_;

    // Affects only Outline rendering: --------------------------

    megamol::core::param::ParamSlot outline_width_slot_;

    /*********************************************************************/
    /* FUNCTIONS                                                         */
    /*********************************************************************/

    /**
     * Return specified render mode as human readable string.
     */
    static std::string getRenderModeString(RenderMode rm);

    /**
     * Return specified shading mode as human readable string.
     */
    static std::string getShadingModeString(ShadingMode sm);

    /**
     * TODO: Document
     *
     * @param t           ...
     * @param outScaling  ...
     *
     * @return Pointer to MultiParticleDataCall ...
     */
    MultiParticleDataCall* getData(unsigned int t, float& out_scaling);

    /**
     * Return clipping information.
     *
     * @param clipDat  Points to four floats ...
     * @param clipCol  Points to four floats ....
     */
    void getClipData(glm::vec4& out_clip_dat, glm::vec4& out_clip_col);

    /**
     * Check if specified render mode or all render mode are available.
     *
     * @param rm      ...
     * @param silent  ...
     *
     * @return 'True' on success, 'false' otherwise.
     */
    bool isRenderModeAvailable(RenderMode rm, bool silent = false);

    /**
     * Check if specified render mode or all render mode are available.
     *
     * @return 'True' on success, 'false' otherwise.
     */
    bool isFlagStorageAvailable();

    /**
     * Create shaders for given render mode.
     *
     * @return 'True' on success, 'false' otherwise.
     */
    bool createResources();

    /**
     * Reset all OpenGL resources.
     *
     * @return 'True' on success, 'false' otherwise.
     */
    bool resetOpenGLResources();

    /**
     * Reset the state of all conditional parameters.
     */
    void resetConditionalParameters();

    /**
     * Render spheres in different render modes.
     *
     * @param cr3d       Pointer to the current calling render call.
     * @param mpdc       Pointer to the current multi particle data call.
     *
     * @return           True if success, false otherwise.
     */
    bool renderSimple(mmstd_gl::CallRender3DGL& cr3d, MultiParticleDataCall* mpdc);
    bool renderGeometryShader(mmstd_gl::CallRender3DGL& cr3d, MultiParticleDataCall* mpdc);
    bool renderSSBO(mmstd_gl::CallRender3DGL& cr3d, MultiParticleDataCall* mpdc);
    bool renderSplat(mmstd_gl::CallRender3DGL& cr3d, MultiParticleDataCall* mpdc);
    bool renderBufferArray(mmstd_gl::CallRender3DGL& cr3d, MultiParticleDataCall* mpdc);
    bool renderAmbientOcclusion(mmstd_gl::CallRender3DGL& cr3d, MultiParticleDataCall* mpdc);
    bool renderOutline(mmstd_gl::CallRender3DGL& cr3d, MultiParticleDataCall* mpdc);

    /**
     * Set pointers to vertex and color buffers and corresponding shader variables.
     *
     * @param prgm             The current program.
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
    bool enableBufferData(const std::shared_ptr<glowl::GLSLProgram> prgm, const MultiParticleDataCall::Particles& parts,
        GLuint vert_buf, const void* vert_ptr, GLuint col_buf, const void* col_ptr, bool create_buffer_data = false);

    /**
     * Unset pointers to vertex and color buffers.
     *
     * @param prgm  The current program.
     *
     * @return 'True' on success, 'false' otherwise.
     */
    bool disableBufferData(const std::shared_ptr<glowl::GLSLProgram> prgm);

    /**
     * Set pointers to vertex and color buffers and corresponding shader variables.
     *
     * @param prgm             The current program.
     * @param parts            The current particles of a list.
     *
     * @return 'True' on success, 'false' otherwise.
     */
    bool enableShaderData(std::shared_ptr<glowl::GLSLProgram> prgm, const MultiParticleDataCall::Particles& parts);

    /**
     * Unset pointers to vertex and color buffers.
     *
     * @return 'True' on success, 'false' otherwise.
     */
    bool disableShaderData();

    /**
     * Enables the transfer function texture.
     *
     * @param prgm    The current program.
     *
     * @return 'True' on success, 'false' otherwise.
     */
    bool enableTransferFunctionTexture(std::shared_ptr<glowl::GLSLProgram> prgm);

    /**
     * Disables the transfer function texture.
     *
     * @return 'True' on success, 'false' otherwise.
     */
    bool disableTransferFunctionTexture();

    /**
     * Enable flag storage.
     *
     * @param prgm             The current program.
     * @param parts            The current particles of a list.
     *
     * @return 'True' on success, 'false' otherwise.
     */
    bool enableFlagStorage(const std::shared_ptr<glowl::GLSLProgram> prgm, MultiParticleDataCall* mpdc);

    /**
     * Enable flag storage.
     *
     * @param prgm             The current program.
     *
     * @return 'True' on success, 'false' otherwise.
     */
    bool disableFlagStorage(const std::shared_ptr<glowl::GLSLProgram> prgm);

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
    void getBytesAndStride(const MultiParticleDataCall::Particles& parts, unsigned int& out_col_bytes,
        unsigned int& out_vert_bytes, unsigned int& out_col_stride, unsigned int& out_vert_stride,
        bool& out_interleaved);

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
    bool makeColorString(const MultiParticleDataCall::Particles& parts, std::string& out_code,
        std::string& out_declaration, bool interleaved, msf::ShaderFactoryOptionsOpenGL& shader_options);

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
    bool makeVertexString(const MultiParticleDataCall::Particles& parts, std::string& out_code,
        std::string& out_declaration, bool interleaved, msf::ShaderFactoryOptionsOpenGL& shader_options);

    /**
     * Make SSBO shaders.
     *
     * @param prgm  ...
     *
     * @return ...
     */
    std::shared_ptr<glowl::GLSLProgram> makeShader(
        const std::string& prgm_name, const msf::ShaderFactoryOptionsOpenGL& shader_options);

    /**
     * Generate SSBO shaders.
     *
     * @param parts  ...
     *
     * @return ...
     */
    std::shared_ptr<glowl::GLSLProgram> generateShader(
        const MultiParticleDataCall::Particles& parts, const std::string& prgm_name);

    /**
     * Returns GLSL minor and major version.
     *
     * @param major The major version of the currently available GLSL version.
     * @param minor The minor version of the currently available GLSL version.
     */
    void getGLSLVersion(int& out_major, int& out_minor) const;

#if defined(SPHERE_MIN_OGL_BUFFER_ARRAY) || defined(SPHERE_MIN_OGL_SPLAT)

    /**
     * Lock single.
     *
     * @param syncObj  ...
     */
    void lockSingle(GLsync& out_sync_obj);

    /**
     * Wait single.
     *
     * @param syncObj  ...
     */
    void waitSingle(const GLsync& sync_obj);

#endif // defined(SPHERE_MIN_OGL_BUFFER_ARRAY) || defined(SPHERE_MIN_OGL_SPLAT)

    // ONLY used for Ambient Occlusion rendering: -------------------------

    /**
     * Rebuild the ambient occlusion gBuffer.
     *
     * @return ...
     */
    bool rebuildGBuffer();

    /**
     * Rebuild working data.
     *
     * @param cr3d    ...
     * @param mpdc    ...
     * @param prgm    ...
     */
    void rebuildWorkingData(megamol::mmstd_gl::CallRender3DGL& cr3d, MultiParticleDataCall* mpdc,
        const std::shared_ptr<glowl::GLSLProgram> prgm);

    /**
     * Render deferred pass.
     *
     * @param cr3d  ...
     */
    void renderDeferredPass(megamol::mmstd_gl::CallRender3DGL& cr3d);

    /**
     * Currently not in use.
     * Generate direction shader array string.
     *
     * @param directions      ...
     * @param directionsName  ...
     *
     * @return ...
     */
    std::string generateDirectionShaderArrayString(
        const std::vector<glm::vec4>& directions, const std::string& directions_name);

    /**
     * Generate 3 cone directions.
     *
     * @param directions  ...
     * @param apex        ...
     */
    void generate3ConeDirections(std::vector<glm::vec4>& out_directions, float apex);
};

} // namespace megamol::moldyn_gl::rendering
