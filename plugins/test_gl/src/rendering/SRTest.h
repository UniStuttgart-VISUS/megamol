/**
 * MegaMol
 * Copyright (c) 2023, MegaMol Dev Team
 * All rights reserved.
 */

#pragma once

#include <array>
#include <unordered_map>

#include "PerformanceManager.h"
#include "geometry_calls/MultiParticleDataCall.h"
#include "mmcore/Call.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/param/ParamSlot.h"
#include "mmcore_gl/utility/SSBOBufferArray.h"
#include "mmcore_gl/utility/ShaderFactory.h"
#include "mmstd_gl/renderer/Renderer3DModuleGL.h"

#include "srtest/mesh_shader_task.h"
#include "srtest/rendering_task.h"
#include "srtest/ssbo_shader_task.h"

#ifdef USE_NVPERF
#include <NvPerfReportGeneratorOpenGL.h>
#include <nvperf_host.h>
#endif

namespace megamol::test_gl::rendering {

class vao_rt : public rendering_task {
public:
    vao_rt(msf::ShaderFactoryOptionsOpenGL const& options);

    virtual ~vao_rt() = default;

    bool render(GLuint ubo) override;

    bool upload(data_package_t const& package) override;

    bool cleanup() override;

private:
    std::vector<GLuint> vaos_;
    std::vector<GLuint> vbos_;
    std::vector<GLuint> cbos_;
    std::vector<uint64_t> num_prims_;
    per_list_package_t pl_data_;
};

#define MESH_WARP_SIZE 31

#define VERT_BASE_IDX "gl_VertexID / 6"
#define VERT_INV_IDX "gl_VertexID % 3"
#define VERT_BUMP_IDX "gl_VertexID % 6 / 3"

#define QUADS_BASE_IDX "gl_VertexID / 4"
#define QUADS_INV_IDX "gl_VertexID % 4"
#define QUADS_BUMP_IDX "0"

#define STRIP_BASE_IDX "gl_InstanceID"
#define STRIP_INV_IDX "gl_VertexID" //"gl_VertexID + 1"
#define STRIP_BUMP_IDX "0"          //"-1"

#define MUZIC_BASE_IDX "gl_VertexID / 4"
#define MUZIC_INV_IDX "gl_VertexID % 4" //"gl_VertexID % 4 + 1"
#define MUZIC_BUMP_IDX "0"              //"-1"

static draw_cmd_t dc_points = [](unsigned int num_points) { glDrawArrays(GL_POINTS, 0, num_points); };
static draw_cmd_t dc_verts = [](unsigned int num_points) { glDrawArrays(GL_TRIANGLES, 0, num_points * 6); };
static draw_cmd_t dc_quads = [](unsigned int num_points) { glDrawArrays(GL_QUADS, 0, num_points * 4); };
static draw_cmd_t dc_strip = [](unsigned int num_points) {
    glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 4, num_points);
};
static auto dc_muzic = [](unsigned int num_points, std::vector<uint32_t> const& indices) -> void {
    glDrawElements(GL_TRIANGLE_STRIP, num_points * 6 - 2, GL_UNSIGNED_INT, indices.data());
};
static draw_cmd_t dc_mesh = [](unsigned int num_points) { glDrawMeshTasksNV(0, num_points / MESH_WARP_SIZE + 1); };

class ssbo_rt : public ssbo_shader_task {
public:
    ssbo_rt(upload_mode const& mode, msf::ShaderFactoryOptionsOpenGL const& options);

    virtual ~ssbo_rt() = default;
};

class ssbo_geo_rt : public ssbo_shader_task {
public:
    ssbo_geo_rt(upload_mode const& mode, msf::ShaderFactoryOptionsOpenGL const& options);

    virtual ~ssbo_geo_rt() = default;
};

class ssbo_vert_rt : public ssbo_shader_task {
public:
    ssbo_vert_rt(upload_mode const& mode, msf::ShaderFactoryOptionsOpenGL const& options);

    virtual ~ssbo_vert_rt() = default;
};

class ssbo_quad_rt : public ssbo_shader_task {
public:
    ssbo_quad_rt(upload_mode const& mode, msf::ShaderFactoryOptionsOpenGL const& options);

    virtual ~ssbo_quad_rt() = default;
};

class ssbo_strip_rt : public ssbo_shader_task {
public:
    ssbo_strip_rt(upload_mode const& mode, msf::ShaderFactoryOptionsOpenGL const& options);

    virtual ~ssbo_strip_rt() = default;
};

class ssbo_muzic_rt : public ssbo_shader_task {
public:
    ssbo_muzic_rt(upload_mode const& mode, msf::ShaderFactoryOptionsOpenGL const& options);

    virtual ~ssbo_muzic_rt() = default;

    bool render(GLuint ubo) override;

    bool upload(data_package_t const& package) override;

    bool cleanup() override;

private:
    std::vector<std::vector<uint32_t>> indices_;

    std::vector<GLuint> ind_buf_;

    std::vector<GLuint> cmd_buf_;

    std::vector<GLuint> offset_buf_;

    std::vector<uint32_t> cmd_count_;
};

class mesh_rt : public ssbo_shader_task {
public:
    mesh_rt(upload_mode const& mode, msf::ShaderFactoryOptionsOpenGL const& options);

    virtual ~mesh_rt() = default;
};

class mesh_geo_rt : public ssbo_shader_task {
public:
    mesh_geo_rt(upload_mode const& mode, msf::ShaderFactoryOptionsOpenGL const& options);

    virtual ~mesh_geo_rt() = default;
};

class SRTest : public mmstd_gl::Renderer3DModuleGL {
public:
    static void requested_lifetime_resources(frontend_resources::ResourceRequest& req) {
        /*std::vector<std::string> resources = */ ModuleGL::requested_lifetime_resources(req);
#ifdef MEGAMOL_USE_PROFILING
        req.require<frontend_resources::PerformanceManager>();
#endif
        //resources.emplace_back(frontend_resources::PerformanceManager_Req_Name);
        //return resources;
    }

    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName(void) {
        return "SRTest";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description(void) {
        return "Test suit for different data upload approaches";
    }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable(void) {
        return true;
    }

    SRTest();

    ~SRTest();

protected:
    bool create() override;

    void release() override;

private:
    enum class method_e : uint8_t {
        VAO,
        SSBO,
        SSBO_GEO,
        SSBO_VERT,
        SSBO_QUAD,
        SSBO_STRIP,
        SSBO_MUZIC,
        MESH,
        MESH_GEO,
        MESH_GEO_TASK
    };

    using method_ut = std::underlying_type_t<method_e>;

    /*std::array<std::string, 10> method_strings = {"VAO", "SSBO", "SSBO_GEO", "SSBO_VERT", "SSBO_QUAD", "SSBO_STRIP",
        "SSBO_MUZIC", "MESH", "MESH_GEO", "MESH_GEO_TASK"};*/

    std::array<std::string, 10> method_strings = {"VAO", "Point_Classic", "Geometry_Shader", "Triangles_Classic",
        "Quads", "Triangle_Strip", "SSBO_MUZIC", "Point_Mesh_Shader", "Triangles_Mesh_Shader", "MESH_GEO_TASK"};

    bool Render(mmstd_gl::CallRender3DGL& call) override;

    bool GetExtents(mmstd_gl::CallRender3DGL& call) override;

    void loadData(geocalls::MultiParticleDataCall& in_data);

    bool createShaders();

    bool updateUploadSetting();

    core::CallerSlot data_in_slot_;

    core::CallerSlot getLightsSlot;

    core::param::ParamSlot method_slot_;

    core::param::ParamSlot upload_mode_slot_;

    core::param::ParamSlot enforce_upload_slot_;

    core::param::ParamSlot use_con_ras_slot_;

    // core::param::ParamSlot clip_thres_slot_;

    std::unordered_map<method_e, std::shared_ptr<rendering_task>> rendering_tasks_;

    data_package_t data_;

    uint64_t in_data_hash_ = std::numeric_limits<uint64_t>::max();

    unsigned int frame_id_ = std::numeric_limits<unsigned int>::max();

    GLuint ubo_;

    core::view::Camera old_cam_;

    glm::vec3 lower_;

    glm::vec3 upper_;

    bool mesh_shader_avail_ = false;

#ifdef MEGAMOL_USE_PROFILING
    frontend_resources::PerformanceManager::handle_vector timing_handles_;
#endif

#ifdef USE_NVPERF
    nv::perf::profiler::ReportGeneratorOpenGL nvperf;
    double nvperfWarmupTime = 0.5;
    NVPW_Device_ClockStatus clockStatus = NVPW_DEVICE_CLOCK_STATUS_UNKNOWN;
#endif
};
} // namespace megamol::test_gl::rendering
