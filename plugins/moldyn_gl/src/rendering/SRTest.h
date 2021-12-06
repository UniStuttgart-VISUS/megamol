#pragma once

#include <array>
#include <unordered_map>

#include "PerformanceManager.h"
#include "geometry_calls/MultiParticleDataCall.h"
#include "mmcore/Call.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/param/ParamSlot.h"
#include "mmcore_gl/utility/ShaderFactory.h"
#include "mmcore_gl/view/Renderer3DModuleGL.h"

#include "srtest/mesh_shader_task.h"
#include "srtest/rendering_task.h"
#include "srtest/ssbo_shader_task.h"

namespace megamol::moldyn_gl::rendering {

class vao_rt : public rendering_task {
public:
    vao_rt(msf::ShaderFactoryOptionsOpenGL const& options);

    virtual ~vao_rt() = default;

    bool render(GLuint ubo) override;

    bool upload(data_package_t const& package) override;

private:
    std::vector<GLuint> vaos_;
    std::vector<GLuint> vbos_;
    std::vector<GLuint> cbos_;
    std::vector<uint64_t> num_prims_;
    per_list_package_t pl_data_;
};

class tex_rt : public rendering_task {
public:
    tex_rt(msf::ShaderFactoryOptionsOpenGL const& options);

    virtual ~tex_rt() = default;

    bool render(GLuint ubo) override;

    bool upload(data_package_t const& package) override;

private:
    std::vector<GLuint> tex_;
    std::vector<GLuint> buf_;
    std::vector<uint64_t> num_prims_;
    per_list_package_t pl_data_;
};

class copy_rt : public rendering_task {
public:
    copy_rt(msf::ShaderFactoryOptionsOpenGL const& options);

    virtual ~copy_rt();

    bool render(GLuint ubo) override;

    bool upload(data_package_t const& package) override;

private:
    std::vector<GLuint> copy_bos_;

    std::vector<GLuint> xbos_;
    std::vector<GLuint> ybos_;
    std::vector<GLuint> zbos_;
    std::vector<GLuint> radbos_;
    std::vector<GLuint> rbos_;
    std::vector<GLuint> gbos_;
    std::vector<GLuint> bbos_;
    std::vector<GLuint> abos_;

    std::vector<uint64_t> num_prims_;
    per_list_package_t pl_data_;

    std::unique_ptr<glowl::GLSLProgram> comp_program_;
};

class copy_vert_rt : public rendering_task {
public:
    copy_vert_rt(msf::ShaderFactoryOptionsOpenGL const& options);

    virtual ~copy_vert_rt();

    bool render(GLuint ubo) override;

    bool upload(data_package_t const& package) override;

private:
    std::vector<GLuint> copy_bos_;

    std::vector<GLuint> xbos_;
    std::vector<GLuint> ybos_;
    std::vector<GLuint> zbos_;
    std::vector<GLuint> radbos_;
    std::vector<GLuint> rbos_;
    std::vector<GLuint> gbos_;
    std::vector<GLuint> bbos_;
    std::vector<GLuint> abos_;

    std::vector<uint64_t> num_prims_;
    per_list_package_t pl_data_;

    std::unique_ptr<glowl::GLSLProgram> comp_program_;
};

static draw_cmd_t dc_points = [](unsigned int num_points) { glDrawArrays(GL_POINTS, 0, num_points); };
static draw_cmd_t dc_verts = [](unsigned int num_points) { glDrawArrays(GL_QUADS, 0, num_points * 4); };
static draw_cmd_t dc_mesh = [](unsigned int num_points) { glDrawMeshTasksNV(0, num_points / 32 + 1); };

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

class mesh_rt : public ssbo_shader_task {
public:
    mesh_rt(upload_mode const& mode, msf::ShaderFactoryOptionsOpenGL const& options);

    virtual ~mesh_rt() = default;
};

class mesh_altn_rt : public ssbo_shader_task {
public:
    mesh_altn_rt(upload_mode const& mode, msf::ShaderFactoryOptionsOpenGL const& options);

    virtual ~mesh_altn_rt() = default;
};

class mesh_geo_rt : public ssbo_shader_task {
public:
    mesh_geo_rt(upload_mode const& mode, msf::ShaderFactoryOptionsOpenGL const& options);

    virtual ~mesh_geo_rt() = default;
};

class mesh_geo_altn_rt : public ssbo_shader_task {
public:
    mesh_geo_altn_rt(upload_mode const& mode, msf::ShaderFactoryOptionsOpenGL const& options);

    virtual ~mesh_geo_altn_rt() = default;
};

class mesh_geo_task_rt : public mesh_shader_task {
public:
    mesh_geo_task_rt(upload_mode const& mode, msf::ShaderFactoryOptionsOpenGL const& options);

    virtual ~mesh_geo_task_rt() = default;
};

class SRTest : public core_gl::view::Renderer3DModuleGL {
public:
    std::vector<std::string> requested_lifetime_resources() override {
        std::vector<std::string> resources = ModuleGL::requested_lifetime_resources();
        resources.emplace_back(frontend_resources::PerformanceManager_Req_Name);
        return resources;
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
        TEX,
        COPY,
        COPY_VERT,
        SSBO,
        SSBO_GEO,
        SSBO_VERT,
        MESH,
        MESH_ALTN,
        MESH_GEO,
        MESH_GEO_TASK,
        MESH_GEO_ALTN
    };

    using method_ut = std::underlying_type_t<method_e>;

    std::array<std::string, 11> method_strings = {"VAO", "TEX", "COPY", "SSBO", "SSBO_GEO", "SSBO_VERT", "MESH",
        "MESH_ALTN", "MESH_GEO", "MESH_GEO_TASK", "MESH_GEO_ALTN"};

    bool Render(core_gl::view::CallRender3DGL& call) override;

    bool GetExtents(core_gl::view::CallRender3DGL& call) override;

    void loadData(geocalls::MultiParticleDataCall& in_data);

    bool create_shaders();

    bool update_upload_setting();

    core::CallerSlot data_in_slot_;

    core::CallerSlot getLightsSlot;

    core::param::ParamSlot method_slot_;

    core::param::ParamSlot upload_mode_slot_;

    core::param::ParamSlot enforce_upload_slot_;

    // core::param::ParamSlot clip_thres_slot_;

    std::unordered_map<method_e, std::unique_ptr<rendering_task>> rendering_tasks_;

    data_package_t data_;

    uint64_t in_data_hash_ = std::numeric_limits<uint64_t>::max();

    unsigned int frame_id_ = std::numeric_limits<unsigned int>::max();

    GLuint ubo_;

    core::view::Camera old_cam_;

#ifdef PROFILING
    frontend_resources::PerformanceManager::handle_vector timing_handles_;
#endif
};
} // namespace megamol::moldyn_gl::rendering
