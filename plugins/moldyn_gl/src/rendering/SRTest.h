#pragma once

#include <unordered_map>

#include "geometry_calls/MultiParticleDataCall.h"
#include "mmcore/Call.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/param/ParamSlot.h"
#include "mmcore/utility/ShaderFactory.h"
#include "mmcore/view/Renderer3DModuleGL.h"

namespace megamol::moldyn_gl::rendering {
struct data_package {
    std::vector<std::vector<float>> data;
    std::vector<uint64_t> data_sizes;
};

using data_package_t = data_package;

struct param_package {
    glm::vec3 dir, up, right, pos;
    float rad;
    glm::vec3 light_dir;
    glm::vec4 global_col;
    glm::vec4 attr;
    glm::mat4 mvp;
    glm::mat4 mvp_inv;
    glm::mat4 mvp_trans;
    float near_;
    float far_;
};

using param_package_t = param_package;

class rendering_task {
public:
    template<typename... Paths>
    rendering_task(std::string const& label, msf::ShaderFactoryOptionsOpenGL const& options, Paths... paths) {
        try {
            program_ = core::utility::make_glowl_shader(label, options, std::forward<Paths>(paths)...);
        } catch (...) {
            core::utility::log::Log::DefaultLog.WriteError("[rendering_task] Failed to create program");
            throw;
        }
    }

    virtual ~rendering_task() = default;

    virtual bool render(param_package_t const& package) = 0;

    virtual bool upload(data_package_t const& package) = 0;

protected:
    glowl::GLSLProgram* get_program() const {
        return program_.get();
    }

private:
    std::unique_ptr<glowl::GLSLProgram> program_;
};

class vao_rt : public rendering_task {
public:
    vao_rt(msf::ShaderFactoryOptionsOpenGL const& options);

    virtual ~vao_rt() = default;

    bool render(param_package_t const& package) override;

    bool upload(data_package_t const& package) override;

private:
    std::vector<GLuint> vaos_;
    std::vector<GLuint> vbos_;
    std::vector<uint64_t> num_prims_;
};

class SRTest : public core::view::Renderer3DModuleGL {
public:
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
    enum class method_e : uint8_t { VAO, SSBO };

    using method_ut = std::underlying_type_t<method_e>;

    bool Render(core::view::CallRender3DGL& call) override;

    bool GetExtents(core::view::CallRender3DGL& call) override;

    void loadData(geocalls::MultiParticleDataCall& in_data);

    core::CallerSlot data_in_slot_;

    core::param::ParamSlot method_slot_;

    std::unordered_map<method_e, std::unique_ptr<rendering_task>> rendering_tasks_;

    data_package_t data_;

    uint64_t in_data_hash_ = std::numeric_limits<uint64_t>::max();

    unsigned int frame_id_ = std::numeric_limits<unsigned int>::max();
};
} // namespace megamol::moldyn_gl::rendering
