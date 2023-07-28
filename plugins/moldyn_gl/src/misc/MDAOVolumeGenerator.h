#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glowl/glowl.h>

#include "OpenGL_Context.h"
#include "mmcore_gl/utility/ShaderFactory.h"
#include "vislib/math/Cuboid.h"

namespace megamol::moldyn_gl::misc {

class MDAOVolumeGenerator {

public:
    MDAOVolumeGenerator();
    ~MDAOVolumeGenerator();

    bool Init(frontend_resources::OpenGL_Context const& ogl_ctx);

    void SetResolution(float res_x, float res_y, float res_z);

    void ClearVolume();
    void StartInsertion(const vislib::math::Cuboid<float>& obb, const glm::vec4& clip_dat);
    void InsertParticles(unsigned int count, float global_radius, GLuint vertex_array);
    void EndInsertion();

    void SetShaderSourceFactory(msf::ShaderFactoryOptionsOpenGL* so);

    GLuint GetVolumeTextureHandle();
    unsigned int GetDataVersion();

    const vislib::math::Dimension<float, 3>& GetExtents();

    void RecreateMipmap();

private:
    msf::ShaderFactoryOptionsOpenGL* shader_options_;

    GLuint fbo_handle_, volume_handle_;
    std::unique_ptr<glowl::GLSLProgram> volume_prgm_;
    std::unique_ptr<glowl::GLSLProgram> mipmap_prgm_;
    unsigned int data_version_;
    bool compute_available_, clear_available_;

    unsigned char* clear_buffer_;

    glm::vec4 clip_dat_;
    glm::vec3 bounds_min_;
    vislib::math::Dimension<float, 3> bounds_size_inverse_;
    vislib::math::Dimension<float, 3> volume_res_;

    // Previous OpenGL State
    GLint viewport_[4];
    GLint prev_fbo_;
};

} // namespace megamol::moldyn_gl::misc
