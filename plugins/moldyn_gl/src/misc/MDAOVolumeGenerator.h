#pragma once

#include "stdafx.h"

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "mmcore_gl/utility/ShaderSourceFactory.h"

#include "vislib/math/Cuboid.h"
#include "vislib_gl/graphics/gl/GLSLComputeShader.h"
#include "vislib_gl/graphics/gl/GLSLGeometryShader.h"

#include "OpenGL_Context.h"


namespace megamol {
namespace moldyn_gl {
namespace misc {

class MDAOVolumeGenerator {

public:
    MDAOVolumeGenerator();
    ~MDAOVolumeGenerator();

    bool Init(frontend_resources::OpenGL_Context const& ogl_ctx);

    void SetResolution(float resX, float resY, float resZ);

    void ClearVolume();
    void StartInsertion(const vislib::math::Cuboid<float>& obb, const glm::vec4& clipDat);
    void InsertParticles(unsigned int count, float globalRadius, GLuint vertexArray);
    void EndInsertion();

    void SetShaderSourceFactory(megamol::core_gl::utility::ShaderSourceFactory* factory);

    GLuint GetVolumeTextureHandle();
    unsigned int GetDataVersion();

    const vislib::math::Dimension<float, 3>& GetExtents();

    void RecreateMipmap();

private:
    GLuint fboHandle, volumeHandle;
    vislib_gl::graphics::gl::GLSLGeometryShader volumeShader;
    vislib_gl::graphics::gl::GLSLComputeShader mipmapShader;
    megamol::core_gl::utility::ShaderSourceFactory* factory;
    unsigned int dataVersion;
    bool computeAvailable, clearAvailable;

    unsigned char* clearBuffer;

    glm::vec4 clipDat;
    glm::vec3 boundsMin;
    vislib::math::Dimension<float, 3> boundsSizeInverse;
    vislib::math::Dimension<float, 3> volumeRes;

    // Previous OpenGL State
    GLint viewport[4];
    GLint prevFBO;
};

} /* end namespace misc */
} // namespace moldyn_gl
} /* end namespace megamol */
