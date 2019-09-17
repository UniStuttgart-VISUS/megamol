#pragma once

#include "stdafx.h"

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "mmcore/utility/ShaderSourceFactory.h"

#include "vislib/math/Cuboid.h"
#include "vislib/graphics/gl/GLSLGeometryShader.h"
#include "vislib/graphics/gl/GLSLComputeShader.h"


namespace megamol {
namespace stdplugin {
namespace moldyn {
namespace misc {

    class MDAOVolumeGenerator {

    public:
        MDAOVolumeGenerator();
        ~MDAOVolumeGenerator();

        bool Init();

        void SetResolution(float resX, float resY, float resZ);

        void ClearVolume();
        void StartInsertion(const vislib::math::Cuboid<float> &obb, const glm::vec4 &clipDat);
        void InsertParticles(unsigned int count, float globalRadius, GLuint vertexArray);
        void EndInsertion();

        void SetShaderSourceFactory(megamol::core::utility::ShaderSourceFactory *factory);

        GLuint GetVolumeTextureHandle();
        unsigned int GetDataVersion();

        const vislib::math::Dimension< float, 3>& GetExtents();

        void RecreateMipmap();

    private:
        GLuint fboHandle, volumeHandle;
        vislib::graphics::gl::GLSLGeometryShader volumeShader;
        vislib::graphics::gl::GLSLComputeShader mipmapShader;
        megamol::core::utility::ShaderSourceFactory *factory;
        unsigned int dataVersion;
        GLint viewport[4];
        GLint prevFBO;
        bool computeAvailable, clearAvailable;

        unsigned char* clearBuffer;

        glm::vec4 clipDat;
        glm::vec3 boundsMin;
        vislib::math::Dimension<float, 3> boundsSizeInverse;
        vislib::math::Dimension<float, 3> volumeRes;
    };

} /* end namespace misc */
} /* end namespace moldyn */
} /* end namespace core */
} /* end namespace megamol */