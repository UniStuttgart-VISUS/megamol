#pragma once

#include "stdafx.h"

#include "vislib/graphics/gl/GLSLGeometryShader.h"
#include "vislib/graphics/gl/GLSLComputeShader.h"
#include "vislib/math/Cuboid.h"
#include "mmcore/utility/ShaderSourceFactory.h"

namespace megamol {
    namespace core {
        namespace utility {

            class MDAOVolumeGenerator {
            public:
                MDAOVolumeGenerator();
                ~MDAOVolumeGenerator();

                bool Init();

                void SetResolution(float resX, float resY, float resZ);

                void ClearVolume();
                void StartInsertion(const vislib::math::Cuboid<float> &obb, const vislib::math::Vector<float, 4> &clipDat);
                void InsertParticles(unsigned int count, float globalRadius, GLuint vertexArray);
                void EndInsertion();

                void SetShaderSourceFactory(megamol::core::utility::ShaderSourceFactory *factory);

                GLuint GetVolumeTextureHandle();
                unsigned int GetDataVersion();

                const vislib::math::Dimension<float, 3>& GetExtents();

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

                vislib::math::Point< float, 3 > boundsMin;
                vislib::math::Dimension<float, 3> boundsSizeInverse;
                vislib::math::Dimension<float, 3> volumeRes;
                vislib::math::Vector<float, 4> clipDat;

            };

        }
    }
}