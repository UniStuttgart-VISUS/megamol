#pragma once

#include "vislib/graphics/gl/IncludeAllGL.h"
#include "salm/SplineTubes.h"

namespace megamol {
namespace beztube {
namespace salm {

    class SplineTubesTess : public SplineTubes {
    public:
        enum TessTubeShadingStyle {
            Wip = 0,
            Color = 1,
            SmoothNormal = 2,
            HardNormal = 3,
            HardNormalWF = 4,
            BlueWF = 5,
            LitColor = 6
        };
        enum TessTubeRenderSetting : int {
            Null = 0,
            UseCaps = 1,
            ShaderBufferTypeIsSSBO = 2,
            UseGeoShader = 4,
            UseBackPatchCulling = 8,
            UsePerVertexNormals = 16,
            UseFineCapsTessellation = 32,
            InvertColorsForCaps = 64
        };

        SplineTubesTess();
        virtual ~SplineTubesTess();

        void Allocate(TessTubeShadingStyle shadingStyle, TessTubeRenderSetting renderSettings, GLenum bufferUsageHintType, 
            int bufferMaxElementCount, int *nodeCount, int vertexRingCount, int vertexRingVertexCount);
    };

}
}
}
