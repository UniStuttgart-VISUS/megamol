#pragma once

namespace megamol::trisoup::volumetrics {

class MarchingCubeTables {
public:
    MarchingCubeTables();
    ~MarchingCubeTables();

    static const unsigned int a2fVertexOffset[8][3];
    static const unsigned int a2iEdgeConnection[12][2];
    static const float a2fEdgeDirection[12][3];
    static const unsigned int aiCubeEdgeFlags[256];
    static const int a2iTriangleConnectionTable[256][16];
    static const unsigned char a2ucTriangleSurfaceID[256][5];
    static const unsigned char a2ucTriangleConnectionCount[256];
    static const int neighbourTable[12][2][3];
};

} // namespace megamol::trisoup::volumetrics
