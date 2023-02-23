#pragma once

namespace megamol::trisoup::volumetrics {

class MarchingSquareTables {
public:
    MarchingSquareTables();
    ~MarchingSquareTables();

    static const unsigned int a2fVertexOffset[4][2];
    static const unsigned int a2iEdgeConnection[4][2];
    // here goes the correspondence table I guess
    static const float a2fEdgeDirection[4][2];
    static const unsigned int aiSquareEdgeFlags[16];
};

} // namespace megamol::trisoup::volumetrics
