#ifndef MEGAMOLCORE_MARCHINGSQUARETABLES_H_INCLUDED
#define MEGAMOLCORE_MARCHINGSQUARETABLES_H_INCLUDED
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

#endif /* MEGAMOLCORE_MARCHINGSQUARETABLES_H_INCLUDED */
