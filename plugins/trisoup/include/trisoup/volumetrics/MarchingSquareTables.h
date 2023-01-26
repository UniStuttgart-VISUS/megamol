#ifndef MEGAMOLCORE_MARCHINGSQUARETABLES_H_INCLUDED
#define MEGAMOLCORE_MARCHINGSQUARETABLES_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

namespace megamol {
namespace trisoup {
namespace volumetrics {

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

} /* end namespace volumetrics */
} /* end namespace trisoup */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_MARCHINGSQUARETABLES_H_INCLUDED */
