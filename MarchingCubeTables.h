#ifndef MEGAMOLCORE_MARCHINGCUBETABLES_H_INCLUDED
#define MEGAMOLCORE_MARCHINGCUBETABLES_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

namespace megamol {
namespace trisoup {

class MarchingCubeTables {
public:
	MarchingCubeTables(void);
	~MarchingCubeTables(void);

	static const unsigned int a2fVertexOffset[8][3];
	static const unsigned int a2iEdgeConnection[12][2];
	static const float a2fEdgeDirection[12][3];
	static const unsigned int aiCubeEdgeFlags[256];
	static const int a2iTriangleConnectionTable[256][16];
	static const int MarchingCubeTables::a2iTriangleConnectionCount[256];
	static const int neighbourTable[12][2][3];
};

} /* end namespace trisoup */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_MARCHINGCUBETABLES_H_INCLUDED */