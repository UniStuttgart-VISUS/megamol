#pragma once

class MarchingSquareTables {
public:
	MarchingSquareTables(void);
	~MarchingSquareTables(void);

	static const unsigned int MarchingSquareTables::a2fVertexOffset[4][2];
	static const unsigned int MarchingSquareTables::a2iEdgeConnection[4][2];
	// here goes the correspondence table I guess
	static const float MarchingSquareTables::a2fEdgeDirection[4][2];
	static const unsigned int MarchingSquareTables::aiSquareEdgeFlags[16];
};