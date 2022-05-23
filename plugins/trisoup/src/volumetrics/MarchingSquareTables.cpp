#include "trisoup/volumetrics/MarchingSquareTables.h"

using namespace megamol;
using namespace megamol::trisoup;
using namespace megamol::trisoup::volumetrics;

MarchingSquareTables::MarchingSquareTables(void) {}

MarchingSquareTables::~MarchingSquareTables(void) {}

const unsigned int MarchingSquareTables::a2fVertexOffset[4][2] = {{0, 0}, {1, 0}, {1, 1}, {0, 1}};

const unsigned int MarchingSquareTables::a2iEdgeConnection[4][2] = {{0, 1}, {1, 2}, {2, 3}, {3, 0}};

// here goes the correspondence table I guess

const float MarchingSquareTables::a2fEdgeDirection[4][2] = {{1.0, 0.0}, {0.0, 1.0}, {-1.0, 0.0}, {0.0, -1.0}};

// this is 4-bit
const unsigned int MarchingSquareTables::aiSquareEdgeFlags[16] = {
    // 0000 1001 0011 1010 0110 1111 0101 1100
    // 1100 0101 1111 0110 1010 0011 1001 0000
    0, 9, 3, 10, 6, 15, 5, 12, 12, 5, 15, 6, 10, 3, 9, 0};
