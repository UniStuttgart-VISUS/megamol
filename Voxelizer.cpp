#include "stdafx.h"
#include "Voxelizer.h"
#include "JobStructures.h"
#include "vislib/Log.h"
#include "vislib/ShallowPoint.h"
#include "MarchingCubeTables.h"

using namespace megamol;
using namespace megamol::trisoup;

Voxelizer::Voxelizer(void) : terminate(false), sjd(NULL), cellFIFO(NULL), fifoLen(0), fifoCur(0), fifoEnd(0) {
	triangleSoup.SetCapacityIncrement(90); // AKA 10 triangles?
}

Voxelizer::~Voxelizer(void) {
}

void Voxelizer::marchCell(FatVoxel *theVolume, TagVolume &markedCells, unsigned int x, unsigned int y, unsigned int z) {
    unsigned int i;
    float CubeValues[8];
    int flagIndex, edgeFlags, edge, triangle;
	int corner, vertex;
    float offset;
	vislib::math::Point<float, 3> EdgeVertex[12];

    //Make a local copy of the values at the cube's corners
    for (i = 0; i < 8; i++) {
        CubeValues[i] = theVolume[
            ((z + MarchingCubeTables::a2fVertexOffset[i][2]) * sjd->resY
                + y + MarchingCubeTables::a2fVertexOffset[i][1]) * sjd->resX
                + x + MarchingCubeTables::a2fVertexOffset[i][0]
		].distField;
    }

    //Find which vertices are inside of the surface and which are outside
    flagIndex = 0;
    for (i = 0; i < 8; i++) {
        if (CubeValues[i] < 0.0f) {
            flagIndex |= 1 << i;
        }
    }

    //Find which edges are intersected by the surface
    edgeFlags = MarchingCubeTables::aiCubeEdgeFlags[flagIndex];

    //If the cube is entirely inside or outside of the surface, then there will be no intersections
    if (edgeFlags == 0) {
        return;
    }

    //Find the point of intersection of the surface with each edge
    //Then find the normal to the surface at those points
//
//- im aktuellen Eintrag der a2iTriangleConnectionTable die Edges paarweise nehmen
//- Edge-Richtung in a2fEdgeDirection nachschlagen
//- Cross Product gibt Richtung der Nachbarzelle, die auf den Stack gelegt werden muss.

    for (edge = 0; edge < 12; edge++) {
        //if there is an intersection on this edge
        if (edgeFlags & (1 << edge)) {
            offset = getOffset(CubeValues[MarchingCubeTables::a2iEdgeConnection[edge][0]],
                CubeValues[MarchingCubeTables::a2iEdgeConnection[edge][1]], 0.0f);
            vislib::math::Point<float, 3> p(sjd->Bounds.Left() + x * sjd->CellSize,
				sjd->Bounds.Bottom() + y * sjd->CellSize,
				sjd->Bounds.Back() + z * sjd->CellSize);

            EdgeVertex[edge].SetX(p.X() + (MarchingCubeTables::a2fVertexOffset[MarchingCubeTables::a2iEdgeConnection[edge][0]][0]
            + offset * MarchingCubeTables::a2fEdgeDirection[edge][0]) * sjd->CellSize);
            EdgeVertex[edge].SetY(p.Y() + (MarchingCubeTables::a2fVertexOffset[MarchingCubeTables::a2iEdgeConnection[edge][0]][1]
            + offset * MarchingCubeTables::a2fEdgeDirection[edge][1]) * sjd->CellSize);
            EdgeVertex[edge].SetZ(p.Z() + (MarchingCubeTables::a2fVertexOffset[MarchingCubeTables::a2iEdgeConnection[edge][0]][2]
            + offset * MarchingCubeTables::a2fEdgeDirection[edge][2]) * sjd->CellSize);

            for (i = 0; i < 2; i++) {
                int lx = int(x) + MarchingCubeTables::neighbourTable[edge][i][0];
                int ly = int(y) + MarchingCubeTables::neighbourTable[edge][i][1];
                int lz = int(z) + MarchingCubeTables::neighbourTable[edge][i][2];
                if ((lx < 0) || (ly < 0) || (lz < 0)
                    || (lx >= int(sjd->resX - 1)) || (ly >= int(sjd->resY - 1)) || (lz >= int(sjd->resZ - 1)))
                    continue;
                if (!markedCells.IsTagged(lx, ly, lz)) {
                    markedCells.Tag(lx, ly, lz);

                    cellFIFO[fifoEnd++].Set(lx, ly, lz);
                    if (fifoEnd >= fifoLen) fifoEnd = 0;
                }
            }
        }
    }

    // make triangles
	vislib::math::Vector<float, 3> normal;
	vislib::math::Vector<float, 3> a, b;
	for (triangle = 0; triangle < 5; triangle++) {
        if (MarchingCubeTables::a2iTriangleConnectionTable[flagIndex][3*triangle] < 0) {
            break;
        }

        a = EdgeVertex[MarchingCubeTables::a2iTriangleConnectionTable[flagIndex][3*triangle + 0]] 
        - EdgeVertex[MarchingCubeTables::a2iTriangleConnectionTable[flagIndex][3*triangle + 1]];
        b = EdgeVertex[MarchingCubeTables::a2iTriangleConnectionTable[flagIndex][3*triangle + 0]] 
        - EdgeVertex[MarchingCubeTables::a2iTriangleConnectionTable[flagIndex][3*triangle + 2]];
        normal = a.Cross(b);

        sjd->Result.surface += normal.Length() / 2.0f;
        for (corner = 0; corner < 3; corner++) {
            vertex = MarchingCubeTables::a2iTriangleConnectionTable[flagIndex][3*triangle + corner];
			sjd->Result.vertices.Append(vislib::math::Point<float,3>(EdgeVertex[vertex]));
			sjd->Result.indices.Append(sjd->Result.vertices.Count() - 1);
        }
    }
}

// fGetOffset finds the approximate point of intersection of the surface
// between two points with the values fValue1 and fValue2
// TODO intersect with the sphere we know is nearest! code below makes no sense for our situation
float Voxelizer::getOffset(float fValue1, float fValue2, float fValueDesired) {
        double fDelta = fValue2 - fValue1;
        if(fDelta == 0.0) {
                return 0.5;
        }
        return (float)((fValueDesired - fValue1)/fDelta);
}

bool Voxelizer::isCellNotEmpty(FatVoxel *theVolume, unsigned x, unsigned y, unsigned z) {
    unsigned int i;
    bool neg = false, pos = false;
    float f;

    //Make a local copy of the values at the cube's corners
    for (i = 0; i < 8; i++) {
        f = theVolume[
            ((z + MarchingCubeTables::a2fVertexOffset[i][2]) * sjd->resY
                + y + MarchingCubeTables::a2fVertexOffset[i][1]) * sjd->resX
                + x + MarchingCubeTables::a2fVertexOffset[i][0]
		].distField;
        neg = neg | (f < 0.0f);
        pos = pos | (f >= 0.0f);
    }
    return neg && pos;
}

DWORD Voxelizer::Run(void *userData) {
	using vislib::sys::Log;

	unsigned int x, y, z;
	unsigned int vertFloatSize = 0;
	float currRad = 0.f, maxRad = -FLT_MAX;
	float currDist;
	vislib::math::Point<unsigned int, 3> pStart, pEnd;
	vislib::math::Point<float, 3> p;
	sjd = static_cast<SubJobData*>(userData);

	TagVolume markedCells(sjd->resX - 1, sjd->resY - 1, sjd->resZ - 1);
	fifoLen = (sjd->resX - 1) * (sjd->resY - 1) * (sjd->resZ - 1) + 1;
	cellFIFO = new vislib::math::Point<unsigned int, 3>[fifoLen];
	unsigned int fifoEnd = 0, fifoCur = 0;
	FatVoxel *volume = new FatVoxel[sjd->resX * sjd->resY * sjd->resZ];
	for (SIZE_T i = 0; i < sjd->resX * sjd->resY * sjd->resZ; i++) {
		volume[i].distField = FLT_MAX;
	}

	UINT64 numParticles = sjd->Particles.GetCount();
	unsigned int stride = sjd->Particles.GetVertexDataStride();
	core::moldyn::MultiParticleDataCall::Particles::VertexDataType dataType =
		sjd->Particles.GetVertexDataType();
	unsigned char *vertexData = (unsigned char*)sjd->Particles.GetVertexData();
	switch (dataType) {
		case core::moldyn::MultiParticleDataCall::Particles::VERTDATA_NONE:
			Log::DefaultLog.WriteError("void vertex data. wut?");
			return -1;
		case core::moldyn::MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZ:
			vertFloatSize = 3 * sizeof(float);
			maxRad = sjd->Particles.GetGlobalRadius();
			break;
		case core::moldyn::MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZR:
			vertFloatSize = 4 * sizeof(float);
			for (UINT64 l = 0; l < numParticles; l++) {
				currRad = (float)vertexData[(vertFloatSize + stride) * l + 3 * sizeof(float)];
				if (currRad > maxRad) {
					maxRad = currRad;
				}
			}
			break;
		case core::moldyn::MultiParticleDataCall::Particles::VERTDATA_SHORT_XYZ:
			Log::DefaultLog.WriteError("This module does not yet like quantized data");
			return -2;
	}

	// sample everything into our temporary volume
	currRad = sjd->Particles.GetGlobalRadius();
	vislib::math::Cuboid<float> bx(sjd->Bounds);
	bx.Grow(maxRad);
	for (UINT64 l = 0; l < numParticles; l++) {
		vislib::math::ShallowPoint<float, 3> sp((float*)&vertexData[(vertFloatSize + stride) * l]);
		if (dataType == core::moldyn::MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZR) {
			currRad = (float)vertexData[(vertFloatSize + stride) * l + 3 * sizeof(float)];
		}
		if (!bx.Contains(sp, vislib::math::Cuboid<float>::FACE_ALL)) {
			continue;
		}
		x = static_cast<unsigned int>((sp.X() - currRad - sjd->Bounds.Left()) / sjd->CellSize);
		if (x > 0) x--;
		if (x < 0) x = 0;
		y = static_cast<unsigned int>((sp.Y() - currRad - sjd->Bounds.Bottom()) / sjd->CellSize);
		if (y > 0) y--;
		if (y < 0) y = 0;
		z = static_cast<unsigned int>((sp.Z() - currRad - sjd->Bounds.Back()) / sjd->CellSize);
		if (z > 0) z--;
		if (z < 0) z = 0;
		pStart.Set(x, y, z);

		x = static_cast<unsigned int>((sp.X() + currRad - sjd->Bounds.Left()) / sjd->CellSize);
		if (x + 1 < sjd->resX) x++;
		if (x + 1 >= sjd->resX) x = sjd->resX - 1;
		y = static_cast<unsigned int>((sp.Y() + currRad - sjd->Bounds.Bottom()) / sjd->CellSize);
		if (y + 1 < sjd->resY) y++;
		if (y + 1 >= sjd->resY) y = sjd->resY - 1;
		z = static_cast<unsigned int>((sp.Z() + currRad - sjd->Bounds.Back()) / sjd->CellSize);
		if (z + 1 < sjd->resZ) z++;
		if (z + 1 >= sjd->resZ) z = sjd->resZ - 1;
		pEnd.Set(x, y, z);

		for (z = pStart.Z(); z <= pEnd.Z(); z++) {
			for (y = pStart.Y(); y <= pEnd.Y(); y++) {
				for (x = pStart.X(); x <= pEnd.X(); x++) {
					// TODO think about this. here the voxel content is determined by a corner
					/*p.Set(
					sjd->Bounds.Left() + x * sjd->CellSize,
					sjd->Bounds.Bottom() + y * sjd->CellSize,
					sjd->Bounds.Back() + z * sjd->CellSize);*/

					// and here it is the center!
					p.Set(
						sjd->Bounds.Left() + x * sjd->CellSize + sjd->CellSize * 0.5f,
						sjd->Bounds.Bottom() + y * sjd->CellSize + sjd->CellSize * 0.5f,
						sjd->Bounds.Back() + z * sjd->CellSize + sjd->CellSize * 0.5f);
					currDist = sp.Distance(p) - currRad;
					SIZE_T i = (z * sjd->resY + y) * sjd->resX + x;
					volume[i].distField = vislib::math::Min(volume[i].distField, currDist);
				}
			}
		}
	}

	// march it
	for (x = 0; x < sjd->resX - 1; x++) {
		for (y = 0; y < sjd->resY - 1; y++) {
			for (z = 0; z < sjd->resZ - 1; z++) {
				if (!markedCells.IsTagged(x, y, z) && isCellNotEmpty(volume, x, y, z)) {
					markedCells.Tag(x, y, z);
					cellFIFO[fifoEnd++].Set(x, y, z);
					if (fifoEnd >= fifoLen) fifoEnd= 0;
					while (fifoCur != fifoEnd) {
						vislib::math::Point<unsigned int, 3> &pos = cellFIFO[fifoCur++];
						if (fifoCur >= fifoLen) fifoCur = 0;
						marchCell(volume, markedCells, pos.X(), pos.Y(), pos.Z());
					}
				}
			}
		}
	}

	sjd->Result.done = true;

	return 0;
}

bool Voxelizer::Terminate(void) {
	terminate = true;
	return true;
}