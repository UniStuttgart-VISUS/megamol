/*
 * VoluMetricJob.cpp
 *
 * Copyright (C) 2010 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */
#include "stdafx.h"
#include "VoluMetricJob.h"
#include "param/FilePathParam.h"
#include "param/BoolParam.h"
#include "vislib/Log.h"
#include "vislib/ShallowPoint.h"
#include "vislib/NamedColours.h"
#include "vislib/threadpool.h"
#include "MarchingCubeTables.h"
#include "Voxelizer.h"

using namespace megamol;
using namespace megamol::trisoup;


/*
 * VoluMetricJob::VoluMetricJob
 */
VoluMetricJob::VoluMetricJob(void) : core::job::AbstractThreadedJob(), core::Module(),
        getDataSlot("getData", "Slot that connects to a MultiParticleDataCall to fetch the particles in the scene"),
        metricsFilenameSlot("metricsFilenameSlot", "File that will contain the "
		"surfaces and volumes of each particle list per frame"),
		showBoundingBoxesSlot("showBoundingBoxesSlot", "toggle whether the job subdivision grid will be shown"),
		showSurfaceGeometrySlot("showSurfaceGeometrySlot", "toggle whether the the surface triangles will be shown"),
		outLineDataSlot("outLineData", "Slot that outputs debug line geometry"),
		outTriDataSlot("outTriData", "Slot that outputs debug triangle geometry"),
		MaxRad(0), backBufferIndex(0), meshBackBufferIndex(0), hash(0) {

    this->getDataSlot.SetCompatibleCall<core::moldyn::MultiParticleDataCallDescription>();
    this->MakeSlotAvailable(&this->getDataSlot);

    this->metricsFilenameSlot << new core::param::FilePathParam("");
    this->MakeSlotAvailable(&this->metricsFilenameSlot);

	this->showBoundingBoxesSlot << new core::param::BoolParam(false);
	this->MakeSlotAvailable(&this->showBoundingBoxesSlot);

	this->showSurfaceGeometrySlot << new core::param::BoolParam(true);
	this->MakeSlotAvailable(&this->showSurfaceGeometrySlot);

	this->outLineDataSlot.SetCallback("LinesDataCall", "GetData", &VoluMetricJob::getLineDataCallback);
	this->outLineDataSlot.SetCallback("LinesDataCall", "GetExtent", &VoluMetricJob::getLineExtentCallback);
	this->MakeSlotAvailable(&this->outLineDataSlot);

	this->outTriDataSlot.SetCallback("CallTriMeshData", "GetData", &VoluMetricJob::getTriDataCallback);
	this->outTriDataSlot.SetCallback("CallTriMeshData", "GetExtent", &VoluMetricJob::getLineExtentCallback);
	this->MakeSlotAvailable(&this->outTriDataSlot);

	// TODO: Implement

}


/*
 * VoluMetricJob::~VoluMetricJob
 */
VoluMetricJob::~VoluMetricJob(void) {
    this->Release();
}


/*
 * VoluMetricJob::create
 */
bool VoluMetricJob::create(void) {

    // TODO: Implement

    return true;
}


/*
 * VoluMetricJob::release
 */
void VoluMetricJob::release(void) {

    // TODO: Implement

}


/*
 * VoluMetricJob::Run
 */
DWORD VoluMetricJob::Run(void *userData) {
    using vislib::sys::Log;

    core::moldyn::MultiParticleDataCall *datacall = this->getDataSlot.CallAs<core::moldyn::MultiParticleDataCall>();
    if (datacall == NULL) {
        Log::DefaultLog.WriteError("No data source connected to VoluMetricJob");
        return -1;
    }
	if (!(*datacall)(1)) {
		Log::DefaultLog.WriteError("Data source does not answer to extent request");
		return -2;
	}

    unsigned int frameCnt = datacall->FrameCount();
    Log::DefaultLog.WriteInfo("Data source with %u frame(s)", frameCnt);

	vislib::sys::ThreadPool pool;
	vislib::Array<Voxelizer*> voxelizerList;
	vislib::Array<SubJobData*> subJobDataList;

	voxelizerList.SetCapacityIncrement(10);
	subJobDataList.SetCapacityIncrement(10);

    for (unsigned int frameI = 0; frameI < frameCnt; frameI++) {

        datacall->SetFrameID(frameI, true);
		do {
			if (!(*datacall)(0)) {
				Log::DefaultLog.WriteError("ARGH! No frame here");
				return -3;
			}
		} while (datacall->FrameID() != frameI && (vislib::sys::Thread::Sleep(100), true));

        unsigned int partListCnt = datacall->GetParticleListCount();
		MaxRad = -FLT_MAX;
        for (unsigned int partListI = 0; partListI < partListCnt; partListI++) {
			//UINT64 numParticles = datacall->AccessParticles(partListI).GetCount();
            //printf("%u particles in list %u\n", numParticles, partListI);
			float r = datacall->AccessParticles(partListI).GetGlobalRadius();
			if (r > MaxRad) {
				MaxRad = r;
			}
			if (datacall->AccessParticles(partListI).GetVertexDataType() ==
				core::moldyn::MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZR) {
				UINT64 numParticles = datacall->AccessParticles(partListI).GetCount();
				unsigned int stride = datacall->AccessParticles(partListI).GetVertexDataStride();
				unsigned char *vertexData = (unsigned char*)datacall->AccessParticles(partListI).GetVertexData();
				for (UINT64 l = 0; l < numParticles; l++) {
					vislib::math::ShallowPoint<float, 3> sp((float*)&vertexData[(4 * sizeof(float) + stride) * l]);
					float currRad = (float)vertexData[(4 * sizeof(float) + stride) * l + 3];
					if (currRad > MaxRad) {
						MaxRad = currRad;
					}
				}
			}
        }
		float cellSize = MaxRad * 0.25;//* 0.075f;
		int bboxBytes = 8 * 3 * sizeof(float);
		int bboxIdxes = 12 * 2 * sizeof(unsigned int);
		int vertSize = bboxBytes * partListCnt;
		int idxSize = bboxIdxes * partListCnt;
		bboxVertData[backBufferIndex].AssertSize(vertSize);
		bboxIdxData[backBufferIndex].AssertSize(idxSize);
		SIZE_T bboxOffset = 0;
		unsigned int vertFloatSize = 0;
		unsigned int idxNumOffset = 0;

		// TODO BUG HAZARD FIXME! this is debug! it must start from 0!!!!!
        for (unsigned int partListI = 0; partListI < partListCnt; partListI++) {
			core::moldyn::MultiParticleDataCall::Particles &parts = datacall->AccessParticles(partListI);
			UINT64 numParticles = parts.GetCount();
			unsigned int stride = parts.GetVertexDataStride();
			switch (parts.GetVertexDataType()) {
				case core::moldyn::MultiParticleDataCall::Particles::VERTDATA_NONE:
					Log::DefaultLog.WriteError("void vertex data. wut?");
					return -4;
				case core::moldyn::MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZ:
					vertFloatSize = 3 * sizeof(float);
					break;
				case core::moldyn::MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZR:
					vertFloatSize = 4 * sizeof(float);
					break;
				case core::moldyn::MultiParticleDataCall::Particles::VERTDATA_SHORT_XYZ:
					Log::DefaultLog.WriteError("This module does not yet like quantized data");
					return -5;
			}
			unsigned char *vertexData = (unsigned char*)parts.GetVertexData();
			vislib::math::Cuboid<float> b = vislib::math::Cuboid<float>(vislib::math::ShallowPoint<float, 3>((float*)vertexData),
				vislib::math::Dimension<float, 3>(MaxRad * 2, MaxRad * 2, MaxRad * 2));
			for (UINT64 part = 1; part < numParticles; part++) {
				b.GrowToPoint(vislib::math::ShallowPoint<float, 3>(
					(float*)&vertexData[(vertFloatSize + stride) * part]));
			}
			b.Grow(MaxRad);

			int resX = (int) ((float)b.Width() / cellSize) + 2;
			int resY = (int) ((float)b.Height() / cellSize) + 2;
			int resZ = (int) ((float)b.Depth() / cellSize) + 2;
			b.SetWidth(resX * cellSize);
			b.SetHeight(resY * cellSize);
			b.SetDepth(resZ * cellSize);

			appendBox(bboxVertData[backBufferIndex], b, bboxOffset);
			appendBoxIndices(bboxIdxData[backBufferIndex], idxNumOffset);

			int subVolCells = 64;
			int divX = (int) ceil((float)resX / subVolCells);
			int divY = (int) ceil((float)resY / subVolCells);
			int divZ = (int) ceil((float)resZ / subVolCells);
	
			vertSize += bboxBytes * divX * divY * divZ;
			idxSize += bboxIdxes * divX * divY * divZ;
			bboxVertData[backBufferIndex].AssertSize(vertSize, true);
			bboxIdxData[backBufferIndex].AssertSize(idxSize, true);

			for (int x = 0; x < divX; x++) {
				for (int y = 0; y < divY; y++) {
					for (int z = 0; z < divZ; z++) {
						float left = b.Left() + x * subVolCells * cellSize;
						int restX = resX - x * subVolCells;
						restX = (restX > subVolCells) ? subVolCells + 1: restX;
						float right = left + restX * cellSize;
						float bottom = b.Bottom() + y * subVolCells * cellSize;
						int restY = resY - y * subVolCells;
						restY = (restY > subVolCells) ? subVolCells + 1: restY;
						float top = bottom + restY * cellSize;
						float back = b.Back() + z * subVolCells * cellSize;
						int restZ = resZ - z * subVolCells;
						restZ = (restZ > subVolCells) ? subVolCells + 1 : restZ;
						float front = back + restZ * cellSize;
						vislib::math::Cuboid<float> bx = vislib::math::Cuboid<float>(left, bottom, back,
							right, top, front);
						appendBox(bboxVertData[backBufferIndex], bx, bboxOffset);
						appendBoxIndices(bboxIdxData[backBufferIndex], idxNumOffset);

						SubJobData *sjd = new SubJobData(parts);
						sjd->Bounds = bx;
						sjd->CellSize = cellSize;
						sjd->resX = restX;
						sjd->resY = restY;
						sjd->resZ = restZ;
						sjd->MaxRad = MaxRad;
						subJobDataList.Add(sjd);
						Voxelizer *v = new Voxelizer();
						voxelizerList.Add(v);

						pool.QueueUserWorkItem(v, sjd);
					}
				}
			}
		}
		this->debugLines[backBufferIndex][0].Set(
				static_cast<unsigned int>(idxNumOffset * 2),
                this->bboxIdxData[backBufferIndex].As<unsigned int>(), this->bboxVertData[backBufferIndex].As<float>(),
				vislib::graphics::NamedColours::BlanchedAlmond);
		
		backBufferIndex = 1 - backBufferIndex;
		this->hash++;

		while(1) {
			if (pool.Wait(500) && pool.CountUserWorkItems() == 0) {
						// we are done
						break;
			}
			copyMeshesToBackbuffer(subJobDataList);
		}
		copyMeshesToBackbuffer(subJobDataList);
    }

    return 0;
}

bool VoluMetricJob::getLineDataCallback(core::Call &caller) {
	core::misc::LinesDataCall *ldc = dynamic_cast<core::misc::LinesDataCall*>(&caller);
    if (ldc == NULL) return false;

	if (this->hash == 0) {
		ldc->SetData(0, NULL);
		ldc->SetDataHash(0);
	} else {
		// TODO: only passing the bounding boxes for now
		if (this->showBoundingBoxesSlot.Param<megamol::core::param::BoolParam>()->Value()) {
			ldc->SetData(1, this->debugLines[1 - this->backBufferIndex]);
		} else {
			ldc->SetData(0, NULL);
		}
		ldc->SetDataHash(this->hash);
	}

	return true;
}

bool VoluMetricJob::getTriDataCallback(core::Call &caller) {
	CallTriMeshData *mdc = dynamic_cast<CallTriMeshData*>(&caller);
	if (mdc == NULL) return false;

	if (this->hash == 0) {
		mdc->SetObjects(0, NULL);
		mdc->SetDataHash(0);
	} else {
		if (this->showSurfaceGeometrySlot.Param<megamol::core::param::BoolParam>()->Value()) {
			mdc->SetObjects(1, &(this->debugMeshes[1 - this->meshBackBufferIndex]));
		} else {
			mdc->SetObjects(0, NULL);
		}
		mdc->SetDataHash(this->hash);
	}

	return true;
}

bool VoluMetricJob::getLineExtentCallback(core::Call &caller) {
	core::AbstractGetData3DCall *ldc = dynamic_cast<core::AbstractGetData3DCall*>(&caller);
    if (ldc == NULL) return false;

	core::moldyn::MultiParticleDataCall *datacall = this->getDataSlot.CallAs<core::moldyn::MultiParticleDataCall>();
	if ((datacall == NULL) || (!(*datacall)(1))) {
        ldc->AccessBoundingBoxes().Clear();
        ldc->SetFrameCount(1);
        ldc->SetDataHash(0);
	} else {
		if (MaxRad == 0) {
			ldc->AccessBoundingBoxes() = datacall->AccessBoundingBoxes();
		} else {
			ldc->AccessBoundingBoxes().SetObjectSpaceBBox(datacall->AccessBoundingBoxes().ObjectSpaceBBox());
			vislib::math::Cuboid<float> b = datacall->AccessBoundingBoxes().ObjectSpaceClipBox();
			// TODO: maybe senseless paranoia?
			b.Set(b.Left() - MaxRad, b.Bottom() - MaxRad, b.Back() - MaxRad, b.Right() + MaxRad,
				b.Top() + MaxRad, b.Front() + MaxRad);
			ldc->AccessBoundingBoxes().SetObjectSpaceClipBox(b);
		}
        ldc->SetFrameCount(1);
		ldc->SetDataHash(this->hash);
	}
	return true;
}

void VoluMetricJob::appendBox(vislib::RawStorage &data, vislib::math::Cuboid<float> &b, SIZE_T &offset) {
	vislib::math::ShallowPoint<float, 3> (data.AsAt<float>(offset + 0 * 3 * sizeof(float))) = b.GetLeftBottomFront();
	vislib::math::ShallowPoint<float, 3> (data.AsAt<float>(offset + 1 * 3 * sizeof(float))) = b.GetRightBottomFront();
	vislib::math::ShallowPoint<float, 3> (data.AsAt<float>(offset +	2 * 3 * sizeof(float))) = b.GetRightTopFront();
	vislib::math::ShallowPoint<float, 3> (data.AsAt<float>(offset +	3 * 3 * sizeof(float))) = b.GetLeftTopFront();
	vislib::math::ShallowPoint<float, 3> (data.AsAt<float>(offset +	4 * 3 * sizeof(float))) = b.GetLeftBottomBack();
	vislib::math::ShallowPoint<float, 3> (data.AsAt<float>(offset +	5 * 3 * sizeof(float))) = b.GetRightBottomBack();
	vislib::math::ShallowPoint<float, 3> (data.AsAt<float>(offset +	6 * 3 * sizeof(float))) = b.GetRightTopBack();
	vislib::math::ShallowPoint<float, 3> (data.AsAt<float>(offset +	7 * 3 * sizeof(float))) = b.GetLeftTopBack();
	//return 8 * 3 * sizeof(float) + offset;
	offset += 8 * 3 * sizeof(float);
}

void VoluMetricJob::appendBoxIndices(vislib::RawStorage &data, unsigned int &numOffset) {
	for (int i = 0; i < 12; i++) {
		*data.AsAt<unsigned int>((2 * (numOffset + i) + 0) * sizeof(unsigned int)) =
			MarchingCubeTables::a2iEdgeConnection[i][0] + numOffset * 8 / 12;
		*data.AsAt<unsigned int>((2 * (numOffset + i) + 1) * sizeof(unsigned int)) =
			MarchingCubeTables::a2iEdgeConnection[i][1] + numOffset * 8 / 12;
	}
	//return 12 * 2 * sizeof(unsigned int) + offset;
	//rawOffset += 12 * 2 * sizeof(unsigned int);
	numOffset += 12;
}

void VoluMetricJob::copyMeshesToBackbuffer(vislib::Array<SubJobData*> &subJobDataList) {
	// copy finished meshes to output
	float *vert, *norm;
	unsigned int *tri;
	vislib::Array<unsigned int> todos;
	todos.SetCapacityIncrement(10);
	for (int i = 0; i < subJobDataList.Count(); i++) {
		if (subJobDataList[i]->Result.done) {
			todos.Add(i);
		}
	}
	unsigned int numVertices = 0;
	for (int i = 0; i < todos.Count(); i++) {
		numVertices += subJobDataList[todos[i]]->Result.vertices.Count();
	}
	vert = new float[numVertices * 3];
	norm = new float[numVertices * 3];
	tri = new unsigned int[numVertices * 3];
	SIZE_T vertOffset = 0;
	SIZE_T triOffset = 0;
	SIZE_T idxOffset = 0;
	for (int i = 0; i < todos.Count(); i++) {
		SubJobData *sjd = subJobDataList[todos[i]];
		for (unsigned int j = 0; j < sjd->Result.vertices.Count(); j++) {
			memcpy(&(vert[vertOffset]), (sjd->Result.vertices[j].PeekCoordinates()), 3 * sizeof(float));
			memcpy(&(norm[vertOffset]), (sjd->Result.normals[j].PeekComponents()), 3 * sizeof(float));
			vertOffset += 3;
			//memcpy(&(tri[triOffset]), &sjd->Result.indices[j], 1 * sizeof(unsigned int));
			//triOffset += 1;
			tri[triOffset++] = sjd->Result.indices[j] + idxOffset;
		}
		idxOffset += sjd->Result.vertices.Count();
	}
	debugMeshes[meshBackBufferIndex].SetVertexData(numVertices, vert, norm, NULL, NULL, true);
	debugMeshes[meshBackBufferIndex].SetTriangleData(numVertices / 3, tri, true);
	meshBackBufferIndex = 1 - meshBackBufferIndex;
	this->hash++;
}
