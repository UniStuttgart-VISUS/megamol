/*
 * VoluMetricJob.cpp
 *
 * Copyright (C) 2010 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */
#include "stdafx.h"
#include "VoluMetricJob.h"
#include "param/FilePathParam.h"
#include "vislib/Log.h"
#include "vislib/ShallowPoint.h"
#include "vislib/NamedColours.h"
#include "MarchingCubeTables.h"

using namespace megamol;
using namespace megamol::trisoup;


/*
 * VoluMetricJob::VoluMetricJob
 */
VoluMetricJob::VoluMetricJob(void) : core::job::AbstractThreadedJob(), core::Module(),
        getDataSlot("getData", "Slot that connects to a MultiParticleDataCall to fetch the particles in the scene"),
        metricsFilenameSlot("metricsFilenameSlot", "File that will contain the "
		"surfaces and volumes of each particle list per frame"),
		outLineDataSlot("outLineData", "Slot that outputs debug line geometry"),
		MaxRad(0), backBufferIndex(0), hash(0) {

    this->getDataSlot.SetCompatibleCall<core::moldyn::MultiParticleDataCallDescription>();
    this->MakeSlotAvailable(&this->getDataSlot);

    this->metricsFilenameSlot << new core::param::FilePathParam("");
    this->MakeSlotAvailable(&this->metricsFilenameSlot);

	this->outLineDataSlot.SetCallback("LinesDataCall", "GetData", &VoluMetricJob::getLineDataCallback);
	this->outLineDataSlot.SetCallback("LinesDataCall", "GetExtent", &VoluMetricJob::getLineExtentCallback);
	this->MakeSlotAvailable(&this->outLineDataSlot);

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

    for (unsigned int frameI = 0; frameI < frameCnt; frameI++) {

        datacall->SetFrameID(frameI, true);
		do {
			if (!(*datacall)(0)) {
				Log::DefaultLog.WriteError("ARGH! No frame here", frameI);
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
        }
		float cellSize = MaxRad * 0.075f;
		int bboxBytes = 8 * 3 * sizeof(float);
		int bboxIdxes = 12 * 2 * sizeof(unsigned int);
		int vertSize = bboxBytes * partListCnt;
		int idxSize = bboxIdxes * partListCnt;
		bboxVertData[backBufferIndex].AssertSize(vertSize);
		bboxIdxData[backBufferIndex].AssertSize(idxSize);
		SIZE_T bboxOffset = 0;
		//SIZE_T idxRawOffset = 0;
		unsigned int idxNumOffset = 0;

        for (unsigned int partListI = 0; partListI < partListCnt; partListI++) {
			core::moldyn::MultiParticleDataCall::Particles &parts = datacall->AccessParticles(partListI);
			UINT64 numParticles = parts.GetCount();
			float *vertexData = (float*)parts.GetVertexData();
			vislib::math::Cuboid<float> b = vislib::math::Cuboid<float>(vislib::math::ShallowPoint<float, 3>(vertexData),
				vislib::math::Dimension<float, 3>(MaxRad * 2, MaxRad * 2, MaxRad * 2));
			for (UINT64 part = 1; part < numParticles; part++) {
				b.GrowToPoint(vislib::math::ShallowPoint<float, 3>(&vertexData[3 * part]));
			}
			b.SetLeft(b.GetLeft() - MaxRad);
			b.SetRight(b.GetRight() + MaxRad);
			b.SetTop(b.GetTop() + MaxRad);
			b.SetBottom(b.GetBottom() - MaxRad);
			b.SetFront(b.GetFront() + MaxRad);
			b.SetBack(b.GetBack() - MaxRad);

			int resX = (int) ((float)b.Width() / cellSize + 0.5f);
			int resY = (int) ((float)b.Height() / cellSize + 0.5f);
			int resZ = (int) ((float)b.Depth() / cellSize + 0.5f);
			b.SetWidth(resX * cellSize);
			b.SetHeight(resY * cellSize);
			b.SetDepth(resZ * cellSize);

			appendBox(bboxVertData[backBufferIndex], b, bboxOffset);
			appendBoxIndices(bboxIdxData[backBufferIndex], idxNumOffset);

			//vislib::math::ShallowPoint<float, 3> (bboxVertData[backBufferIndex].AsAt<float>(bboxBytes * partListI +
			//	0 * 3 * sizeof(float))) = b.GetLeftBottomFront();
			//vislib::math::ShallowPoint<float, 3> (bboxVertData[backBufferIndex].AsAt<float>(bboxBytes * partListI +
			//	1 * 3 * sizeof(float))) = b.GetRightBottomFront();
			//vislib::math::ShallowPoint<float, 3> (bboxVertData[backBufferIndex].AsAt<float>(bboxBytes * partListI +
			//	2 * 3 * sizeof(float))) = b.GetRightTopFront();
			//vislib::math::ShallowPoint<float, 3> (bboxVertData[backBufferIndex].AsAt<float>(bboxBytes * partListI +
			//	3 * 3 * sizeof(float))) = b.GetLeftTopFront();
			//vislib::math::ShallowPoint<float, 3> (bboxVertData[backBufferIndex].AsAt<float>(bboxBytes * partListI +
			//	4 * 3 * sizeof(float))) = b.GetLeftBottomBack();
			//vislib::math::ShallowPoint<float, 3> (bboxVertData[backBufferIndex].AsAt<float>(bboxBytes * partListI +
			//	5 * 3 * sizeof(float))) = b.GetRightBottomBack();
			//vislib::math::ShallowPoint<float, 3> (bboxVertData[backBufferIndex].AsAt<float>(bboxBytes * partListI +
			//	6 * 3 * sizeof(float))) = b.GetRightTopBack();
			//vislib::math::ShallowPoint<float, 3> (bboxVertData[backBufferIndex].AsAt<float>(bboxBytes * partListI +
			//	7 * 3 * sizeof(float))) = b.GetLeftTopBack();
			//for (int i = 0; i < 12; i++) {
			//	*bboxIdxData[backBufferIndex].AsAt<unsigned int>(bboxIdxes * partListI + (2 * i + 0) * sizeof(unsigned int)) =
			//		MarchingCubeTables::a2iEdgeConnection[i][0];
			//	*bboxIdxData[backBufferIndex].AsAt<unsigned int>(bboxIdxes * partListI + (2 * i + 1) * sizeof(unsigned int)) =
			//		MarchingCubeTables::a2iEdgeConnection[i][1];
			//}

			int subVolCells = 256;
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
						int rest;
						float left = b.Left() + x * subVolCells * cellSize;
						rest = resX - x * subVolCells;
						float right = left + (rest > subVolCells ? subVolCells : rest) * cellSize;
						float bottom = b.Bottom() + y * subVolCells * cellSize;
						rest = resY - y * subVolCells;
						float top = bottom + (rest > subVolCells ? subVolCells : rest) * cellSize;
						float back = b.Back() + z * subVolCells * cellSize;
						rest = resZ - z * subVolCells;
						float front = back + (rest > subVolCells ? subVolCells : rest) * cellSize;
						vislib::math::Cuboid<float> bx = vislib::math::Cuboid<float>(left, bottom, back,
							right, top, front);
						appendBox(bboxVertData[backBufferIndex], bx, bboxOffset);
						appendBoxIndices(bboxIdxData[backBufferIndex], idxNumOffset);
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
		ldc->SetData(1, this->debugLines[1 - this->backBufferIndex]);
		ldc->SetDataHash(this->hash);
	}

	return true;
}

bool VoluMetricJob::getLineExtentCallback(core::Call &caller) {
	core::misc::LinesDataCall *ldc = dynamic_cast<core::misc::LinesDataCall*>(&caller);
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