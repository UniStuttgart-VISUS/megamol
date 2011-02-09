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
#include "param/FloatParam.h"
#include "param/IntParam.h"
#include "vislib/Log.h"
#include "vislib/ShallowPoint.h"
#include "vislib/NamedColours.h"
#include "vislib/threadpool.h"
#include "MarchingCubeTables.h"
#include "TetraVoxelizer.h"

using namespace megamol;
using namespace megamol::trisoup;
using namespace megamol::trisoup::volumetrics;


/*
 * VoluMetricJob::VoluMetricJob
 */
VoluMetricJob::VoluMetricJob(void) : core::job::AbstractThreadedJob(), core::Module(),
        getDataSlot("getData", "Slot that connects to a MultiParticleDataCall to fetch the particles in the scene"),
        metricsFilenameSlot("metricsFilenameSlot", "File that will contain the "
		"surfaces and volumes of each particle list per frame"),
        showBorderGeometrySlot("showBorderGeometrySlot", "toggle whether the the surface triangles will be replaced by the border triangles"),
		showBoundingBoxesSlot("showBoundingBoxesSlot", "toggle whether the job subdivision grid will be shown"),
		showSurfaceGeometrySlot("showSurfaceGeometrySlot", "toggle whether the the surface triangles will be shown"),
		radiusMultiplierSlot("radiusMultiplierSlot", "multiplier for the particle radius"),
		continueToNextFrameSlot("continueToNextFrameSlot", "continue computation immediately after a frame finishes,"
		"erasing all debug geometry"),
		resetContinueSlot("resetContinueSlot", "reset the continueToNextFrameSlot to false automatically"),
		outLineDataSlot("outLineData", "Slot that outputs debug line geometry"),
		outTriDataSlot("outTriData", "Slot that outputs debug triangle geometry"),
        cellSizeRatioSlot("cellSizeRatioSlot", "Fraction of the minimal particle radius that is used as cell size"),
        subVolumeResolutionSlot("subVolumeResolutionSlot", "maximum edge length of a subvolume processed as a separate job"),
		MaxRad(0), backBufferIndex(0), meshBackBufferIndex(0), hash(0) {

    this->getDataSlot.SetCompatibleCall<core::moldyn::MultiParticleDataCallDescription>();
    this->MakeSlotAvailable(&this->getDataSlot);

    this->metricsFilenameSlot << new core::param::FilePathParam("");
    this->MakeSlotAvailable(&this->metricsFilenameSlot);

    this->showBorderGeometrySlot << new core::param::BoolParam(false);
    this->MakeSlotAvailable(&this->showBorderGeometrySlot);

	this->showBoundingBoxesSlot << new core::param::BoolParam(false);
	this->MakeSlotAvailable(&this->showBoundingBoxesSlot);

	this->showSurfaceGeometrySlot << new core::param::BoolParam(true);
	this->MakeSlotAvailable(&this->showSurfaceGeometrySlot);

	this->continueToNextFrameSlot << new core::param::BoolParam(false);
	this->MakeSlotAvailable(&this->continueToNextFrameSlot);

	this->resetContinueSlot << new core::param::BoolParam(true);
	this->MakeSlotAvailable(&this->resetContinueSlot);

	this->radiusMultiplierSlot << new core::param::FloatParam(1.0f, 0.0001f, 10000.f);
	this->MakeSlotAvailable(&this->radiusMultiplierSlot);

    this->cellSizeRatioSlot << new core::param::FloatParam(0.5f, 0.01f, 10.0f);
    this->MakeSlotAvailable(&this->cellSizeRatioSlot);

    this->subVolumeResolutionSlot << new core::param::IntParam(128, 16, 2048);
    this->MakeSlotAvailable(&this->subVolumeResolutionSlot);

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
	vislib::Array<TetraVoxelizer*> voxelizerList;
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

		// clear submitted stuff, dealloc.
		while (voxelizerList.Count() > 0) {
			delete voxelizerList[0];
			voxelizerList.RemoveAt(0);
		}
		while (subJobDataList.Count() > 0) {
			delete subJobDataList[0];
			subJobDataList.RemoveAt(0);
		}

        unsigned int partListCnt = datacall->GetParticleListCount();
		MaxRad = -FLT_MAX;
        MinRad = FLT_MAX;
        for (unsigned int partListI = 0; partListI < partListCnt; partListI++) {
			//UINT64 numParticles = datacall->AccessParticles(partListI).GetCount();
            //printf("%u particles in list %u\n", numParticles, partListI);
			VoxelizerFloat r = datacall->AccessParticles(partListI).GetGlobalRadius();
			if (r > MaxRad) {
				MaxRad = r;
			}
            if (r < MinRad) {
                MinRad = r;
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
                    if (currRad < MinRad) {
                        MinRad = currRad;
                    }
				}
			}
        }

		VoxelizerFloat RadMult = this->radiusMultiplierSlot.Param<megamol::core::param::FloatParam>()->Value();
		MaxRad *= RadMult;
        MinRad *= RadMult;
        VoxelizerFloat cellSize = MinRad * this->cellSizeRatioSlot.Param<megamol::core::param::FloatParam>()->Value();
		int bboxBytes = 8 * 3 * sizeof(VoxelizerFloat);
		int bboxIdxes = 12 * 2 * sizeof(unsigned int);
		int vertSize = bboxBytes * partListCnt;
		int idxSize = bboxIdxes * partListCnt;
		bboxVertData[backBufferIndex].AssertSize(vertSize);
		bboxIdxData[backBufferIndex].AssertSize(idxSize);
		SIZE_T bboxOffset = 0;
		unsigned int vertFloatSize = 0;
		unsigned int idxNumOffset = 0;

		vislib::math::Cuboid<VoxelizerFloat> b;
		if (datacall->AccessBoundingBoxes().IsObjectSpaceClipBoxValid()) {
			b = datacall->AccessBoundingBoxes().ObjectSpaceClipBox();
		} else {
			b = datacall->AccessBoundingBoxes().ObjectSpaceBBox();
		}
        // HAZARD that was bullshit, was it?
		//b.Grow(MaxRad);

		int resX = (int) ((VoxelizerFloat)b.Width() / cellSize) + 2;
		int resY = (int) ((VoxelizerFloat)b.Height() / cellSize) + 2;
		int resZ = (int) ((VoxelizerFloat)b.Depth() / cellSize) + 2;
		b.SetWidth(resX * cellSize);
		b.SetHeight(resY * cellSize);
		b.SetDepth(resZ * cellSize);

		appendBox(bboxVertData[backBufferIndex], b, bboxOffset);
		appendBoxIndices(bboxIdxData[backBufferIndex], idxNumOffset);

		int divX = 1;
		int divY = 1;
		int divZ = 1;
        int subVolCells = this->subVolumeResolutionSlot.Param<megamol::core::param::IntParam>()->Value();

		while (divX == 1 && divY == 1 && divZ ==1) {
			subVolCells /= 2;
			divX = (int) ceil((VoxelizerFloat)resX / subVolCells);
			divY = (int) ceil((VoxelizerFloat)resY / subVolCells);
			divZ = (int) ceil((VoxelizerFloat)resZ / subVolCells);
		}

		vertSize += bboxBytes * divX * divY * divZ;
		idxSize += bboxIdxes * divX * divY * divZ;
		bboxVertData[backBufferIndex].AssertSize(vertSize, true);
		bboxIdxData[backBufferIndex].AssertSize(idxSize, true);

		for (int x = 0; x < divX; x++) {
			for (int y = 0; y < divY; y++) {
            //for (int y = 0; y < 1; y++) {
				for (int z = 0; z < divZ; z++) {
                //for (int z = 0; z < 1; z++) {
					VoxelizerFloat left = b.Left() + x * subVolCells * cellSize;
					int restX = resX - x * subVolCells;
					restX = (restX > subVolCells) ? subVolCells + 1: restX;
					VoxelizerFloat right = left + restX * cellSize;
					VoxelizerFloat bottom = b.Bottom() + y * subVolCells * cellSize;
					int restY = resY - y * subVolCells;
					restY = (restY > subVolCells) ? subVolCells + 1: restY;
					VoxelizerFloat top = bottom + restY * cellSize;
					VoxelizerFloat back = b.Back() + z * subVolCells * cellSize;
					int restZ = resZ - z * subVolCells;
					restZ = (restZ > subVolCells) ? subVolCells + 1 : restZ;
					VoxelizerFloat front = back + restZ * cellSize;
					vislib::math::Cuboid<VoxelizerFloat> bx = vislib::math::Cuboid<VoxelizerFloat>(left, bottom, back,
						right, top, front);
					appendBox(bboxVertData[backBufferIndex], bx, bboxOffset);
					appendBoxIndices(bboxIdxData[backBufferIndex], idxNumOffset);

					SubJobData *sjd = new SubJobData();
					sjd->datacall = datacall;
					sjd->Bounds = bx;
                    sjd->CellSize = (VoxelizerFloat)MinRad 
                        * this->cellSizeRatioSlot.Param<megamol::core::param::FloatParam>()->Value();
					sjd->resX = restX;
					sjd->resY = restY;
					sjd->resZ = restZ;
                    sjd->offsetX = x * subVolCells;
                    sjd->offsetY = y * subVolCells;
                    sjd->offsetZ = z * subVolCells;
					sjd->RadMult = RadMult;
					sjd->MaxRad = MaxRad / RadMult;
					subJobDataList.Add(sjd);
					TetraVoxelizer *v = new TetraVoxelizer();
					voxelizerList.Add(v);

                    //if (z == 0 && y == 0) {
					    pool.QueueUserWorkItem(v, sjd);
                    //}
				}
			}
		}
		//}
		this->debugLines[backBufferIndex][0].Set(
				static_cast<unsigned int>(idxNumOffset * 2),
                this->bboxIdxData[backBufferIndex].As<unsigned int>(), this->bboxVertData[backBufferIndex].As<VoxelizerFloat>(),
				vislib::graphics::NamedColours::BlanchedAlmond);
		
		backBufferIndex = 1 - backBufferIndex;
		this->hash++;

        SIZE_T lastCount = pool.CountUserWorkItems();
		while(1) {
			if (pool.Wait(500) && pool.CountUserWorkItems() == 0) {
						// we are done
						break;
			}
            if (lastCount != pool.CountUserWorkItems()) {
			    copyMeshesToBackbuffer(subJobDataList);
                lastCount = pool.CountUserWorkItems();
            }
		}
		copyMeshesToBackbuffer(subJobDataList, true);
        Log::DefaultLog.WriteInfo("Done marching.");

		while(! this->continueToNextFrameSlot.Param<megamol::core::param::BoolParam>()->Value()) {
			Sleep(500);
		}
		if (this->resetContinueSlot.Param<megamol::core::param::BoolParam>()->Value()) {
			this->continueToNextFrameSlot.Param<megamol::core::param::BoolParam>()->SetValue(false);
		}
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
			//b.Grow(MaxRad);
			ldc->AccessBoundingBoxes().SetObjectSpaceClipBox(b);
		}
        ldc->SetFrameCount(1);
		ldc->SetDataHash(this->hash);
	}
	return true;
}

void VoluMetricJob::appendBox(vislib::RawStorage &data, vislib::math::Cuboid<VoxelizerFloat> &b, SIZE_T &offset) {
	vislib::math::ShallowPoint<VoxelizerFloat, 3> (data.AsAt<VoxelizerFloat>(offset 
        + 0 * 3 * sizeof(VoxelizerFloat))) = b.GetLeftBottomFront();
	vislib::math::ShallowPoint<VoxelizerFloat, 3> (data.AsAt<VoxelizerFloat>(offset 
        + 1 * 3 * sizeof(VoxelizerFloat))) = b.GetRightBottomFront();
	vislib::math::ShallowPoint<VoxelizerFloat, 3> (data.AsAt<VoxelizerFloat>(offset 
        +	2 * 3 * sizeof(VoxelizerFloat))) = b.GetRightTopFront();
	vislib::math::ShallowPoint<VoxelizerFloat, 3> (data.AsAt<VoxelizerFloat>(offset 
        +	3 * 3 * sizeof(VoxelizerFloat))) = b.GetLeftTopFront();
	vislib::math::ShallowPoint<VoxelizerFloat, 3> (data.AsAt<VoxelizerFloat>(offset 
        +	4 * 3 * sizeof(VoxelizerFloat))) = b.GetLeftBottomBack();
	vislib::math::ShallowPoint<VoxelizerFloat, 3> (data.AsAt<VoxelizerFloat>(offset 
        +	5 * 3 * sizeof(VoxelizerFloat))) = b.GetRightBottomBack();
	vislib::math::ShallowPoint<VoxelizerFloat, 3> (data.AsAt<VoxelizerFloat>(offset 
        +	6 * 3 * sizeof(VoxelizerFloat))) = b.GetRightTopBack();
	vislib::math::ShallowPoint<VoxelizerFloat, 3> (data.AsAt<VoxelizerFloat>(offset 
        +	7 * 3 * sizeof(VoxelizerFloat))) = b.GetLeftTopBack();
	//return 8 * 3 * sizeof(float) + offset;
	offset += 8 * 3 * sizeof(VoxelizerFloat);
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

bool VoluMetricJob::doBordersTouch(vislib::Array<BorderVoxel *> &border1, vislib::Array<BorderVoxel *> &border2) {
    for (SIZE_T i = 0; i < border1.Count(); i++) {
        for (SIZE_T j = 0; j < border2.Count(); j++) {
            if (border1[i]->doesTouch(border2[j])) {
                return true;
            }
        }
    }
    return false;
}

void VoluMetricJob::copyMeshesToBackbuffer(vislib::Array<SubJobData*> &subJobDataList,
                                           bool outputStatistics) {
	// copy finished meshes to output

	VoxelizerFloat *vert, *norm;
    unsigned char *col;
	//unsigned int *tri;
	vislib::Array<unsigned int> todos;
	todos.SetCapacityIncrement(10);
	for (int i = 0; i < subJobDataList.Count(); i++) {
		if (subJobDataList[i]->Result.done) {
			todos.Add(i);
		}
	}

    if (todos.Count() == 0) {
        return;
    }

    vislib::Array<vislib::Array<unsigned int> > globalSurfaceIDs;
    globalSurfaceIDs.SetCount(todos.Count());
    unsigned int gsi = 0;
    for (int i = 0; i < todos.Count(); i++) {
        SIZE_T sc = subJobDataList[todos[i]]->Result.surfaces.Count();
        globalSurfaceIDs[i].SetCount(sc);
        for (int j = 0; j < sc; j++) {
            globalSurfaceIDs[i][j] = gsi++;
        }
    }

    // TODO: collect totally full subvolumes!
restart:
    for (int i = 0; i < todos.Count(); i++) {
        for (int j = 0; j < subJobDataList[todos[i]]->Result.surfaces.Count(); j++) {
            for (int k = i; k < todos.Count(); k++) {
                // are these neighbors or in the same subvolume?
                vislib::math::Cuboid<VoxelizerFloat> c = subJobDataList[todos[i]]->Bounds;
                c.Union(subJobDataList[todos[k]]->Bounds);
                if ((i == k) || (c.Volume() <= subJobDataList[todos[i]]->Bounds.Volume() 
                                                + subJobDataList[todos[k]]->Bounds.Volume())) {
                    for (int l = 0; l < subJobDataList[todos[k]]->Result.surfaces.Count(); l++) {
                        if (globalSurfaceIDs[k][l] != globalSurfaceIDs[i][j]) {
                            if (doBordersTouch(subJobDataList[todos[i]]->Result.surfaces[j].border,
                                subJobDataList[todos[k]]->Result.surfaces[l].border)) {
                                    if (globalSurfaceIDs[k][l] < globalSurfaceIDs[i][j]) {
                                        globalSurfaceIDs[i][j] = globalSurfaceIDs[k][l];
                                    } else {
                                        globalSurfaceIDs[k][l] = globalSurfaceIDs[i][j];
                                    }
                                    // UGLY HAZARD the devil's left ass cheek!
                                    //i = j = k = 0; l = -1;

                                    // restart
                                    goto restart;
                            }
                        }
                    }
                }
            }
        }
    }
    vislib::Array<unsigned int> uniqueIDs;
    vislib::Array<unsigned int> countPerID;
    vislib::Array<VoxelizerFloat> surfPerID;
    vislib::Array<VoxelizerFloat> volPerID;
    for (int i = 0; i < todos.Count(); i++) {
        for (int j = 0; j < subJobDataList[todos[i]]->Result.surfaces.Count(); j++) {
            SIZE_T pos = uniqueIDs.IndexOf(globalSurfaceIDs[i][j]);
            if (pos == vislib::Array<unsigned int>::INVALID_POS) {
                uniqueIDs.Add(globalSurfaceIDs[i][j]);
                countPerID.Add(subJobDataList[todos[i]]->Result.surfaces[j].mesh.Count() / 9);
                surfPerID.Add(subJobDataList[todos[i]]->Result.surfaces[j].surface);
                volPerID.Add(subJobDataList[todos[i]]->Result.surfaces[j].volume);
            } else {
                countPerID[pos] = countPerID[pos] + (subJobDataList[todos[i]]->Result.surfaces[j].mesh.Count() / 9);
                surfPerID[pos] = surfPerID[pos] + subJobDataList[todos[i]]->Result.surfaces[j].surface;
                volPerID[pos] = volPerID[pos] + subJobDataList[todos[i]]->Result.surfaces[j].volume;
            }
        }
    }
    unsigned int numTriangles = 0;
    for (int i = 0; i < uniqueIDs.Count(); i++) {
        numTriangles += countPerID[i];
        if (outputStatistics) {
            vislib::sys::Log::DefaultLog.WriteInfo("surface %u: %u triangles, surface %f, volume %f", uniqueIDs[i],
                countPerID[i], surfPerID[i], volPerID[i]);
        }
    }
    vert = new VoxelizerFloat[numTriangles * 9];
    norm = new VoxelizerFloat[numTriangles * 9];
    col = new unsigned char[numTriangles * 9];
    //tri = new unsigned int[numTriangles * 3];
    SIZE_T vertOffset = 0;
    SIZE_T triOffset = 0;
    SIZE_T idxOffset = 0;

    if (this->showBorderGeometrySlot.Param<megamol::core::param::BoolParam>()->Value()) {

        for (int i = 0; i < uniqueIDs.Count(); i++) {
            vislib::graphics::ColourRGBAu8 c(rand() * 255, rand() * 255, rand() * 255, 255);
            //vislib::math::ShallowShallowTriangle<double, 3> sst(vert);

            for (int j = 0; j < todos.Count(); j++) {
                for (int k = 0; k < subJobDataList[todos[j]]->Result.surfaces.Count(); k++) {
                    if (globalSurfaceIDs[j][k] == uniqueIDs[i]) {
                        for (SIZE_T l = 0; l < subJobDataList[todos[j]]->Result.surfaces[k].border.Count(); l++) {
                            SIZE_T vertCount = subJobDataList[todos[j]]->Result.surfaces[k].border[l]->triangles.Count() / 3;
                            // TODO hurz
                            //memcpy(&(vert[vertOffset]), subJobDataList[todos[j]]->Result.borderVoxels[k][l]->triangles.PeekElements(),
                            //    vertCount * 3 * sizeof(double));
                            for (SIZE_T m = 0; m < vertCount; m++) {
                                //tri[vertOffset + l] = vertOffset + l;
                                vert[vertOffset + m * 3] = static_cast<float>(
                                    *(subJobDataList[todos[j]]->Result.surfaces[k].border[l]->triangles.PeekElements() + m * 3));
                                vert[vertOffset + m * 3 + 1] = static_cast<float>(
                                    *(subJobDataList[todos[j]]->Result.surfaces[k].border[l]->triangles.PeekElements() + m * 3 + 1));
                                vert[vertOffset + m * 3 + 2] = static_cast<float>(
                                    *(subJobDataList[todos[j]]->Result.surfaces[k].border[l]->triangles.PeekElements() + m * 3 + 2));

                                col[vertOffset + m * 3] = c.R();
                                col[vertOffset + m * 3 + 1] = c.G();
                                col[vertOffset + m * 3 + 2] = c.B();
                            }
                            vertOffset += vertCount * 3;
                        }
                    }
                }
            }
        }
    } else {
        for (int i = 0; i < uniqueIDs.Count(); i++) {
            vislib::graphics::ColourRGBAu8 c(rand() * 255, rand() * 255, rand() * 255, 255);
            //vislib::math::ShallowShallowTriangle<double, 3> sst(vert);

            for (int j = 0; j < todos.Count(); j++) {
                for (int k = 0; k < subJobDataList[todos[j]]->Result.surfaces.Count(); k++) {
                    if (globalSurfaceIDs[j][k] == uniqueIDs[i]) {
                        SIZE_T vertCount = subJobDataList[todos[j]]->Result.surfaces[k].mesh.Count() / 3;
                        // TODO hurz
                        //memcpy(&(vert[vertOffset]), subJobDataList[todos[j]]->Result.surfaces[k].PeekElements(),
                        //     vertCount * 3 * sizeof(double));
                        for (SIZE_T l = 0; l < vertCount; l++) {
                            //tri[vertOffset + l] = vertOffset + l;
                            vert[vertOffset + l * 3] = static_cast<float>(
                                *(subJobDataList[todos[j]]->Result.surfaces[k].mesh.PeekElements() + l * 3));
                            vert[vertOffset + l * 3 + 1] = static_cast<float>(
                                *(subJobDataList[todos[j]]->Result.surfaces[k].mesh.PeekElements() + l * 3 + 1));
                            vert[vertOffset + l * 3 + 2] = static_cast<float>(
                                *(subJobDataList[todos[j]]->Result.surfaces[k].mesh.PeekElements() + l * 3 + 2));

                            col[vertOffset + l * 3] = c.R();
                            col[vertOffset + l * 3 + 1] = c.G();
                            col[vertOffset + l * 3 + 2] = c.B();
                        }
                        //for (unsigned int l = 0; l < triCount * 9; l += 9) {
                        //    sst.SetPointer
                        //}
                        vertOffset += vertCount * 3;
                    }
                }
            }
        }
    }

    for (SIZE_T i = 0; i < vertOffset / 9; i++) {
        vislib::math::ShallowShallowTriangle<VoxelizerFloat, 3> sst(&(vert[i * 9]));
        vislib::math::Vector<VoxelizerFloat, 3> n;
        sst.Normal(n);
        memcpy(&(norm[i * 9]), n.PeekComponents(), sizeof(VoxelizerFloat) * 3);
        memcpy(&(norm[i * 9 + 3]), n.PeekComponents(), sizeof(VoxelizerFloat) * 3);
        memcpy(&(norm[i * 9 + 6]), n.PeekComponents(), sizeof(VoxelizerFloat) * 3);
    }

    debugMeshes[meshBackBufferIndex].SetVertexData(vertOffset / 3, vert, norm, col, NULL, true);
	//debugMeshes[meshBackBufferIndex].SetTriangleData(vertOffset / 3, tri, true);
    debugMeshes[meshBackBufferIndex].SetTriangleData(0, NULL, false);

	meshBackBufferIndex = 1 - meshBackBufferIndex;
	this->hash++;
}
