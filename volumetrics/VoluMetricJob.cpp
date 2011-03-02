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
#include "vislib/sysfunctions.h"
#include "vislib/ConsoleProgressBar.h"
#include "vislib/SystemInformation.h"

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

    this->continueToNextFrameSlot << new core::param::BoolParam(true);
    this->MakeSlotAvailable(&this->continueToNextFrameSlot);

    this->resetContinueSlot << new core::param::BoolParam(false);
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

    // Intentionally empty

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

    if (!metricsFilenameSlot.Param<core::param::FilePathParam>()->Value().IsEmpty()) {
        if (!this->statisticsFile.Open(metricsFilenameSlot.Param<core::param::FilePathParam>()->Value(),
            vislib::sys::File::WRITE_ONLY, vislib::sys::File::SHARE_READ,
            vislib::sys::File::CREATE_OVERWRITE)) {
            Log::DefaultLog.WriteError("Could not open statistics file for writing");
            return -3;
        }
    }

    vislib::Array<TetraVoxelizer*> voxelizerList;

    voxelizerList.SetCapacityIncrement(16);
    SubJobDataList.SetCapacityIncrement(16);

    for (unsigned int frameI = 0; frameI < frameCnt; frameI++) {

        vislib::sys::ThreadPool pool;
        pool.SetThreadCount(static_cast<SIZE_T>(vislib::sys::SystemInformation::ProcessorCount() * 1.5));

        datacall->SetFrameID(frameI, true);
        do {
            if (!(*datacall)(0)) {
                Log::DefaultLog.WriteError("ARGH! No frame here");
                return -3;
            }
        } while (datacall->FrameID() != frameI && (vislib::sys::Thread::Sleep(100), true));

        this->MaxGlobalID = 0;

        // clear submitted stuff, dealloc.
        while (voxelizerList.Count() > 0) {
            delete voxelizerList[0];
            voxelizerList.RemoveAt(0);
        }
        while (SubJobDataList.Count() > 0) {
            delete SubJobDataList[0];
            SubJobDataList.RemoveAt(0);
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

        int resX = (int) ((VoxelizerFloat)b.Width() / cellSize) + 2;
        int resY = (int) ((VoxelizerFloat)b.Height() / cellSize) + 2;
        int resZ = (int) ((VoxelizerFloat)b.Depth() / cellSize) + 2;
        b.SetWidth(resX * cellSize);
        b.SetHeight(resY * cellSize);
        b.SetDepth(resZ * cellSize);

        appendBox(bboxVertData[backBufferIndex], b, bboxOffset);
        appendBoxIndices(bboxIdxData[backBufferIndex], idxNumOffset);

        divX = 1;
        divY = 1;
        divZ = 1;
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

        bool storeMesh = 
            (this->outTriDataSlot.GetStatus() == megamol::core::AbstractSlot::STATUS_CONNECTED);

        vislib::sys::ConsoleProgressBar pb;
        pb.Start("Computing Frame", divX * divY * divZ);

        vislib::sys::Log::DefaultLog.WriteInfo("Grid: %ux%ux%u", divX, divY, divZ);
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
                    sjd->parent = this;
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
                    sjd->gridX = x;
                    sjd->gridY = y;
                    sjd->gridZ = z;
                    sjd->RadMult = RadMult;
                    sjd->MaxRad = MaxRad / RadMult;
                    sjd->storeMesh = storeMesh;
                    SubJobDataList.Add(sjd);
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

        vislib::Array<vislib::Array<unsigned int> > globalSurfaceIDs;
        vislib::Array<unsigned int> uniqueIDs;
        vislib::Array<SIZE_T> countPerID;
        vislib::Array<VoxelizerFloat> surfPerID;
        vislib::Array<VoxelizerFloat> volPerID;

        SIZE_T lastCount = pool.CountUserWorkItems();
        while(1) {
            if (pool.Wait(500) && pool.CountUserWorkItems() == 0) {
                        // we are done
                        break;
            }
            if (lastCount != pool.CountUserWorkItems()) {
                pb.Set(divX * divY * divZ - pool.CountUserWorkItems());
                generateStatistics(uniqueIDs, countPerID, surfPerID, volPerID);
                if (storeMesh) {
                    copyMeshesToBackbuffer(uniqueIDs);
                }
                lastCount = pool.CountUserWorkItems();
            }
        }
        generateStatistics(uniqueIDs, countPerID, surfPerID, volPerID);
        outputStatistics(frameI, uniqueIDs, countPerID, surfPerID, volPerID);
        if (storeMesh) {
            copyMeshesToBackbuffer(uniqueIDs);
        }
        pb.Stop();
        Log::DefaultLog.WriteInfo("Done marching.");
        pool.Terminate(true);

        while(! this->continueToNextFrameSlot.Param<megamol::core::param::BoolParam>()->Value()) {
            Sleep(500);
        }
        if (this->resetContinueSlot.Param<megamol::core::param::BoolParam>()->Value()) {
            this->continueToNextFrameSlot.Param<megamol::core::param::BoolParam>()->SetValue(false);
        }
    }

    if (!metricsFilenameSlot.Param<core::param::FilePathParam>()->Value().IsEmpty()) {
        statisticsFile.Close();
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
        ldc->AccessBoundingBoxes() = datacall->AccessBoundingBoxes();
        ldc->SetFrameCount(1);
        ldc->SetDataHash(this->hash);
    }
    return true;
}


VISLIB_FORCEINLINE bool VoluMetricJob::areSurfacesJoinable(int i, int j, int k, int l) {

    if (SubJobDataList[i]->Result.surfaces[j].globalID
            != SubJobDataList[k]->Result.surfaces[l].globalID) {
        if (SubJobDataList[i]->Result.surfaces[j].surface == 0.0
            && SubJobDataList[k]->Result.surfaces[l].surface == 0.0) {
                // both are full, can be joined trivially
                return true;
        } else if (SubJobDataList[i]->Result.surfaces[j].surface == 0.0) {
            if (isSurfaceJoinableWithSubvolume(SubJobDataList[k], l, 
                SubJobDataList[i])) {
                    return true;
            }
        } else if (SubJobDataList[k]->Result.surfaces[l].surface == 0.0) {
            if (isSurfaceJoinableWithSubvolume(SubJobDataList[i], j, 
                SubJobDataList[k])) {
                    return true;
            }
        } else {
            if (SubJobDataList[i]->Result.surfaces[j].border != NULL &&
                SubJobDataList[k]->Result.surfaces[l].border != NULL) {
                    if (doBordersTouch(*SubJobDataList[i]->Result.surfaces[j].border,
                        *SubJobDataList[k]->Result.surfaces[l].border)) {
                            return true;
                    }
            } else {
                //ASSERT(false);
#ifdef ULTRADEBUG
                vislib::sys::Log::DefaultLog.WriteInfo("tried to compare (%u,%u,%u)[%u,%u][%u]%s with (%u,%u,%u)[%u,%u][%u]%s",
                    SubJobDataList[i]->gridX, SubJobDataList[i]->gridY, SubJobDataList[i]->gridZ,
                    i, j, SubJobDataList[i]->Result.surfaces[j].globalID,
                    (SubJobDataList[i]->Result.surfaces[j].border == NULL ? " (NULL)" : ""),
                    SubJobDataList[k]->gridX, SubJobDataList[k]->gridY, SubJobDataList[k]->gridZ,
                    k, l, SubJobDataList[k]->Result.surfaces[l].globalID,
                    (SubJobDataList[k]->Result.surfaces[l].border == NULL ? " (NULL)" : ""));
#endif /* ULTRADEBUG */
            }
        }
    }
    return false;
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

bool VoluMetricJob::doBordersTouch(BorderVoxelArray &border1, BorderVoxelArray &border2) {
    for (SIZE_T i = 0; i < border1.Count(); i++) {
        for (SIZE_T j = 0; j < border2.Count(); j++) {
            if (border1[i]->doesTouch(border2[j])) {
                return true;
            }
        }
    }
    return false;
}

//VISLIB_FORCEINLINE void VoluMetricJob::joinSurfaces(vislib::Array<vislib::Array<unsigned int> > &globalSurfaceIDs,
//                                 int i, int j, int k, int l) {
//    vislib::sys::Log::DefaultLog.WriteInfo("joined global IDs %u and %u", globalSurfaceIDs[i][j], globalSurfaceIDs[k][l]);
//    if (globalSurfaceIDs[k][l] < globalSurfaceIDs[i][j]) {
//        globalSurfaceIDs[i][j] = globalSurfaceIDs[k][l];
//    } else {
//        globalSurfaceIDs[k][l] = globalSurfaceIDs[i][j];
//    }
//}

VISLIB_FORCEINLINE void VoluMetricJob::joinSurfaces(int i, int j, int k, int l) {
#ifdef ULTRADEBUG
    vislib::sys::Log::DefaultLog.WriteInfo("joining global IDs (%u,%u,%u)[%u,%u][%u] and (%u,%u,%u)[%u,%u][%u]",
        SubJobDataList[i]->gridX, SubJobDataList[i]->gridY, SubJobDataList[i]->gridZ,
        i, j, SubJobDataList[i]->Result.surfaces[j].globalID,
        SubJobDataList[k]->gridX, SubJobDataList[k]->gridY, SubJobDataList[k]->gridZ,
        k, l, SubJobDataList[k]->Result.surfaces[l].globalID);
#endif ULTRADEBUG

    unsigned int src, dst;

    if (SubJobDataList[k]->Result.surfaces[l].globalID < SubJobDataList[i]->Result.surfaces[j].globalID) {
        //SubJobDataList[i]->Result.surfaces[j].globalID = SubJobDataList[k]->Result.surfaces[l].globalID;
        src = SubJobDataList[k]->Result.surfaces[l].globalID;
        dst = SubJobDataList[i]->Result.surfaces[j].globalID;
    } else {
        //subJobDataList[k]->Result.surfaces[l].globalID = subJobDataList[i]->Result.surfaces[j].globalID;
        src = SubJobDataList[i]->Result.surfaces[j].globalID;
        dst = SubJobDataList[k]->Result.surfaces[l].globalID;
    }

    for (int x = 0; x < SubJobDataList.Count(); x++) {
        if (SubJobDataList[x]->Result.done) {
            for (int y = 0; y < SubJobDataList[x]->Result.surfaces.Count(); y++) {
                if (SubJobDataList[x]->Result.surfaces[y].globalID == dst) {
                    SubJobDataList[x]->Result.surfaces[y].globalID = src;
                }
            }
        }
    }

#ifdef ULTRADEBUG
    vislib::sys::Log::DefaultLog.WriteInfo("joined global IDs (%u,%u,%u)[%u,%u][%u] and (%u,%u,%u)[%u,%u][%u]",
        SubJobDataList[i]->gridX, SubJobDataList[i]->gridY, SubJobDataList[i]->gridZ,
        i, j, SubJobDataList[i]->Result.surfaces[j].globalID,
        SubJobDataList[k]->gridX, SubJobDataList[k]->gridY, SubJobDataList[k]->gridZ,
        k, l, SubJobDataList[k]->Result.surfaces[l].globalID);
#endif
}

VISLIB_FORCEINLINE bool VoluMetricJob::isSurfaceJoinableWithSubvolume(SubJobData *surfJob, int surfIdx, SubJobData *volume) {
    for (int i = 0; i < 6; i++) {
        if (surfJob->Result.surfaces[surfIdx].fullFaces & (1 << i)) {
            // are they located accordingly to each other?
            int x = surfJob->offsetX + TetraVoxelizer::moreNeighbors[i].X() * (surfJob->resX - 1);
            int y = surfJob->offsetY + TetraVoxelizer::moreNeighbors[i].Y() * (surfJob->resY - 1);
            int z = surfJob->offsetZ + TetraVoxelizer::moreNeighbors[i].Z() * (surfJob->resZ - 1);
            if (volume->offsetX == x && volume->offsetY == y && volume->offsetZ == z) {
                return true;
            }
        }
    }
    return false;
}

void VoluMetricJob::generateStatistics(vislib::Array<unsigned int> &uniqueIDs,
                                       vislib::Array<SIZE_T> &countPerID,
                                       vislib::Array<VoxelizerFloat> &surfPerID,
                                       vislib::Array<VoxelizerFloat> &volPerID) {

    //globalSurfaceIDs.Clear();
    uniqueIDs.Clear();
    countPerID.Clear();
    surfPerID.Clear();
    volPerID.Clear();

    vislib::Array<unsigned int> todos;
    todos.SetCapacityIncrement(10);
    for (int i = 0; i < SubJobDataList.Count(); i++) {
        if (SubJobDataList[i]->Result.done) {
            todos.Add(i);
        }
    }

    if (todos.Count() == 0) {
        return;
    }

    //globalSurfaceIDs.SetCount(todos.Count());
    //unsigned int gsi = 0;
    for (int i = 0; i < todos.Count(); i++) {
        SIZE_T sc = SubJobDataList[todos[i]]->Result.surfaces.Count();
        //globalSurfaceIDs[i].SetCount(sc);
        for (int j = 0; j < sc; j++) {
            //globalSurfaceIDs[i][j] = gsi++;
            if (SubJobDataList[todos[i]]->Result.surfaces[j].globalID == UINT_MAX) {
                SubJobDataList[todos[i]]->Result.surfaces[j].globalID = this->MaxGlobalID++;
            }
        }
    }

    AccessGlobalID.Lock();
restart:
    for (int i = 0; i < todos.Count(); i++) {
        for (int j = 0; j < SubJobDataList[todos[i]]->Result.surfaces.Count(); j++) {
            for (int k = i; k < todos.Count(); k++) {
                vislib::math::Cuboid<VoxelizerFloat> c = SubJobDataList[todos[i]]->Bounds;
                c.Union(SubJobDataList[todos[k]]->Bounds);
                // are these neighbors or in the same subvolume?
                //if ((i == k) || (c.Volume() <= subJobDataList[todos[i]]->Bounds.Volume() 
                if ((todos[i] != todos[k]) &&(c.Volume() <= SubJobDataList[todos[i]]->Bounds.Volume() 
                                                + SubJobDataList[todos[k]]->Bounds.Volume())) {
                    for (int l = 0; l < SubJobDataList[todos[k]]->Result.surfaces.Count(); l++) {
                        if (areSurfacesJoinable(todos[i], j, todos[k], l)) {
                            joinSurfaces(todos[i], j, todos[k], l);
                            goto restart;
                        }
//                        //if (globalSurfaceIDs[k][l] != globalSurfaceIDs[i][j]) {
//                        if (subJobDataList[todos[i]]->Result.surfaces[j].globalID
//                                != subJobDataList[todos[k]]->Result.surfaces[l].globalID) {
//                            if (subJobDataList[todos[i]]->Result.surfaces[j].surface == 0.0
//                                    && subJobDataList[todos[k]]->Result.surfaces[l].surface == 0.0) {
//                                // both are full, can be joined trivially
//                                joinSurfaces(subJobDataList, todos[i], j, todos[k], l);
//                                // restart
//                                goto restart;
//                            } else if (subJobDataList[todos[i]]->Result.surfaces[j].surface == 0.0) {
//                                if (isSurfaceJoinableWithSubvolume(subJobDataList[todos[k]], l, 
//                                        subJobDataList[todos[i]])) {
//                                    joinSurfaces(subJobDataList, todos[i], j, todos[k], l);
//                                    // restart
//                                    goto restart;
//                                }
//                            } else if (subJobDataList[todos[k]]->Result.surfaces[l].surface == 0.0) {
//                                if (isSurfaceJoinableWithSubvolume(subJobDataList[todos[i]], j, 
//                                        subJobDataList[todos[k]])) {
//                                    joinSurfaces(subJobDataList, todos[i], j, todos[k], l);
//                                    // restart
//                                    goto restart;
//                                }
//                            } else {
//                                if (subJobDataList[todos[i]]->Result.surfaces[j].border != NULL &&
//                                    subJobDataList[todos[k]]->Result.surfaces[l].border != NULL) {
//                                    if (doBordersTouch(*subJobDataList[todos[i]]->Result.surfaces[j].border,
//                                            *subJobDataList[todos[k]]->Result.surfaces[l].border)) {
//                                        joinSurfaces(subJobDataList, todos[i], j, todos[k], l);
//                                        // restart
//                                        goto restart;
//                                    }
//                                } else {
//                                    //ASSERT(false);
//#ifdef ULTRADEBUG
//                                    vislib::sys::Log::DefaultLog.WriteInfo("tried to compare (%u,%u,%u)[%u,%u][%u]%s with (%u,%u,%u)[%u,%u][%u]%s",
//                                        subJobDataList[todos[i]]->gridX, subJobDataList[todos[i]]->gridY, subJobDataList[todos[i]]->gridZ,
//                                        todos[i], j, subJobDataList[todos[i]]->Result.surfaces[j].globalID,
//                                        (subJobDataList[todos[i]]->Result.surfaces[j].border == NULL ? " (NULL)" : ""),
//                                        subJobDataList[todos[k]]->gridX, subJobDataList[todos[k]]->gridY, subJobDataList[todos[k]]->gridZ,
//                                        todos[k], l, subJobDataList[todos[k]]->Result.surfaces[l].globalID,
//                                        (subJobDataList[todos[k]]->Result.surfaces[l].border == NULL ? " (NULL)" : ""));
//#endif /* ULTRADEBUG */
//                                }
//                            }
//                        }
                    }
                }
            }
        }
    }
    AccessGlobalID.Unlock();

    for (int i = 0; i < todos.Count(); i++) {
        for (int j = 0; j < SubJobDataList[todos[i]]->Result.surfaces.Count(); j++) {
            if (SubJobDataList[todos[i]]->Result.surfaces[j].border != NULL
                && SubJobDataList[todos[i]]->Result.surfaces[j].border->Count() > 0) {
                // TODO: destroy border geometry in cells ALL of whose neighbors are already processed.
                int numProcessed = 0;

                for (int k = 0; k < 6; k++) {
                    int x = SubJobDataList[todos[i]]->gridX + TetraVoxelizer::moreNeighbors[k].X();
                    int y = SubJobDataList[todos[i]]->gridY + TetraVoxelizer::moreNeighbors[k].Y();
                    int z = SubJobDataList[todos[i]]->gridZ + TetraVoxelizer::moreNeighbors[k].Z();
                    if (x == -1 || y == -1 || z == -1 || x >= divX || y >= divY || z >= divZ) {
                        numProcessed++;
                    } else {
                        for (int l = 0; l < todos.Count(); l++) {
                            if (SubJobDataList[todos[l]]->gridX == x
                                    && SubJobDataList[todos[l]]->gridY == y
                                    && SubJobDataList[todos[l]]->gridZ == z) {
                                ASSERT(SubJobDataList[todos[l]]->Result.done);
                                numProcessed++;
                                break;
                            }
                        }
                    }
                }
                if (numProcessed == 6) {
                    SubJobDataList[todos[i]]->Result.surfaces[j].border = NULL;//->Clear();
#ifdef ULTRADEBUG
                    vislib::sys::Log::DefaultLog.WriteInfo("deleted border of (%u,%u,%u)[%u,%u][%u]",
                        SubJobDataList[todos[i]]->gridX, SubJobDataList[todos[i]]->gridY, SubJobDataList[todos[i]]->gridZ,
                        todos[i], j, SubJobDataList[todos[i]]->Result.surfaces[j].globalID);
#endif /* ULTRADEBUG */
                }
            }
        }
    }

    for (int i = 0; i < todos.Count(); i++) {
        for (int j = 0; j < SubJobDataList[todos[i]]->Result.surfaces.Count(); j++) {
            SIZE_T pos = uniqueIDs.IndexOf(SubJobDataList[todos[i]]->Result.surfaces[j].globalID);
            if (pos == vislib::Array<unsigned int>::INVALID_POS) {
                uniqueIDs.Add(SubJobDataList[todos[i]]->Result.surfaces[j].globalID);
                countPerID.Add(SubJobDataList[todos[i]]->Result.surfaces[j].mesh.Count() / 9);
                surfPerID.Add(SubJobDataList[todos[i]]->Result.surfaces[j].surface);
                volPerID.Add(SubJobDataList[todos[i]]->Result.surfaces[j].volume);
            } else {
                countPerID[pos] = countPerID[pos] + (SubJobDataList[todos[i]]->Result.surfaces[j].mesh.Count() / 9);
                surfPerID[pos] = surfPerID[pos] + SubJobDataList[todos[i]]->Result.surfaces[j].surface;
                volPerID[pos] = volPerID[pos] + SubJobDataList[todos[i]]->Result.surfaces[j].volume;
            }
        }
    }
}

void VoluMetricJob::outputStatistics(unsigned int frameNumber,
                                     vislib::Array<unsigned int> &uniqueIDs,
                                     vislib::Array<SIZE_T> &countPerID,
                                     vislib::Array<VoxelizerFloat> &surfPerID,
                                     vislib::Array<VoxelizerFloat> &volPerID) {
    //SIZE_T numTriangles = 0;
    for (int i = 0; i < uniqueIDs.Count(); i++) {
        //numTriangles += countPerID[i];
        vislib::sys::Log::DefaultLog.WriteInfo("surface %u: %u triangles, surface %f, volume %f", uniqueIDs[i],
            countPerID[i], surfPerID[i], volPerID[i]);
        if (!metricsFilenameSlot.Param<core::param::FilePathParam>()->Value().IsEmpty()) {
            vislib::sys::WriteFormattedLineToFile(this->statisticsFile, "%u\t%u\t%u\t%f\t%f\n",
                frameNumber, uniqueIDs[i], countPerID[i], surfPerID[i], volPerID[i]);
        }
    }
}


void VoluMetricJob::copyMeshesToBackbuffer(vislib::Array<unsigned int> &uniqueIDs) {
    // copy finished meshes to output
    SIZE_T numTriangles = 0;
    vislib::Array<SIZE_T> todos;
    todos.SetCapacityIncrement(10);
    for (SIZE_T i = 0; i < SubJobDataList.Count(); i++) {
        if (SubJobDataList[i]->storeMesh && SubJobDataList[i]->Result.done) {
            todos.Add(i);
            for (SIZE_T j = 0; j < SubJobDataList[i]->Result.surfaces.Count(); j++) {
                numTriangles += SubJobDataList[i]->Result.surfaces[j].mesh.Count() / 9;
            }
        }
    }
    if (todos.Count() == 0) {
        return;
    }

    VoxelizerFloat *vert, *norm;
    unsigned char *col;

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
                for (int k = 0; k < SubJobDataList[todos[j]]->Result.surfaces.Count(); k++) {
                    if (SubJobDataList[todos[j]]->Result.surfaces[k].globalID == uniqueIDs[i]) {
                        for (SIZE_T l = 0; l < SubJobDataList[todos[j]]->Result.surfaces[k].border->Count(); l++) {
                            SIZE_T vertCount = (*SubJobDataList[todos[j]]->Result.surfaces[k].border)[l]->triangles.Count() / 3;
                            memcpy(&(vert[vertOffset]), (*SubJobDataList[todos[j]]->Result.surfaces[k].border)[l]->triangles.PeekElements(),
                                vertCount * 3 * sizeof(VoxelizerFloat));
                            for (SIZE_T m = 0; m < vertCount; m++) {
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
                for (int k = 0; k < SubJobDataList[todos[j]]->Result.surfaces.Count(); k++) {
                    if (SubJobDataList[todos[j]]->Result.surfaces[k].globalID == uniqueIDs[i]) {
                        SIZE_T vertCount = SubJobDataList[todos[j]]->Result.surfaces[k].mesh.Count() / 3;
                        memcpy(&(vert[vertOffset]), SubJobDataList[todos[j]]->Result.surfaces[k].mesh.PeekElements(),
                             vertCount * 3 * sizeof(VoxelizerFloat));
                        for (SIZE_T l = 0; l < vertCount; l++) {
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
