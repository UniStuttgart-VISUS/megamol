/*
 * VoluMetricJob.cpp
 *
 * Copyright (C) 2010 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */
#include "VoluMetricJob.h"
#include "TetraVoxelizer.h"
#include "geometry_calls/MultiParticleDataCall.h"
#include "mmcore/param/BoolParam.h"
#include "mmcore/param/FilePathParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/IntParam.h"
#include "mmcore/utility/log/Log.h"
#include "trisoup/volumetrics/MarchingCubeTables.h"
#include "vislib/graphics/NamedColours.h"
#include "vislib/math/ShallowPoint.h"
#include "vislib/math/ShallowShallowTriangle.h"
#include "vislib/math/Vector.h"
#include "vislib/sys/ConsoleProgressBar.h"
#include "vislib/sys/SystemInformation.h"
#include "vislib/sys/Thread.h"
#include "vislib/sys/ThreadPool.h"
#include "vislib/sys/sysfunctions.h"
#include <cfloat>
#include <climits>

using namespace megamol::trisoup_gl::volumetrics;

/*
 * VoluMetricJob::VoluMetricJob
 */
VoluMetricJob::VoluMetricJob()
        : core::job::AbstractThreadedJob()
        , core::Module()
        , getDataSlot("getData", "Slot that connects to a MultiParticleDataCall to fetch the particles in the scene")
        , metricsFilenameSlot("metricsFilenameSlot", "File that will contain the "
                                                     "surfaces and volumes of each particle list per frame")
        , showBorderGeometrySlot("showBorderGeometrySlot",
              "toggle whether the the surface triangles will be replaced by the border triangles")
        , showBoundingBoxesSlot("showBoundingBoxesSlot", "toggle whether the job subdivision grid will be shown")
        , showSurfaceGeometrySlot("showSurfaceGeometrySlot", "toggle whether the the surface triangles will be shown")
        , radiusMultiplierSlot("radiusMultiplierSlot", "multiplier for the particle radius")
        , continueToNextFrameSlot("continueToNextFrameSlot", "continue computation immediately after a frame finishes,"
                                                             "erasing all debug geometry")
        , resetContinueSlot("resetContinueSlot", "reset the continueToNextFrameSlot to false automatically")
        , outLineDataSlot("outLineData", "Slot that outputs debug line geometry")
        , outTriDataSlot("outTriData", "Slot that outputs debug triangle geometry")
        , outVolDataSlot("outVolData", "Slot that outputs debug volume data")
        , cellSizeRatioSlot("cellSizeRatioSlot", "Fraction of the minimal particle radius that is used as cell size")
        , subVolumeResolutionSlot(
              "subVolumeResolutionSlot", "maximum edge length of a subvolume processed as a separate job")
        , MaxRad(0)
        , backBufferIndex(0)
        , meshBackBufferIndex(0)
        , hash(0) {

    this->getDataSlot.SetCompatibleCall<geocalls::MultiParticleDataCallDescription>();
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

    this->outVolDataSlot.SetCallback("CallVolumetricData", "GetData", &VoluMetricJob::getVolDataCallback);
    this->outVolDataSlot.SetCallback("CallVolumetricData", "GetExtent", &VoluMetricJob::getLineExtentCallback);
    this->MakeSlotAvailable(&this->outVolDataSlot);

    this->globalIdBoxes.SetCapacityIncrement(100); // thomasmbm
}


/*
 * VoluMetricJob::~VoluMetricJob
 */
VoluMetricJob::~VoluMetricJob() {
    this->Release();
}


/*
 * VoluMetricJob::create
 */
bool VoluMetricJob::create() {

    // Intentionally empty

    return true;
}


/*
 * VoluMetricJob::release
 */
void VoluMetricJob::release() {

    // TODO: Implement
}


/*
 * VoluMetricJob::Run
 */
DWORD VoluMetricJob::Run(void* userData) {
    using megamol::core::utility::log::Log;

    geocalls::MultiParticleDataCall* datacall = this->getDataSlot.CallAs<geocalls::MultiParticleDataCall>();
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

    if (!metricsFilenameSlot.Param<core::param::FilePathParam>()->Value().empty()) {
        if (!this->statisticsFile.Open(
                metricsFilenameSlot.Param<core::param::FilePathParam>()->Value().native().c_str(),
                vislib::sys::File::WRITE_ONLY, vislib::sys::File::SHARE_READ, vislib::sys::File::CREATE_OVERWRITE)) {
            Log::DefaultLog.WriteError("Could not open statistics file for writing");
            return -3;
        }
    }

    vislib::Array<TetraVoxelizer*> voxelizerList;

    voxelizerList.SetCapacityIncrement(16);
    SubJobDataList.SetCapacityIncrement(16);

    for (unsigned int frameI = 0; frameI < frameCnt; frameI++) {

        vislib::sys::ThreadPool pool;
        //pool.SetThreadCount(static_cast<SIZE_T>(vislib::sys::SystemInformation::ProcessorCount() * 1.5));

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
            trisoup::volumetrics::VoxelizerFloat r = datacall->AccessParticles(partListI).GetGlobalRadius();
            if (r > MaxRad) {
                MaxRad = r;
            }
            if (r < MinRad) {
                MinRad = r;
            }
            if (datacall->AccessParticles(partListI).GetVertexDataType() ==
                geocalls::MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZR) {
                UINT64 numParticles = datacall->AccessParticles(partListI).GetCount();
                unsigned int stride = datacall->AccessParticles(partListI).GetVertexDataStride();
                unsigned char* vertexData = (unsigned char*)datacall->AccessParticles(partListI).GetVertexData();
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

        trisoup::volumetrics::VoxelizerFloat RadMult =
            this->radiusMultiplierSlot.Param<megamol::core::param::FloatParam>()->Value();
        MaxRad *= RadMult;
        MinRad *= RadMult;
        trisoup::volumetrics::VoxelizerFloat cellSize =
            MinRad * this->cellSizeRatioSlot.Param<megamol::core::param::FloatParam>()->Value();
        int bboxBytes = 8 * 3 * sizeof(trisoup::volumetrics::VoxelizerFloat);
        int bboxIdxes = 12 * 2 * sizeof(unsigned int);
        int vertSize = bboxBytes * partListCnt;
        int idxSize = bboxIdxes * partListCnt;
        bboxVertData[backBufferIndex].AssertSize(vertSize);
        bboxIdxData[backBufferIndex].AssertSize(idxSize);
        SIZE_T bboxOffset = 0;
        unsigned int vertFloatSize = 0;
        unsigned int idxNumOffset = 0;

        vislib::math::Cuboid<trisoup::volumetrics::VoxelizerFloat> b;
        if (datacall->AccessBoundingBoxes().IsObjectSpaceClipBoxValid()) {
            b = datacall->AccessBoundingBoxes().ObjectSpaceClipBox();
        } else {
            b = datacall->AccessBoundingBoxes().ObjectSpaceBBox();
        }

        int subVolCells = this->subVolumeResolutionSlot.Param<megamol::core::param::IntParam>()->Value();
#if 1 // ndef _DEBUG
        int resX = (int)((trisoup::volumetrics::VoxelizerFloat)b.Width() / cellSize) + 2;
        int resY = (int)((trisoup::volumetrics::VoxelizerFloat)b.Height() / cellSize) + 2;
        int resZ = (int)((trisoup::volumetrics::VoxelizerFloat)b.Depth() / cellSize) + 2;
        b.SetWidth(resX * cellSize);
        b.SetHeight(resY * cellSize);
        b.SetDepth(resZ * cellSize);

        appendBox(bboxVertData[backBufferIndex], b, bboxOffset);
        appendBoxIndices(bboxIdxData[backBufferIndex], idxNumOffset);

        divX = 1;
        divY = 1;
        divZ = 1;

        while (divX == 1 && divY == 1 && divZ == 1) {
            subVolCells /= 2;
            divX = (int)ceil((trisoup::volumetrics::VoxelizerFloat)resX / subVolCells);
            divY = (int)ceil((trisoup::volumetrics::VoxelizerFloat)resY / subVolCells);
            divZ = (int)ceil((trisoup::volumetrics::VoxelizerFloat)resZ / subVolCells);
        }
#else
        divX = 1;
        divY = 1;
        divZ = 1;

        int resX = subVolCells + 2;
        int resY = subVolCells + 2;
        int resZ = subVolCells + 2;
        cellSize = (trisoup::volumetrics::VoxelizerFloat)b.Width() / subVolCells /*resX*/;
#endif

        vertSize += bboxBytes * divX * divY * divZ;
        idxSize += bboxIdxes * divX * divY * divZ;
        bboxVertData[backBufferIndex].AssertSize(vertSize, true);
        bboxIdxData[backBufferIndex].AssertSize(idxSize, true);

        bool storeMesh = (this->outTriDataSlot.GetStatus() == megamol::core::AbstractSlot::STATUS_CONNECTED);
        bool storeVolume = //storeMesh; // debug for now ...
            (this->outVolDataSlot.GetStatus() == megamol::core::AbstractSlot::STATUS_CONNECTED);

        vislib::sys::ConsoleProgressBar pb;
        pb.Start("Computing Frame", divX * divY * divZ);

        megamol::core::utility::log::Log::DefaultLog.WriteInfo("Grid: %ux%ux%u", divX, divY, divZ);
        for (int x = 0; x < divX; x++) {
            for (int y = 0; y < divY; y++) {
                //for (int y = 0; y < 1; y++) {
                for (int z = 0; z < divZ; z++) {
                    //for (int z = 0; z < 1; z++) {
                    trisoup::volumetrics::VoxelizerFloat left = b.Left() + x * subVolCells * cellSize;
                    int restX = resX - x * subVolCells;
                    restX = (restX > subVolCells) ? subVolCells + 1 : restX;
                    trisoup::volumetrics::VoxelizerFloat right = left + restX * cellSize;
                    trisoup::volumetrics::VoxelizerFloat bottom = b.Bottom() + y * subVolCells * cellSize;
                    int restY = resY - y * subVolCells;
                    restY = (restY > subVolCells) ? subVolCells + 1 : restY;
                    trisoup::volumetrics::VoxelizerFloat top = bottom + restY * cellSize;
                    trisoup::volumetrics::VoxelizerFloat back = b.Back() + z * subVolCells * cellSize;
                    int restZ = resZ - z * subVolCells;
                    restZ = (restZ > subVolCells) ? subVolCells + 1 : restZ;
                    trisoup::volumetrics::VoxelizerFloat front = back + restZ * cellSize;
                    vislib::math::Cuboid<trisoup::volumetrics::VoxelizerFloat> bx =
                        vislib::math::Cuboid<trisoup::volumetrics::VoxelizerFloat>(
                            left, bottom, back, right, top, front);
                    appendBox(bboxVertData[backBufferIndex], bx, bboxOffset);
                    appendBoxIndices(bboxIdxData[backBufferIndex], idxNumOffset);

                    trisoup::volumetrics::SubJobData* sjd = new trisoup::volumetrics::SubJobData();
                    sjd->parent = this;
                    sjd->datacall = datacall;
                    sjd->Bounds = bx;
                    sjd->CellSize = (trisoup::volumetrics::VoxelizerFloat)MinRad *
                                    this->cellSizeRatioSlot.Param<megamol::core::param::FloatParam>()->Value();
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
                    sjd->storeVolume = storeVolume;
                    SubJobDataList.Add(sjd);
                    TetraVoxelizer* v = new TetraVoxelizer();
                    voxelizerList.Add(v);

                    //if (z == 0 && y == 0) {
                    pool.QueueUserWorkItem(v, sjd);
                    //}
                }
            }
        }
        //}
        this->debugLines[backBufferIndex][0].Set(static_cast<unsigned int>(idxNumOffset * 2),
            this->bboxIdxData[backBufferIndex].As<unsigned int>(),
            this->bboxVertData[backBufferIndex].As<trisoup::volumetrics::VoxelizerFloat>(),
            vislib::graphics::NamedColours::BlanchedAlmond);

        backBufferIndex = 1 - backBufferIndex;
        this->hash++;

        vislib::Array<vislib::Array<unsigned int>> globalSurfaceIDs;
        vislib::Array<unsigned int> uniqueIDs;
        vislib::Array<SIZE_T> countPerID;
        vislib::Array<trisoup::volumetrics::VoxelizerFloat> surfPerID;
        vislib::Array<trisoup::volumetrics::VoxelizerFloat> volPerID;
        vislib::Array<trisoup::volumetrics::VoxelizerFloat> voidVolPerID;

        SIZE_T lastCount = pool.CountUserWorkItems();
        while (1) {
            if (pool.Wait(500) && pool.CountUserWorkItems() == 0) {
                // we are done
                break;
            }
            if (lastCount != pool.CountUserWorkItems()) {
                pb.Set(
                    static_cast<vislib::sys::ConsoleProgressBar::Size>(divX * divY * divZ - pool.CountUserWorkItems()));
                generateStatistics(uniqueIDs, countPerID, surfPerID, volPerID, voidVolPerID);
                if (storeMesh)
                    copyMeshesToBackbuffer(uniqueIDs);
                if (storeVolume)
                    copyVolumesToBackBuffer();
                lastCount = pool.CountUserWorkItems();
            }
        }
        generateStatistics(uniqueIDs, countPerID, surfPerID, volPerID, voidVolPerID);
        outputStatistics(frameI, uniqueIDs, countPerID, surfPerID, volPerID, voidVolPerID);
        if (storeMesh)
            copyMeshesToBackbuffer(uniqueIDs);
        if (storeVolume)
            copyVolumesToBackBuffer();
        pb.Stop();
        Log::DefaultLog.WriteInfo("Done marching.");
        pool.Terminate(true);

        while (!this->continueToNextFrameSlot.Param<megamol::core::param::BoolParam>()->Value()) {
            vislib::sys::Thread::Sleep(500);
        }
        if (this->resetContinueSlot.Param<megamol::core::param::BoolParam>()->Value()) {
            this->continueToNextFrameSlot.Param<megamol::core::param::BoolParam>()->SetValue(false);
        }

        // new code to eliminate enclosed surfaces
    }

    if (!metricsFilenameSlot.Param<core::param::FilePathParam>()->Value().empty()) {
        statisticsFile.Close();
    }
    return 0;
}

bool VoluMetricJob::getLineDataCallback(core::Call& caller) {
    megamol::geocalls::LinesDataCall* ldc = dynamic_cast<megamol::geocalls::LinesDataCall*>(&caller);
    if (ldc == NULL)
        return false;

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

bool VoluMetricJob::getTriDataCallback(core::Call& caller) {
    megamol::geocalls_gl::CallTriMeshDataGL* mdc = dynamic_cast<megamol::geocalls_gl::CallTriMeshDataGL*>(&caller);
    if (mdc == NULL)
        return false;

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

bool VoluMetricJob::getVolDataCallback(core::Call& caller) {
    trisoup::trisoupVolumetricDataCall* dataCall = dynamic_cast<trisoup::trisoupVolumetricDataCall*>(&caller);

    if (dataCall == NULL)
        return false;

    if (this->hash == 0) {
        dataCall->SetVolumes(this->debugVolumes);
        dataCall->SetDataHash(0);
    } else {
        if (this->showSurfaceGeometrySlot.Param<megamol::core::param::BoolParam>()->Value()) {
            dataCall->SetVolumes(this->debugVolumes);
        } else {
            dataCall->SetVolumes(this->debugVolumes);
        }
        dataCall->SetDataHash(this->hash);
    }

    return true;
}

bool VoluMetricJob::getLineExtentCallback(core::Call& caller) {
    core::AbstractGetData3DCall* ldc = dynamic_cast<core::AbstractGetData3DCall*>(&caller);
    if (ldc == NULL)
        return false;

    geocalls::MultiParticleDataCall* datacall = this->getDataSlot.CallAs<geocalls::MultiParticleDataCall>();
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


bool VoluMetricJob::areSurfacesJoinable(int sjdIdx1, int surfIdx1, int sjdIdx2, int surfIdx2) {

    if (SubJobDataList[sjdIdx1]->Result.surfaces[surfIdx1].globalID !=
        SubJobDataList[sjdIdx2]->Result.surfaces[surfIdx2].globalID) {
        if (SubJobDataList[sjdIdx1]->Result.surfaces[surfIdx1].surface == 0.0 &&
            SubJobDataList[sjdIdx2]->Result.surfaces[surfIdx2].surface == 0.0) {
            // both are full, can be joined trivially
            return true;
        } else if (SubJobDataList[sjdIdx1]->Result.surfaces[surfIdx1].surface == 0.0) {
            if (isSurfaceJoinableWithSubvolume(SubJobDataList[sjdIdx2], surfIdx2, SubJobDataList[sjdIdx1])) {
                return true;
            }
        } else if (SubJobDataList[sjdIdx2]->Result.surfaces[surfIdx2].surface == 0.0) {
            if (isSurfaceJoinableWithSubvolume(SubJobDataList[sjdIdx1], surfIdx1, SubJobDataList[sjdIdx2])) {
                return true;
            }
        } else {
            if (SubJobDataList[sjdIdx1]->Result.surfaces[surfIdx1].border != NULL &&
                SubJobDataList[sjdIdx2]->Result.surfaces[surfIdx2].border != NULL) {
                if (doBordersTouch(*SubJobDataList[sjdIdx1]->Result.surfaces[surfIdx1].border,
                        *SubJobDataList[sjdIdx2]->Result.surfaces[surfIdx2].border)) {
                    return true;
                }
            } else {
                //ASSERT(false);
#ifdef ULTRADEBUG
                megamol::core::utility::log::Log::DefaultLog.WriteInfo(
                    "tried to compare (%u,%u,%u)[%u,%u][%u]%s with (%u,%u,%u)[%u,%u][%u]%s",
                    SubJobDataList[sjdIdx1]->gridX, SubJobDataList[sjdIdx1]->gridY, SubJobDataList[sjdIdx1]->gridZ,
                    sjdIdx1, surfIdx1, SubJobDataList[sjdIdx1]->Result.surfaces[surfIdx1].globalID,
                    (SubJobDataList[sjdIdx1]->Result.surfaces[surfIdx1].border == NULL ? " (NULL)" : ""),
                    SubJobDataList[sjdIdx2]->gridX, SubJobDataList[sjdIdx2]->gridY, SubJobDataList[sjdIdx2]->gridZ,
                    sjdIdx2, surfIdx2, SubJobDataList[sjdIdx2]->Result.surfaces[surfIdx2].globalID,
                    (SubJobDataList[sjdIdx2]->Result.surfaces[surfIdx2].border == NULL ? " (NULL)" : ""));
#endif /* ULTRADEBUG */
            }
        }
    }
    return false;
}


void VoluMetricJob::appendBox(
    vislib::RawStorage& data, vislib::math::Cuboid<trisoup::volumetrics::VoxelizerFloat>& b, SIZE_T& offset) {
    vislib::math::ShallowPoint<trisoup::volumetrics::VoxelizerFloat, 3>(data.AsAt<trisoup::volumetrics::VoxelizerFloat>(
        offset + 0 * 3 * sizeof(trisoup::volumetrics::VoxelizerFloat))) = b.GetLeftBottomFront();
    vislib::math::ShallowPoint<trisoup::volumetrics::VoxelizerFloat, 3>(data.AsAt<trisoup::volumetrics::VoxelizerFloat>(
        offset + 1 * 3 * sizeof(trisoup::volumetrics::VoxelizerFloat))) = b.GetRightBottomFront();
    vislib::math::ShallowPoint<trisoup::volumetrics::VoxelizerFloat, 3>(data.AsAt<trisoup::volumetrics::VoxelizerFloat>(
        offset + 2 * 3 * sizeof(trisoup::volumetrics::VoxelizerFloat))) = b.GetRightTopFront();
    vislib::math::ShallowPoint<trisoup::volumetrics::VoxelizerFloat, 3>(data.AsAt<trisoup::volumetrics::VoxelizerFloat>(
        offset + 3 * 3 * sizeof(trisoup::volumetrics::VoxelizerFloat))) = b.GetLeftTopFront();
    vislib::math::ShallowPoint<trisoup::volumetrics::VoxelizerFloat, 3>(data.AsAt<trisoup::volumetrics::VoxelizerFloat>(
        offset + 4 * 3 * sizeof(trisoup::volumetrics::VoxelizerFloat))) = b.GetLeftBottomBack();
    vislib::math::ShallowPoint<trisoup::volumetrics::VoxelizerFloat, 3>(data.AsAt<trisoup::volumetrics::VoxelizerFloat>(
        offset + 5 * 3 * sizeof(trisoup::volumetrics::VoxelizerFloat))) = b.GetRightBottomBack();
    vislib::math::ShallowPoint<trisoup::volumetrics::VoxelizerFloat, 3>(data.AsAt<trisoup::volumetrics::VoxelizerFloat>(
        offset + 6 * 3 * sizeof(trisoup::volumetrics::VoxelizerFloat))) = b.GetRightTopBack();
    vislib::math::ShallowPoint<trisoup::volumetrics::VoxelizerFloat, 3>(data.AsAt<trisoup::volumetrics::VoxelizerFloat>(
        offset + 7 * 3 * sizeof(trisoup::volumetrics::VoxelizerFloat))) = b.GetLeftTopBack();
    //return 8 * 3 * sizeof(float) + offset;
    offset += 8 * 3 * sizeof(trisoup::volumetrics::VoxelizerFloat);
}

void VoluMetricJob::appendBoxIndices(vislib::RawStorage& data, unsigned int& numOffset) {
    for (int i = 0; i < 12; i++) {
        *data.AsAt<unsigned int>((2 * (numOffset + i) + 0) * sizeof(unsigned int)) =
            trisoup::volumetrics::MarchingCubeTables::a2iEdgeConnection[i][0] + numOffset * 8 / 12;
        *data.AsAt<unsigned int>((2 * (numOffset + i) + 1) * sizeof(unsigned int)) =
            trisoup::volumetrics::MarchingCubeTables::a2iEdgeConnection[i][1] + numOffset * 8 / 12;
    }
    //return 12 * 2 * sizeof(unsigned int) + offset;
    //rawOffset += 12 * 2 * sizeof(unsigned int);
    numOffset += 12;
}

bool VoluMetricJob::doBordersTouch(
    trisoup::volumetrics::BorderVoxelArray& border1, trisoup::volumetrics::BorderVoxelArray& border2) {
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
//    megamol::core::utility::log::Log::DefaultLog.WriteInfo("joined global IDs %u and %u", globalSurfaceIDs[i][j], globalSurfaceIDs[k][l]);
//    if (globalSurfaceIDs[k][l] < globalSurfaceIDs[i][j]) {
//        globalSurfaceIDs[i][j] = globalSurfaceIDs[k][l];
//    } else {
//        globalSurfaceIDs[k][l] = globalSurfaceIDs[i][j];
//    }
//}

VISLIB_FORCEINLINE void VoluMetricJob::joinSurfaces(int sjdIdx1, int surfIdx1, int sjdIdx2, int surfIdx2) {

    RewriteGlobalID.Lock();

#ifdef ULTRADEBUG
    megamol::core::utility::log::Log::DefaultLog.WriteInfo(
        "joining global IDs (%u,%u,%u)[%u,%u][%u] and (%u,%u,%u)[%u,%u][%u]", SubJobDataList[sjdIdx1]->gridX,
        SubJobDataList[sjdIdx1]->gridY, SubJobDataList[sjdIdx1]->gridZ, sjdIdx1, surfIdx1,
        SubJobDataList[sjdIdx1]->Result.surfaces[surfIdx1].globalID, SubJobDataList[sjdIdx2]->gridX,
        SubJobDataList[sjdIdx2]->gridY, SubJobDataList[sjdIdx2]->gridZ, sjdIdx2, surfIdx2,
        SubJobDataList[sjdIdx2]->Result.surfaces[surfIdx2].globalID);
#endif ULTRADEBUG
    trisoup::volumetrics::Surface *srcSurf, *dstSurf;

    if (SubJobDataList[sjdIdx2]->Result.surfaces[surfIdx2].globalID <
        SubJobDataList[sjdIdx1]->Result.surfaces[surfIdx1].globalID) {
        //SubJobDataList[sjdIdx1]->Result.surfaces[surfIdx1].globalID = SubJobDataList[sjdIdx2]->Result.surfaces[surfIdx2].globalID;
        srcSurf = &SubJobDataList[sjdIdx2]->Result.surfaces[surfIdx2];
        dstSurf = &SubJobDataList[sjdIdx1]->Result.surfaces[surfIdx1];
    } else {
        //subJobDataList[sjdIdx2]->Result.surfaces[surfIdx2].globalID = subJobDataList[sjdIdx1]->Result.surfaces[surfIdx1].globalID;
        srcSurf = &SubJobDataList[sjdIdx1]->Result.surfaces[surfIdx1];
        dstSurf = &SubJobDataList[sjdIdx2]->Result.surfaces[surfIdx2];
    }

#ifdef PARALLEL_BBOX_COLLECT // cs: RewriteGlobalID
    if (this->globalIdBoxes.Count() <= srcSurf->globalID)
        this->globalIdBoxes.SetCount(srcSurf->globalID + 1);
#endif

    for (unsigned int x = 0; x < SubJobDataList.Count(); x++) {
        //if (SubJobDataList[x]->Result.done) {
        for (unsigned int y = 0; y < SubJobDataList[x]->Result.surfaces.Count(); y++) {
            trisoup::volumetrics::Surface& surf = SubJobDataList[x]->Result.surfaces[y];
            if (surf.globalID == dstSurf->globalID) {
#ifdef PARALLEL_BBOX_COLLECT
                // thomasbm: gather global surface-bounding boxes
                this->globalIdBoxes[srcSurf->globalID].Union(srcSurf->boundingBox);
                this->globalIdBoxes[srcSurf->globalID].Union(surf.boundingBox);
#endif
                surf.globalID = srcSurf->globalID;
            }
        }
        //}
    }
#ifdef ULTRADEBUG
    megamol::core::utility::log::Log::DefaultLog.WriteInfo(
        "joined global IDs (%u,%u,%u)[%u,%u][%u] and (%u,%u,%u)[%u,%u][%u]", SubJobDataList[sjdIdx1]->gridX,
        SubJobDataList[sjdIdx1]->gridY, SubJobDataList[sjdIdx1]->gridZ, sjdIdx1, surfIdx1,
        SubJobDataList[sjdIdx1]->Result.surfaces[surfIdx1].globalID, SubJobDataList[sjdIdx2]->gridX,
        SubJobDataList[sjdIdx2]->gridY, SubJobDataList[sjdIdx2]->gridZ, sjdIdx2, surfIdx2,
        SubJobDataList[sjdIdx2]->Result.surfaces[surfIdx2].globalID);
#endif
    RewriteGlobalID.Unlock();
}

VISLIB_FORCEINLINE bool VoluMetricJob::isSurfaceJoinableWithSubvolume(
    trisoup::volumetrics::SubJobData* surfJob, int surfIdx, trisoup::volumetrics::SubJobData* volume) {
    for (unsigned int i = 0; i < 6; i++) {
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

void VoluMetricJob::generateStatistics(vislib::Array<unsigned int>& uniqueIDs, vislib::Array<SIZE_T>& countPerID,
    vislib::Array<trisoup::volumetrics::VoxelizerFloat>& surfPerID,
    vislib::Array<trisoup::volumetrics::VoxelizerFloat>& volPerID,
    vislib::Array<trisoup::volumetrics::VoxelizerFloat>& voidVolPerID) {

    //globalSurfaceIDs.Clear();
    uniqueIDs.Clear();
    countPerID.Clear();
    surfPerID.Clear();
    volPerID.Clear();
    voidVolPerID.Clear();

    vislib::Array<unsigned int> todos;
    todos.SetCapacityIncrement(10);
    for (unsigned int i = 0; i < SubJobDataList.Count(); i++) {
        if (SubJobDataList[i]->Result.done) {
            todos.Add(i);
        }
    }

    if (todos.Count() == 0) {
        return;
    }

    //globalSurfaceIDs.SetCount(todos.Count());
    //unsigned int gsi = 0;
    for (unsigned int todoIdx = 0; todoIdx < todos.Count(); todoIdx++) {
        unsigned int todo = todos[todoIdx];
        trisoup::volumetrics::SubJobData* sjdTodo = this->SubJobDataList[todo];
        SIZE_T surfaceCount = sjdTodo->Result.surfaces.Count();
        //globalSurfaceIDs[todo].SetCount(surfaceCount);
        for (unsigned int surfIdx = 0; surfIdx < surfaceCount; surfIdx++) {
            trisoup::volumetrics::Surface& surf = sjdTodo->Result.surfaces[surfIdx];
            //globalSurfaceIDs[todo][surfIdx] = gsi++;
            if (surf.globalID == UINT_MAX) {
                AccessMaxGlobalID.Lock();
                if (surf.globalID == UINT_MAX) {
                    //surf.globalID = this->MaxGlobalID++;
                    throw new vislib::Exception("surface with no globalID encountered", __FILE__, __LINE__);
                }
                AccessMaxGlobalID.Unlock();
            }
        }
    }

restart:
    for (unsigned int todoIdx = 0; todoIdx < todos.Count(); todoIdx++) {
        unsigned int todo = todos[todoIdx];
        trisoup::volumetrics::SubJobData* sjdTodo = this->SubJobDataList[todo];
        for (unsigned int surfIdx = 0; surfIdx < sjdTodo->Result.surfaces.Count(); surfIdx++) {
            for (unsigned int todoIdx2 = todoIdx; todoIdx2 < todos.Count(); todoIdx2++) {
                unsigned int todo2 = todos[todoIdx2];
                trisoup::volumetrics::SubJobData* sjdTodo2 = this->SubJobDataList[todo2];
                vislib::math::Cuboid<trisoup::volumetrics::VoxelizerFloat> box = sjdTodo->Bounds;
                box.Union(sjdTodo2->Bounds);
                // are these neighbors or in the same subvolume?
                //if ((todoIdx == todoIdx2) || (c.Volume() <= sjdTodo->Bounds.Volume()
                if (todo != todo2 && (box.Volume() <= sjdTodo->Bounds.Volume() + sjdTodo2->Bounds.Volume())) {
                    for (unsigned int surfIdx2 = 0; surfIdx2 < sjdTodo2->Result.surfaces.Count(); surfIdx2++) {
                        if (areSurfacesJoinable(todo, surfIdx, todo2, surfIdx2)) {
                            joinSurfaces(todo, surfIdx, todo2, surfIdx2);
                            goto restart;
                        }
                    }
                }
            }
        }
    }

    for (unsigned int todoIdx = 0; todoIdx < todos.Count(); todoIdx++) {
        unsigned int todo = todos[todoIdx];
        trisoup::volumetrics::SubJobData* sjdTodo = this->SubJobDataList[todo];
        for (unsigned int surfIdx = 0; surfIdx < sjdTodo->Result.surfaces.Count(); surfIdx++) {
            trisoup::volumetrics::Surface& surf = sjdTodo->Result.surfaces[surfIdx];
            if (surf.border != NULL && surf.border->Count() > 0) {
                // TODO: destroy border geometry in cells ALL of whose neighbors are already processed.
                int numProcessed = 0;

                for (unsigned int neighbIdx = 0; neighbIdx < 6; neighbIdx++) {
                    int x = sjdTodo->gridX + TetraVoxelizer::moreNeighbors[neighbIdx].X();
                    int y = sjdTodo->gridY + TetraVoxelizer::moreNeighbors[neighbIdx].Y();
                    int z = sjdTodo->gridZ + TetraVoxelizer::moreNeighbors[neighbIdx].Z();
                    if (x == -1 || y == -1 || z == -1 || x >= divX || y >= divY || z >= divZ) {
                        numProcessed++;
                    } else {
                        for (unsigned int todoIdx2 = 0; todoIdx2 < todos.Count(); todoIdx2++) {
                            trisoup::volumetrics::SubJobData* sjdTodo2 = this->SubJobDataList[todos[todoIdx2]];
                            if (sjdTodo2->gridX == x && sjdTodo2->gridY == y && sjdTodo2->gridZ == z) {
                                ASSERT(sjdTodo2->Result.done);
                                numProcessed++;
                                break;
                            }
                        }
                    }
                }
                if (numProcessed == 6) {
                    surf.border = NULL; //->Clear();
#ifdef ULTRADEBUG
                    megamol::core::utility::log::Log::DefaultLog.WriteInfo("deleted border of (%u,%u,%u)[%u,%u][%u]",
                        sjdTodo->gridX, sjdTodo->gridY, sjdTodo->gridZ, todo, surfIdx, surf.globalID);
#endif /* ULTRADEBUG */
                }
            }
        }
    }

    for (unsigned int todoIdx = 0; todoIdx < todos.Count(); todoIdx++) {
        unsigned int todo = todos[todoIdx];
        trisoup::volumetrics::SubJobData* sjdTodo = this->SubJobDataList[todo];
        for (unsigned int surfIdx = 0; surfIdx < sjdTodo->Result.surfaces.Count(); surfIdx++) {
            trisoup::volumetrics::Surface& surf = sjdTodo->Result.surfaces[surfIdx];
            SIZE_T pos = uniqueIDs.IndexOf(surf.globalID);
            if (pos == vislib::Array<unsigned int>::INVALID_POS) {
                uniqueIDs.Add(surf.globalID);
                countPerID.Add(surf.mesh.Count() / 9);
                surfPerID.Add(surf.surface);
                volPerID.Add(surf.volume);
                voidVolPerID.Add(surf.voidVolume);
                //#ifndef PARALLEL_BBOX_COLLECT
                //                globalIdBoxes.Add(surf.boundingBox);
                //#endif
            } else {
                countPerID[pos] = countPerID[pos] + (surf.mesh.Count() / 9);
                surfPerID[pos] = surfPerID[pos] + surf.surface;
                volPerID[pos] = volPerID[pos] + surf.volume;
                voidVolPerID[pos] = voidVolPerID[pos] + surf.voidVolume;
                //#ifndef PARALLEL_BBOX_COLLECT
                //                globalIdBoxes[pos].Union(surf.boundingBox);
                //#endif
            }
        }
    }
}

/*
bool rayTriangleIntersect(ShallowShallowTriangle<trisoup::volumetrics::VoxelizerFloat,3>& triangle, ) {


    return true;
}
*/

/**
 * returns true if the line starting from 'seedPoint' in direction 'direction' hits the triangle-mesch 'mesh' and the hit point is returned by startPoint+direction*hitfactor.
 */
bool hitTriangleMesh(const vislib::Array<megamol::trisoup::volumetrics::VoxelizerFloat>& mesh,
    const vislib::math::ShallowPoint<megamol::trisoup::volumetrics::VoxelizerFloat, 3>& seedPoint,
    const vislib::math::Vector<megamol::trisoup::volumetrics::VoxelizerFloat, 3>& direction,
    megamol::trisoup::volumetrics::VoxelizerFloat* hitFactor) {

    unsigned int triCount = static_cast<unsigned int>(mesh.Count() / (3 * 3));
    const megamol::trisoup::volumetrics::VoxelizerFloat* triPoints = mesh.PeekElements();
    for (unsigned int triIdx = 0; triIdx < triCount; triIdx++) {
        vislib::math::ShallowShallowTriangle<megamol::trisoup::volumetrics::VoxelizerFloat, 3> triangle(
            const_cast<megamol::trisoup::volumetrics::VoxelizerFloat*>(triPoints + triIdx * 3 * 3));

        vislib::math::Vector<megamol::trisoup::volumetrics::VoxelizerFloat, 3> normal;
        vislib::math::Vector<megamol::trisoup::volumetrics::VoxelizerFloat, 3> w0(triangle[0] - seedPoint);

        triangle.Normal(normal);

        megamol::trisoup::volumetrics::VoxelizerFloat normDirDot = normal.Dot(direction);

        if (normDirDot < 0.9) // direction from "behind"? invalid/deformed triangle?
            continue;

        megamol::trisoup::volumetrics::VoxelizerFloat normW0Dot = normal.Dot(w0);

        megamol::trisoup::volumetrics::VoxelizerFloat intersectFactor =
            normW0Dot / normDirDot; // <triangle[0]-seedPoint,normal> / <direction,normal>
        if (intersectFactor < 0)    // triangle-plane lies behind the ray starting point (seedPoint)
            continue;

        // intersect point of ray with triangle plane
        vislib::math::Point<megamol::trisoup::volumetrics::VoxelizerFloat, 3> projectPoint =
            seedPoint + direction * intersectFactor;

        // test if the projected point on the triangle plane lies inside the triangle
        vislib::math::Vector<megamol::trisoup::volumetrics::VoxelizerFloat, 3> u(triangle[1] - triangle[0]);
        vislib::math::Vector<megamol::trisoup::volumetrics::VoxelizerFloat, 3> v(triangle[2] - triangle[0]);
        vislib::math::Vector<megamol::trisoup::volumetrics::VoxelizerFloat, 3> w(projectPoint - triangle[0]);
        //pointInsideTriangle(projectPoint, triangle, normal);

        // is I inside T?
        megamol::trisoup::volumetrics::VoxelizerFloat uu, uv, vv, wu, wv, D;
        uu = u.Dot(u);
        uv = u.Dot(v);
        vv = v.Dot(v);
        wu = w.Dot(u);
        wv = w.Dot(v);
        D = uv * uv - uu * vv;

        // get and test parametric coords
        megamol::trisoup::volumetrics::VoxelizerFloat s, t;
        s = (uv * wv - vv * wu) / D;
        if (s < 0 || s > 1) // I is outside T
            continue;
        t = (uv * wu - uu * wv) / D;
        if (t < 0 || (s + t) > 1) // I is outside T
            continue;

        // ray is inside triangle
        *hitFactor = intersectFactor;
        return true;
    }

    return false;
}

/**
 * TODO: sinnvolle erklaerung
 */
bool VoluMetricJob::testFullEnclosing(
    int enclosingIdx, int enclosedIdx, vislib::Array<vislib::Array<trisoup::volumetrics::Surface*>>& globaIdSurfaces) {
    // find a random enlosed surface and ise its first triangle as starting point
    trisoup::volumetrics::Surface* enclosedSeed = 0;
    /*  for(int sjdIdx = 0; sjdIdx < SubJobDataList.Count(); sjdIdx++) {
            SubJobData *subJob = SubJobDataList[sjdIdx];
            for (int surfIdx = 0; surfIdx < subJob->Result.surfaces.Count(); surfIdx++)
                if (subJob->Result.surfaces[surfIdx].globalID==enclosedIdx)
                    enclosedSeed = &subJob->Result.surfaces[surfIdx];
        }*/
    vislib::Array<trisoup::volumetrics::Surface*> enclosedSurfaces = globaIdSurfaces[enclosedIdx];
    vislib::Array<trisoup::volumetrics::Surface*> enclosingSurfaces = globaIdSurfaces[enclosingIdx];

    if (enclosedSurfaces.Count() > 0)
        enclosedSeed = enclosedSurfaces[0];

    if (!enclosedSeed)
        return false;

    ASSERT(enclosedSeed->mesh.Count() >= 3);

    vislib::math::ShallowPoint<trisoup::volumetrics::VoxelizerFloat, 3> seedPoint(
        /*enclosedSeed->mesh.PeekElements()*/ &enclosedSeed->mesh[0]);
    // some random direction ...
    trisoup::volumetrics::VoxelizerFloat tmpVal(
        /*static_cast<trisoup::volumetrics::VoxelizerFloat>(1.0 / sqrt(3.0))*/ 0.34524356);
    vislib::math::Vector<trisoup::volumetrics::VoxelizerFloat, 3> direction(tmpVal, 1.2 * tmpVal, -0.86 * tmpVal);
    direction.Normalise();

    int hitCount = 0;

    for (unsigned int enclosingSIdx = 0; enclosingSIdx < enclosingSurfaces.Count(); enclosingSIdx++) {
        trisoup::volumetrics::Surface* surf = enclosingSurfaces[enclosingSIdx];
        trisoup::volumetrics::VoxelizerFloat hitFactor;
        // TODO use surface bounding boxes to speed this up ...
        // if (surf->boundingBox.Intersect(seedPoint, direction, &hitFactor) && hitFactor > 0){
        if (hitTriangleMesh(surf->mesh, seedPoint, direction, &hitFactor) && hitFactor > 0) {
            hitCount++;
            //    seedPoint = trangleCenter(hitTriangle); // dazu müsste man ein "walkthrough" programmieren ?!
            // seedPoint += direction*hitFactor; // alternative
        }
        //}
    }


    /*    while(true) {
            SubJobData *sjd = getSubJobForPos(seedPoint);
            for(int surfIdx = 0; surfIdx < sjd->Result.surfaces.Count(); surfIdx++) {
                Surface& surf = sjd->Result.surfaces[surfIdx];
                if (surf.globalID != enclosingIdx)
                    continue;

                VoxelizerFloat hitFactor;
                // try bounding box hit first
                if (surf.boundingBox.Intersect(seedPoint, direction, &hitFactor) && hitFactor > 0) {
                    if(hitTriangleMesh(seedPoint, direction)) {
                        hitCount++;
                        seedPoint = trangleCenter(hitTriangle);
                    }
                }
            }
            if(hitCount==oldHitCount)
                gotoNextSubJob();
            if(borderReached())
                break
        }*/

    return (hitCount % 2) != 0;
}

void VoluMetricJob::outputStatistics(unsigned int frameNumber, vislib::Array<unsigned int>& uniqueIDs,
    vislib::Array<SIZE_T>& countPerID, vislib::Array<trisoup::volumetrics::VoxelizerFloat>& surfPerID,
    vislib::Array<trisoup::volumetrics::VoxelizerFloat>& volPerID,
    vislib::Array<trisoup::volumetrics::VoxelizerFloat>& voidVolPerID) {

#if 0
    VoxelizerFloat mesh[] = {
        1, 0, 0, 
        0, 1, 1,
        0, -1, 1
    };
    vislib::Array<trisoup::volumetrics::VoxelizerFloat> meshArray(9);
    for(int i = 0; i < 9; i++)
        meshArray.Add(mesh[i]);
    VoxelizerFloat seed[] = {0, 0, 0};
    vislib::math::ShallowPoint<trisoup::volumetrics::VoxelizerFloat,3> seedPoint(seed);
    vislib::math::Vector<trisoup::volumetrics::VoxelizerFloat,3> direction(-1/sqrt(2.0), 0, -1/sqrt(2.0));
    VoxelizerFloat hitFactor;
    hitTriangleMesh(meshArray, seedPoint, direction, &hitFactor);
#endif
    // thomasbm: testing ...
#ifndef PARALLEL_BBOX_COLLECT
    globalIdBoxes.SetCount(uniqueIDs.Count());
    vislib::Array<vislib::Array<trisoup::volumetrics::Surface*>>
        globaIdSurfaces /*(uniqueIDs.Count(), vislib::Array<Surface*>(10)?)*/;
    globaIdSurfaces.SetCount(uniqueIDs.Count());
    for (unsigned int sjdIdx = 0; sjdIdx < SubJobDataList.Count(); sjdIdx++) {
        trisoup::volumetrics::SubJobData* subJob = SubJobDataList[sjdIdx];
        for (unsigned int surfIdx = 0; surfIdx < subJob->Result.surfaces.Count(); surfIdx++) {
            trisoup::volumetrics::Surface& surface = subJob->Result.surfaces[surfIdx];
            int globalId = surface.globalID;
            int uniqueIdPos = static_cast<int>(uniqueIDs.IndexOf(globalId));
            globalIdBoxes[uniqueIdPos].Union(surface.boundingBox);
            globaIdSurfaces[uniqueIdPos].Add(&surface);
        }
    }
#endif

    // thomasbm: final step: find volumes of unique surface id's that contain each other
    for (unsigned int uidIdx = 0; uidIdx < uniqueIDs.Count(); uidIdx++) {
        unsigned int gid = uniqueIDs[uidIdx];

#ifndef PARALLEL_BBOX_COLLECT
        //if (globaIdSurfaces[uidIdx].Count() == 0)
        //    continue;
#endif

        for (unsigned int uidIdx2 = uidIdx + 1; uidIdx2 < uniqueIDs.Count(); uidIdx2++) {
            unsigned int gid2 = uniqueIDs[uidIdx2];
            if (/*uidIdx2==uidIdx*/ gid == gid2)
                continue;
#ifdef PARALLEL_BBOX_COLLECT
            BoundingBox<unsigned int>& box = globalIdBoxes[gid];
            BoundingBox<unsigned int>& box2 = globalIdBoxes[gid2];
#else
            //if (globaIdSurfaces[uidIdx2].Count() == 0)
            //    continue;
            trisoup::volumetrics::BoundingBox<unsigned int>& box = globalIdBoxes[uidIdx];
            trisoup::volumetrics::BoundingBox<unsigned int>& box2 = globalIdBoxes[uidIdx2];
            if (!box.IsInitialized() || !box2.IsInitialized())
                continue;
#endif
            trisoup::volumetrics::BoundingBox<unsigned int>::CLASSIFY_STATUS cls = box.Classify(box2);
            int enclosedIdx, enclosingIdx;
            if (cls == trisoup::volumetrics::BoundingBox<unsigned int>::CONTAINS_OTHER) {
                enclosedIdx = uidIdx2;
                enclosingIdx = uidIdx;
            } else if (cls == trisoup::volumetrics::BoundingBox<unsigned int>::IS_CONTAINED_BY_OTHER) {
                enclosedIdx = uidIdx;
                enclosingIdx = uidIdx2;
            } else
                continue;

#if 0 // thomasbm: das kann solange nicht funktionieren, wie die Dreiecke nicht richtig orientiert sind (innen/aussen)
                // full, triangle based test here ... (meshes need to be stored to do so)
            if(SubJobDataList[0]->storeMesh && !testFullEnclosing(enclosingIdx, enclosedIdx, globaIdSurfaces)) {
                continue;
            }
#endif

#ifdef _DEBUG
            megamol::core::utility::log::Log::DefaultLog.WriteInfo(
                "surface %d enclosed by %d:\n\tenclosed-volume: %f, enclosed-voidVol: %f, enclosed-sum: "
                "%f\n\tenclosing-volume: %f, enclosing-voidVol: %f",
                gid2, gid, volPerID[enclosedIdx], voidVolPerID[enclosedIdx],
                volPerID[enclosedIdx] + voidVolPerID[enclosedIdx], volPerID[enclosingIdx], voidVolPerID[enclosingIdx]);
#endif

            //countPerID[enclosingIdx] += countPerID[enclosedIdx];
            //surfPerID[enclosingIdx] += surfPerID[enclosedIdx];
            volPerID[enclosingIdx] += volPerID[enclosedIdx] + voidVolPerID[enclosedIdx];
            //countPerID[enclosedIdx] = 0;
            //surfPerID[enclosedIdx] = 0;
            volPerID[enclosedIdx] = 0;
        }
    }

    //SIZE_T numTriangles = 0;
    for (unsigned int i = 0; i < uniqueIDs.Count(); i++) {
        //numTriangles += countPerID[i];
        megamol::core::utility::log::Log::DefaultLog.WriteInfo(
            "surface %u: %u triangles, surface %f, volume %f, voidVol %f, entire volume %f", uniqueIDs[i],
            countPerID[i], surfPerID[i], volPerID[i], voidVolPerID[i], volPerID[i] + voidVolPerID[i]);
        if (!metricsFilenameSlot.Param<core::param::FilePathParam>()->Value().empty()) {
            vislib::sys::WriteFormattedLineToFile(this->statisticsFile, "%u\t%u\t%u\t%f\t%f\n", frameNumber,
                uniqueIDs[i], countPerID[i], surfPerID[i], volPerID[i]);
        }
    }
}

void VoluMetricJob::copyVolumesToBackBuffer() {
    SIZE_T prevCount = this->debugVolumes.Count();

    if (prevCount < SubJobDataList.Count()) {
        this->debugVolumes.SetCount(SubJobDataList.Count());
        memset(&this->debugVolumes[prevCount], 0,
            sizeof(this->debugVolumes[0]) * (this->debugVolumes.Count() - prevCount));
    }

    for (SIZE_T i = 0; i < SubJobDataList.Count(); i++) {
        if (SubJobDataList[i]->storeVolume && SubJobDataList[i]->Result.done && !this->debugVolumes[i].volumeData) {
            this->debugVolumes[i] = SubJobDataList[i]->Result.debugVolume;
            SubJobDataList[i]->Result.debugVolume.volumeData = 0;
        }
    }
    this->hash++;
}

void VoluMetricJob::copyMeshesToBackbuffer(vislib::Array<unsigned int>& uniqueIDs) {
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

    trisoup::volumetrics::VoxelizerFloat *vert, *norm;
    unsigned char* col;

    vert = new trisoup::volumetrics::VoxelizerFloat[numTriangles * 9];
    norm = new trisoup::volumetrics::VoxelizerFloat[numTriangles * 9];
    col = new unsigned char[numTriangles * 9];
    //tri = new unsigned int[numTriangles * 3];
    SIZE_T vertOffset = 0;
    SIZE_T triOffset = 0;
    SIZE_T idxOffset = 0;

    if (this->showBorderGeometrySlot.Param<megamol::core::param::BoolParam>()->Value()) {

        for (unsigned int i = 0; i < uniqueIDs.Count(); i++) {
            vislib::graphics::ColourRGBAu8 c(rand() * 255, rand() * 255, rand() * 255, 255);
            //vislib::math::ShallowShallowTriangle<double, 3> sst(vert);

            for (unsigned int j = 0; j < todos.Count(); j++) {
                for (unsigned int k = 0; k < SubJobDataList[todos[j]]->Result.surfaces.Count(); k++) {
                    if (SubJobDataList[todos[j]]->Result.surfaces[k].globalID == uniqueIDs[i]) {
                        for (SIZE_T l = 0; l < SubJobDataList[todos[j]]->Result.surfaces[k].border->Count(); l++) {
                            SIZE_T vertCount =
                                (*SubJobDataList[todos[j]]->Result.surfaces[k].border)[l]->triangles.Count() / 3;
                            memcpy(&(vert[vertOffset]),
                                (*SubJobDataList[todos[j]]->Result.surfaces[k].border)[l]->triangles.PeekElements(),
                                vertCount * 3 * sizeof(trisoup::volumetrics::VoxelizerFloat));
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
        for (unsigned int i = 0; i < uniqueIDs.Count(); i++) {
            vislib::graphics::ColourRGBAu8 c(rand() * 255, rand() * 255, rand() * 255, 255);
            //vislib::math::ShallowShallowTriangle<double, 3> sst(vert);

            for (unsigned int j = 0; j < todos.Count(); j++) {
                for (unsigned int k = 0; k < SubJobDataList[todos[j]]->Result.surfaces.Count(); k++) {
                    if (SubJobDataList[todos[j]]->Result.surfaces[k].globalID == uniqueIDs[i]) {
                        SIZE_T vertCount = SubJobDataList[todos[j]]->Result.surfaces[k].mesh.Count() / 3;
                        memcpy(&(vert[vertOffset]), SubJobDataList[todos[j]]->Result.surfaces[k].mesh.PeekElements(),
                            vertCount * 3 * sizeof(trisoup::volumetrics::VoxelizerFloat));
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
        vislib::math::ShallowShallowTriangle<trisoup::volumetrics::VoxelizerFloat, 3> sst(&(vert[i * 9]));
        vislib::math::Vector<trisoup::volumetrics::VoxelizerFloat, 3> n;
        sst.Normal(n);
        memcpy(&(norm[i * 9]), n.PeekComponents(), sizeof(trisoup::volumetrics::VoxelizerFloat) * 3);
        memcpy(&(norm[i * 9 + 3]), n.PeekComponents(), sizeof(trisoup::volumetrics::VoxelizerFloat) * 3);
        memcpy(&(norm[i * 9 + 6]), n.PeekComponents(), sizeof(trisoup::volumetrics::VoxelizerFloat) * 3);
    }

    debugMeshes[meshBackBufferIndex].SetVertexData(
        static_cast<unsigned int>(vertOffset / 3), vert, norm, col, NULL, true);
    //debugMeshes[meshBackBufferIndex].SetTriangleData(vertOffset / 3, tri, true);
    debugMeshes[meshBackBufferIndex].SetTriangleData(0, NULL, false);

    meshBackBufferIndex = 1 - meshBackBufferIndex;
    this->hash++;
}
