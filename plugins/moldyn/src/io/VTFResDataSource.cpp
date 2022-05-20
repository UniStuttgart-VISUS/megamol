/*
 * VTFResDataSource.cpp
 *
 * Copyright (C) 2014 by Universitaet Stuttgart (VISUS).
 * Alle Rechte vorbehalten.
 */

#include "io/VTFResDataSource.h"
#include "geometry_calls/MultiParticleDataCall.h"
#include "mmcore/CoreInstance.h"
#include "mmcore/param/BoolParam.h"
#include "mmcore/param/FilePathParam.h"
#include "mmcore/param/StringParam.h"
#include "mmcore/utility/log/Log.h"
#include "mmcore/utility/sys/ConsoleProgressBar.h"
#include "stdafx.h"
#include "vislib/PtrArray.h"
#include "vislib/RawStorageWriter.h"
#include "vislib/String.h"
#include "vislib/StringTokeniser.h"
#include "vislib/Trace.h"
#include "vislib/math/ShallowPoint.h"
#include "vislib/math/ShallowQuaternion.h"
#include "vislib/math/ShallowVector.h"
#include "vislib/sys/DirectoryIterator.h"
#include "vislib/sys/Path.h"
#include "vislib/sys/SystemInformation.h"
#include "vislib/sys/error.h"
#include "vislib/sys/sysfunctions.h"
#include <cstdint>

using namespace megamol::core;
using namespace megamol::moldyn;


/* defines for the frame cache size */
// minimum number of frames in the cache (2 for interpolation; 1 for loading)
#define CACHE_SIZE_MIN 3
// maximum number of frames in the cache (just a nice number)
#define CACHE_SIZE_MAX 100000
// factor multiplied to the frame size for estimating the overhead to the pure data.
#define CACHE_FRAME_FACTOR 1.15f

/*****************************************************************************/

/*
 * io::VTFResDataSource::Frame::Frame
 */

io::VTFResDataSource::Frame::Frame(view::AnimDataModule& owner)
        : view::AnimDataModule::Frame(owner)
        , typeCnt(0)
        , partCnt()
        , pos()
        , col()
        , particleGrid() {}


/*
 * io::VTFResDataSource::Frame::~Frame
 */
io::VTFResDataSource::Frame::~Frame() {
    this->typeCnt = 0;
    for (unsigned int t = 0; t < this->partCnt.Count(); ++t) {
        this->pos[t].EnforceSize(0);
        this->col[t].EnforceSize(0);
    }

    for (size_t i = 0; i < this->particleGrid.Count(); ++i)
        this->particleGrid[i].Clear();
    particleGrid.Clear();

    this->partCnt.Clear();
    this->pos.Clear();
    this->col.Clear();
}


/*
 * io::VTFResDataSource::Frame::Clear
 */
void io::VTFResDataSource::Frame::Clear(void) {
    for (unsigned int i = 0; i < this->typeCnt; i++) {
        this->pos[i].EnforceSize(0);
        this->col[i].EnforceSize(0);
    }
    this->partCnt.Clear();
    this->pos.Clear();
    this->col.Clear();
    for (size_t i = 0; i < this->particleGrid.Count(); ++i)
        this->particleGrid[i].Clear();
    particleGrid.Clear();
}


/*
 * io::VTFResDataSource::Frame::LoadFrame
 */
bool io::VTFResDataSource::Frame::LoadFrame(
    vislib::sys::File* file, unsigned int idx, vislib::Array<SimpleType>& types) {

    //- = garbage....
    /*
    Number of particles: 16000
    0 355.626914534 373.332758016 164.395513544
    1 197.734277202 282.285642972 248.881401574
    ...
    15999 178.196066635 58.20785402 218.434803974
-    analyzing
-    analyzing
-
    Number of agglomerates: 2
-    Format: [particle ids] ; size_of_agglomerate longest_distance Df(diameter) Df(radius_of_gyration)
    [4020, 4464] ; 2 1.06054 3 1
    [4105, 11283] ; 2 0.979599 3 1

-    histogram:
-    2 2

    */

    this->frame = idx;
    unsigned int numParticles = 0;
    unsigned int numClusters = 0;
    unsigned int parsedClusters = 0;

    enum ParseMode { NUM_PARTICES, PARTICLES, NUM_CLUSTERS, CLUSTERS, DONE };

    ParseMode pm = NUM_PARTICES;
    unsigned int lineNumber = 0;
    while (!file->IsEOF() && pm != DONE) {
        lineNumber++;
        unsigned int readSize = 2048;
        vislib::sys::File::FileOffset cursor = file->Tell();
        vislib::StringA line = vislib::sys::ReadLineFromFileA(*file, readSize);
        while (line.Length() == readSize) {
            readSize <<= 2;
            file->Seek(cursor);
            line = vislib::sys::ReadLineFromFileA(*file, readSize);
        }

        line.TrimSpaces();

        vislib::Array<vislib::StringA> shreds = vislib::StringTokeniserA::Split(line, ' ', true);

        if (line.IsEmpty())
            continue;

        if (line.StartsWith("analyzing") || line.StartsWith("Format:"))
            continue;

        switch (pm) {
        case NUM_PARTICES: {
            if (line.StartsWith("Number of particles:")) {
                numParticles = vislib::CharTraitsA::ParseInt(shreds[3]);

                this->SetTypeCount(1);
                types[0].SetCount(numParticles);
                this->partCnt.Resize(types.Count());
                this->partCnt[0] = 0;

                this->pos[0].EnforceSize(sizeof(float) * 3 * types[0].GetCount());
                this->col[0].EnforceSize(sizeof(float) * 4 * types[0].GetCount());
                pm = PARTICLES;
                continue;
            }
            break;
        }
        case PARTICLES: {
            vislib::math::Vector<float, 3> pos;
            pos.Set((float)vislib::CharTraitsA::ParseDouble(shreds[1]),
                (float)vislib::CharTraitsA::ParseDouble(shreds[2]), (float)vislib::CharTraitsA::ParseDouble(shreds[3]));

            vislib::math::Vector<float, 4> col;
            col.Set(0, // type
                0,     // cluster id
                0, 0);
            memcpy(this->pos[0].At(sizeof(float) * 3 * this->partCnt[0]), pos.PeekComponents(), 3 * sizeof(float));
            memcpy(this->col[0].At(sizeof(float) * 4 * this->partCnt[0]), col.PeekComponents(), 4 * sizeof(float));
            this->partCnt[0]++;

            if (this->partCnt[0] == numParticles)
                pm = NUM_CLUSTERS;
            break;
        }
        case NUM_CLUSTERS: {
            if (line.StartsWith("Number of agglomerates:")) {
                numClusters = vislib::CharTraitsA::ParseInt(shreds[3]);
                if (numClusters > 0) // TODO scharnkn
                    pm = CLUSTERS;
                continue;
            }
            break;
        }
        case CLUSTERS: {
            /*
            Format: [particle ids] ; size_of_agglomerate longest_distance Df(diameter) Df(radius_of_gyration)
            [4020, 4464] ; 2 1.06054 3 1
            */

            unsigned int i = 0;
            unsigned int particleId = 0;
            while (i < shreds.Count() && !shreds[i].Contains(";", 0)) {
                vislib::StringA s = shreds[i];
                s.Replace("[", "", vislib::StringA::NO_LIMIT);
                s.Replace("]", "", vislib::StringA::NO_LIMIT);
                s.Replace(",", "", vislib::StringA::NO_LIMIT);

                particleId = vislib::CharTraitsA::ParseInt(s);

                // rgba = type=0, cluster, DF
                // NOTE TODO Changed for SFB_DEMO
                float* particleData = col[0].AsAt<float>(4 * sizeof(float) * particleId);
                particleData[1] = static_cast<float>(parsedClusters);
                float soa = (float)vislib::CharTraitsA::ParseDouble(shreds[shreds.Count() - 4]);
                float ld = (float)vislib::CharTraitsA::ParseDouble(shreds[shreds.Count() - 3]);
                //particleData[0] = (float)vislib::CharTraitsA::ParseDouble(shreds[shreds.Count() - 1]);
                particleData[0] = ld / std::pow(soa * 0.25f, 1.0f / 3.0f);

                this->clusterInfos.data[parsedClusters].Append(particleId);

                ++i;
            }

            ++parsedClusters;
            if (parsedClusters == numClusters)
                pm = DONE;
            break;
        }
        default:
            ASSERT(0 && "i dont't know what to do now :(");
            break;
        }
    }
    // count + start + data
    this->clusterInfos.sizeofPlainData =
        2 * this->clusterInfos.data.Count() * sizeof(int) + this->partCnt[0] * sizeof(int);
    this->clusterInfos.plainData = (unsigned int*)malloc(this->clusterInfos.sizeofPlainData);
    this->clusterInfos.numClusters = static_cast<unsigned int>(this->clusterInfos.data.Count());
    unsigned int ptr = 0;
    unsigned int summedSizesSoFar = static_cast<unsigned int>(2 * this->clusterInfos.data.Count());
    auto it = this->clusterInfos.data.GetConstIterator();
    while (it.HasNext()) {
        const auto current = it.Next();
        const auto arr = current.Value();
        this->clusterInfos.plainData[ptr++] = static_cast<unsigned int>(arr.Count());
        this->clusterInfos.plainData[ptr++] = summedSizesSoFar;

        memcpy(&this->clusterInfos.plainData[summedSizesSoFar], arr.PeekElements(), sizeof(unsigned int) * arr.Count());
        summedSizesSoFar += static_cast<unsigned int>(arr.Count());
    }
    /*
            timestep indexed
            0 -1 88.08974923911063 93.53975290469917 41.0842180843088940
            1 -1 50.542528784672555 69.2565090323587 62.71274455546361
            2 -1 62.747125087753524 49.00074973246766 75.57611795542917
            3 -1 9.46248516175452 43.90389079646931 43.07396560057581
            4 -1 0.32858672109087456 58.02125782527474 64.42774367401746
    *

        this->frame = idx;
            this->partCnt.Resize(types.Count());
            this->pos[0].EnforceSize(sizeof(float)* 3 * types[0].GetCount());
            this->col[0].EnforceSize(sizeof(float)* 4 * types[0].GetCount());

            unsigned int id = 0;

            while(!file->IsEOF()) {
            vislib::StringA line = vislib::sys::ReadLineFromFileA(*file);
            line.TrimSpaces();

                    vislib::Array<vislib::StringA> shreds = vislib::StringTokeniserA::Split(line, ' ', true);

                    if (line.IsEmpty() || line.StartsWithInsensitive("time")) {
                break;
                    }

                    int clusterId = (int)vislib::CharTraitsA::ParseInt(shreds[1]);
                    this->clusterInfos.data[clusterId].Append(id);

                    vislib::math::Vector<float, 3> pos;
                    pos.Set((float)vislib::CharTraitsA::ParseDouble(shreds[2]),
                                (float)vislib::CharTraitsA::ParseDouble(shreds[3]),
                                    (float)vislib::CharTraitsA::ParseDouble(shreds[4]));

                    vislib::math::Vector<float, 4> col;
                    col.Set(0, // type
                                    clusterId,
                                0,
                                    0);



                    //this->pos[0].Append(pos.PeekComponents(), 3 * sizeof(float));

                    memcpy(this->pos[0].At(4 * 3 * this->partCnt[0]), pos.PeekComponents(), 3 * sizeof(float));
                    memcpy(this->col[0].At(4 * 4 * this->partCnt[0]), col.PeekComponents(), 4 * sizeof(float));

                    ++this->partCnt[0];
                    ++id;
            }
            // count + start + data
            this->clusterInfos.sizeofPlainData = 2 * this->clusterInfos.data.Count() * sizeof(int)+this->partCnt[0] * sizeof(int);
            this->clusterInfos.plainData = (unsigned int*)malloc(this->clusterInfos.sizeofPlainData);
            this->clusterInfos.numClusters = this->clusterInfos.data.Count();
            unsigned int ptr = 0;
            unsigned int summedSizesSoFar = 2 * this->clusterInfos.data.Count();
            auto it = this->clusterInfos.data.GetConstIterator();
            while (it.HasNext())
            {
                    const auto current = it.Next();
                    const auto arr = current.Value();
                    this->clusterInfos.plainData[ptr++] = arr.Count();
                    this->clusterInfos.plainData[ptr++] = summedSizesSoFar;

                    memcpy(&this->clusterInfos.plainData[summedSizesSoFar], arr.PeekElements(), sizeof(unsigned int)* arr.Count());
                    summedSizesSoFar += arr.Count();
            }

            /*
            this->col[0].AssertSize(this->partCnt[0] * sizeof(float) * 4);
            for(unsigned int i = 0; i < this->partCnt[0]; ++i)
            {
                    *this->col[0].AsAt<float>(4*(4*i + 0)) = 0.5f;
                    *this->col[0].AsAt<float>(4*(4*i + 1)) = 0.5f;
                    *this->col[0].AsAt<float>(4*(4*i + 2)) = 0.5f;
                    *this->col[0].AsAt<float>(4*(4*i + 3)) = 1.0f;
            }
            */
    VLTRACE(VISLIB_TRCELVL_INFO, "Frame %u loaded\n", this->frame);

    return true;
}

/*
 * io::VTFResDataSource::Frame::PartPoss
 */
const float* io::VTFResDataSource::Frame::PartPoss(unsigned int type) const {
    ASSERT(type < this->typeCnt);
    return this->pos[type].As<float>();
}

/*
 * io::VTFResDataSource::Frame::PartCols
 */
const float* io::VTFResDataSource::Frame::PartCols(unsigned int type) const {
    ASSERT(type < this->typeCnt);
    return this->col[type].As<float>();
}

/*
 * io::VTFResDataSource::Frame::UpdatePartColor
 */
void io::VTFResDataSource::Frame::UpdatePartColor(
    unsigned int type, unsigned int idx, vislib::math::Vector<float, 4> color) {
    *this->col[type].AsAt<float>(4 * (4 * idx + 0)) = color.GetX();
    *this->col[type].AsAt<float>(4 * (4 * idx + 1)) = color.GetY();
    *this->col[type].AsAt<float>(4 * (4 * idx + 2)) = color.GetZ();
    *this->col[type].AsAt<float>(4 * (4 * idx + 3)) = color.GetW();
}


/*
 * io::VTFResDataSource::Frame::SizeOf
 */
SIZE_T io::VTFResDataSource::Frame::SizeOf(void) const {
    SIZE_T size = 0;
    for (unsigned int i = 0; i < this->typeCnt; i++) {
        size += this->pos[i].GetSize();
        size += this->col[i].GetSize();
    }
    return size;
}


/*
 * io::VTFResDataSource::Frame::MakeInterpolationFrame
 */
const io::VTFResDataSource::Frame* io::VTFResDataSource::Frame::MakeInterpolationFrame(
    float alpha, const io::VTFResDataSource::Frame& a, const io::VTFResDataSource::Frame& b) {
    ASSERT(a.typeCnt == b.typeCnt);

    if (alpha < 0.0000001f)
        return &a;
    if (alpha > 0.9999999f)
        return &b;
    float beta = 1.0f - alpha;

    for (unsigned int t = 0; t < a.typeCnt; t++) {
        if (a.partCnt[t] != b.partCnt[t])
            return &a;
        this->partCnt[t] = a.partCnt[t];
        this->pos[t].AssertSize(this->partCnt[t] * 3 * sizeof(float));
    }

    for (unsigned int t = 0; t < a.typeCnt; t++) {
        for (unsigned int i = 0; i < this->partCnt[t]; i++) {
            vislib::math::ShallowPoint<float, 3> av((float*)a.pos[t].As<float>() + i * 3);
            vislib::math::ShallowPoint<float, 3> bv((float*)b.pos[t].As<float>() + i * 3);
            vislib::math::ShallowPoint<float, 3> tv(this->pos[t].As<float>() + i * 3);

            if (av.SquareDistance(bv) > 0.01) {
                tv = (alpha < 0.5f) ? av : bv;
            } else {
                tv.Set(av.X() * beta + bv.X() * alpha, av.Y() * beta + bv.Y() * alpha, av.Z() * beta + bv.Z() * alpha);
            }
        }
    }

    return this;
}


/*
 * io::VTFResDataSource::Frame::parseParticleLine
 */
void io::VTFResDataSource::Frame::parseParticleLine(
    vislib::StringA& line, int& outType, float& outX, float& outY, float& outZ) {

    vislib::Array<vislib::StringA> shreds = vislib::StringTokeniserA::Split(line, ' ', true);
    if ((shreds.Count() != 9) && (shreds.Count() != 5)) {
        throw 0; // invalid line separations
    }
    if (!shreds[0].Equals("!")) {
        throw 0; // invalid line marker
    }

    outType = vislib::CharTraitsA::ParseInt(shreds[1]);
    outX = float(vislib::CharTraitsA::ParseInt(shreds[2])); // de-quantization of positions is done later
    outY = float(vislib::CharTraitsA::ParseInt(shreds[3]));
    outZ = float(vislib::CharTraitsA::ParseInt(shreds[4]));
}


/*
 * io::VTFResDataSource::Frame::SetTypeCount
 */
void io::VTFResDataSource::Frame::SetTypeCount(unsigned int cnt) {
    this->Clear();
    this->typeCnt = cnt;
    this->partCnt.Resize(cnt);
    this->pos.Resize(cnt);
    this->col.Resize(cnt);
    for (unsigned int i = 0; i < cnt; i++) {
        this->pos.Append(*new vislib::RawStorage());
        this->col.Append(*new vislib::RawStorage());
        this->partCnt.Append(0);
    }
}

void io::VTFResDataSource::Frame::initParticleGrid(unsigned int N1, unsigned int N2, unsigned int N3) {
    this->particleGridDim1 = N1;
    this->particleGridDim2 = N2;
    this->particleGridDim3 = N3;
    this->particleGrid.SetCount(particleGridDim1 * particleGridDim2 * particleGridDim3);
}

vislib::Array<int>& io::VTFResDataSource::Frame::particleGridCell(unsigned int N1, unsigned int N2, unsigned int N3) {
    N1 = vislib::math::Min<unsigned int>(0, vislib::math::Max<unsigned int>(N1, particleGridDim1 - 1));
    N2 = vislib::math::Min<unsigned int>(0, vislib::math::Max<unsigned int>(N2, particleGridDim2 - 1));
    N3 = vislib::math::Min<unsigned int>(0, vislib::math::Max<unsigned int>(N3, particleGridDim3 - 1));
    return this->particleGrid[N3 * particleGridDim1 * particleGridDim2 + N2 * particleGridDim1 + N3];
}


/*****************************************************************************/


/*
 * io::VTFResDataSource::VTFResDataSource
 */
io::VTFResDataSource::VTFResDataSource(void)
        : view::AnimDataModule()
        ,
        // TODO scharnkn: this is actually supposed to be the directory
        filename("filename", "The path to the trisoup file to load.")
        , getData("getdata", "Slot to request data from this data source.")
        , preprocessSlot("preprocess", "aggregation preprocessing")
        , types()
        , frameIdx()
        , file(NULL)
        , datahash(0) {

    this->filename.SetParameter(new param::FilePathParam(""));
    this->filename.SetUpdateCallback(&VTFResDataSource::filenameChanged);
    this->MakeSlotAvailable(&this->filename);

    preprocessSlot << new param::BoolParam(false);
    this->MakeSlotAvailable(&this->preprocessSlot);

    this->getData.SetCallback("MultiParticleDataCall", "GetData", &VTFResDataSource::getDataCallback);
    this->getData.SetCallback("MultiParticleDataCall", "GetExtent", &VTFResDataSource::getExtentCallback);
    this->MakeSlotAvailable(&this->getData);

    this->setFrameCount(1);
    this->initFrameCache(1);
}


/*
 * io::VTFResDataSource::~VTFResDataSource
 */
io::VTFResDataSource::~VTFResDataSource(void) {
    this->Release(); // implicitly calls 'release'
}


/*
 * io::VTFResDataSource::constructFrame
 */
view::AnimDataModule::Frame* io::VTFResDataSource::constructFrame(void) const {
    Frame* f = new Frame(*const_cast<io::VTFResDataSource*>(this));
    f->SetTypeCount((unsigned int)this->types.Count());
    return f;
}

/*
 * io::VTFResDataSource::create
 */
bool io::VTFResDataSource::create(void) {
    return true;
}


/*
 * io::VTFResDataSource::loadFrame
 */
void io::VTFResDataSource::loadFrame(view::AnimDataModule::Frame* frame, unsigned int idx) {

    Frame* f = dynamic_cast<Frame*>(frame);
    if (f == NULL)
        return;

    if (this->frameFiles.Count() <= idx)
        return;

    vislib::sys::File file;
    file.Open(this->frameFiles[idx], vislib::sys::File::READ_ONLY, vislib::sys::File::SHARE_READ,
        vislib::sys::File::OPEN_ONLY);
    if (file.IsOpen() == false) {
        f->Clear();
        return;
    }
    ASSERT(idx < this->FrameCount());

    f->LoadFrame(&file, idx, this->types);

    if (this->preprocessSlot.Param<param::BoolParam>()->Value())
        preprocessFrame(*f);

    /*
Frame *f = dynamic_cast<Frame*>(frame);
if (f == NULL) return;
if (this->file == NULL) {
    f->Clear();
    return;
}
ASSERT(idx < this->FrameCount());

this->file->Seek(this->frameIdx[idx]);
f->LoadFrame(this->file, idx, this->types);

    if(this->preprocessSlot.Param<param::BoolParam>()->Value())
            preprocessFrame(*f);
    */
}

void io::VTFResDataSource::preprocessFrame(Frame& frame) {
    unsigned int N = 20;

    frame.initParticleGrid(N, N, N);

    // sort into grid
    vislib::Array<int>* cell = NULL;
    for (unsigned int t = 0; t < this->types.Count(); ++t) {
        const float* v = frame.PartPoss(t);
        for (unsigned int p = 0; p < frame.PartCnt(t); ++p) {
            // HAZARD: This cast brackets might be wrong
            unsigned int x = static_cast<unsigned int>(floorf((float)N * v[3 * p + 0])) /
                             static_cast<unsigned int>(this->extents.GetX());
            unsigned int y = static_cast<unsigned int>(floorf((float)N * v[3 * p + 1])) /
                             static_cast<unsigned int>(this->extents.GetY());
            unsigned int z = static_cast<unsigned int>(floorf((float)N * v[3 * p + 2])) /
                             static_cast<unsigned int>(this->extents.GetZ());
            vislib::Array<int>& cell = frame.particleGridCell(x, y, z);
            cell.Add(p);
        }
    }

    vislib::math::Vector<float, 4> red(1.0f, 0.0f, 0.0f, 1.0f);
    // iterate over each particle
    for (unsigned int t = 0; t < this->types.Count(); ++t) {
        const float* v = frame.PartPoss(t);
        const float* cols = frame.PartCols(t);
        for (unsigned int p = 0; p < frame.PartCnt(t); ++p) {
            vislib::math::ShallowVector<float, 3> pos(const_cast<float*>(&v[3 * p]));

            unsigned int x = static_cast<unsigned int>(floorf((float)N * pos.GetX())) /
                             static_cast<unsigned int>(this->extents.GetX());
            unsigned int y = static_cast<unsigned int>(floorf((float)N * pos.GetY())) /
                             static_cast<unsigned int>(this->extents.GetY());
            unsigned int z = static_cast<unsigned int>(floorf((float)N * pos.GetZ())) /
                             static_cast<unsigned int>(this->extents.GetZ());

            // over each neighboring cell
            for (int i = x - 1; i < static_cast<int>(x + 1); ++i)
                for (int j = y - 1; j < static_cast<int>(y + 1); ++j)
                    for (int k = z - 1; k < static_cast<int>(z + 1); ++k) {
                        vislib::Array<int>& cell = frame.particleGridCell(x, y, z);
                        for (unsigned int c = 0; c < cell.Count(); ++c) {
                            unsigned int id = cell[c];
                            if (id == p)
                                continue;

                            vislib::math::ShallowVector<float, 3> posNeighbor(const_cast<float*>(&v[3 * id]));
                            if ((posNeighbor - pos).Length() <= 2.0f * this->types[t].Radius())
                                frame.UpdatePartColor(t, id, red);
                        }
                    }
        }
    }
}

/*
 * io::VTFResDataSource::release
 */
void io::VTFResDataSource::release(void) {
    this->resetFrameCache();
    if (this->file != NULL) {
        vislib::sys::File* f = this->file;
        this->file = NULL;
        f->Close();
        delete f;
    }
    this->types.Clear();
    this->frameIdx.Clear();
}

/*
 * io::VTFResDataSource::filenameChanged
 */
bool io::VTFResDataSource::filenameChanged(param::ParamSlot& slot) {

    this->types.Clear();
    this->frameIdx.Clear();
    this->resetFrameCache();

    this->datahash++;

    ASSERT(this->filename.Param<param::FilePathParam>() != NULL);
    getFrameFiles(this->filename.Param<param::FilePathParam>()->Value().generic_u8string().c_str());

    /*
if (this->file == NULL) {
    //this->file = new vislib::sys::FastFile();
            this->file = new vislib::sys::FastFile();
            this->file->SetBufferSize(2 << 30);
} else {
    this->file->Close();
}
ASSERT(this->filename.Param<param::FilePathParam>() != NULL);

if (!this->file->Open(this->filename.Param<param::FilePathParam>()->Value(),
        vislib::sys::File::READ_ONLY, vislib::sys::File::SHARE_READ, vislib::sys::File::OPEN_ONLY)) {
    vislib::sys::SystemMessage err(::GetLastError());
    this->GetCoreInstance()->Log().WriteMsg(megamol::core::utility::log::Log::LEVEL_ERROR,
        "Unable to open VTF-File \"%s\": %s", vislib::StringA(
        this->filename.Param<param::FilePathParam>()->Value()).PeekBuffer(),
        static_cast<const char*>(err));

    SAFE_DELETE(this->file);
    this->setFrameCount(1);
    this->initFrameCache(1);

    return true;
}

if (!this->parseHeaderAndFrameIndices(this->filename.Param<param::FilePathParam>()->Value())) {
    this->GetCoreInstance()->Log().WriteMsg(megamol::core::utility::log::Log::LEVEL_ERROR,
        "Unable to read VTF-Header from file \"%s\". Wrong format?", vislib::StringA(
        this->filename.Param<param::FilePathParam>()->Value()).PeekBuffer());

    this->file->Close();
    SAFE_DELETE(this->file);
    this->setFrameCount(1);
    this->initFrameCache(1);

    return true;
}
*/
    Frame tmpFrame(*this);
    tmpFrame.SetTypeCount((unsigned int)this->types.Count());
    // use frame zero to estimate the frame size in memory to calculate the
    // frame cache size
    this->loadFrame(&tmpFrame, 0);
    SIZE_T frameSize = tmpFrame.SizeOf();
    tmpFrame.Clear();
    frameSize = static_cast<SIZE_T>(float(frameSize) * CACHE_FRAME_FACTOR);
    UINT64 mem = vislib::sys::SystemInformation::AvailableMemorySize();
    unsigned int cacheSize = static_cast<unsigned int>(mem / frameSize);

    if (cacheSize > CACHE_SIZE_MAX) {
        cacheSize = CACHE_SIZE_MAX;
    }
    if (cacheSize < CACHE_SIZE_MIN) {
        vislib::StringA msg;
        msg.Format("Frame cache size forced to %i. Calculated size was %u.\n", CACHE_SIZE_MIN, cacheSize);
        megamol::core::utility::log::Log::DefaultLog.WriteMsg(megamol::core::utility::log::Log::LEVEL_WARN, msg);
        cacheSize = CACHE_SIZE_MIN;
    } else {
        vislib::StringA msg;
        msg.Format("Frame cache size set to %i.\n", cacheSize);
        megamol::core::utility::log::Log::DefaultLog.WriteMsg(megamol::core::utility::log::Log::LEVEL_INFO, msg);
    }

    this->initFrameCache(cacheSize);

    return true; // to reset the dirty flag of the param slot
}

int numberInString(const vislib::StringA& s) {
    int retVal = 0;

    for (int i = 0; i < s.Length(); ++i) {
        if (s[i] >= '0' && s[i] <= '9')
            retVal = retVal * 10 + (s[i] - '0');
    }

    return retVal;
}

int sortFileArray(const vislib::StringA& lhs, const vislib::StringA& rhs) {
    vislib::StringA l = lhs.Substring(lhs.FindLast("/", vislib::StringA::INVALID_POS) + 1);
    vislib::StringA r = rhs.Substring(rhs.FindLast("/", vislib::StringA::INVALID_POS) + 1);

    return numberInString(l) - numberInString(r);
}

void io::VTFResDataSource::getFrameFiles(vislib::StringA directory) {
    vislib::sys::DirectoryIterator<vislib::CharTraits<char>> di(directory + "/*.res", true, false);
    while (di.HasNext()) {
        this->frameFiles.Append(directory + "/" + di.Next().Path);
    }

    this->frameFiles.Sort(&sortFileArray);

    this->setFrameCount((unsigned int)this->frameFiles.Count());
    printf("Frame Count %u\n", (unsigned int)this->frameFiles.Count());

    // only one type here
    SimpleType type;
    type.SetID(0);
    type.SetRadius(0.5f);
    type.SetCount(0);
    types.Append(type);

    extents.Set(439, 439, 439);
}

/*
 * io::VTFResDataSource::readHeader
 *
bool io::VTFResDataSource::parseHeaderAndFrameIndices(const vislib::TString& filename) {

    ASSERT(this->file->Tell() == 0);

        /*
        pbc 100.0 100.0 100.0
        atom 0:999 radius 0.5 name O type 0

        time index
        0 -1 88.08974923911063 93.53975290469917 41.0842180843088940
        1 -1 50.542528784672555 69.2565090323587 62.71274455546361
        2 -1 62.747125087753524 49.00074973246766 75.57611795542917
        3 -1 9.46248516175452 43.90389079646931 43.07396560057581
        4 -1 0.32858672109087456 58.02125782527474 64.42774367401746

        pbc X Y Z
        atom from:to radius R name O type 0
        *

        bool haveBoundingBox = false;
        bool haveAtomType = false;

        this->types.Clear();

    vislib::sys::ConsoleProgressBar cpb;
    cpb.Start("Progress Loading VTF File", static_cast<vislib::sys::ConsoleProgressBar::Size>(this->file->GetSize()));

    // read the header
    while (!this->file->IsEOF()) {
                vislib::sys::File::FileSize currentFileCursor = this->file->Tell();
        vislib::StringA line = vislib::sys::ReadLineFromFileA(*this->file);
        line.TrimSpaces();

        if (line.IsEmpty())
            continue;

                vislib::Array<vislib::StringA> shreds = vislib::StringTokeniserA::Split(line, ' ', true);

                if(!haveBoundingBox) {
                        if(shreds[0].CompareInsensitive("pbc")){
                                extents.Set((float)vislib::CharTraitsA::ParseDouble(shreds[1]),
                                                (float)vislib::CharTraitsA::ParseDouble(shreds[2]),
                                                        (float)vislib::CharTraitsA::ParseDouble(shreds[3]));

                                haveBoundingBox = true;
                                continue;
                        }
                }
                if(!haveAtomType) {
                        if(shreds[0].CompareInsensitive("atom")){
                                vislib::Array<vislib::StringA> counts = vislib::StringTokeniserA::Split(shreds[1], ':', true);

                                SimpleType type;
                                type.SetID(vislib::CharTraitsA::ParseInt(shreds[7]));
                                type.SetRadius((float)vislib::CharTraitsA::ParseDouble(shreds[3]));
                                type.SetCount(vislib::CharTraitsA::ParseInt(counts[1]) - vislib::CharTraitsA::ParseInt(counts[0]) + 1);
                                this->types.Append(type);
                                haveAtomType = true;
                                continue;
                        }
                }

                if(haveBoundingBox && haveAtomType) {
                        if (shreds[0].CompareInsensitive("time") && shreds[1].CompareInsensitive("index")) {
                                this->frameIdx.Append(this->file->Tell());
                                cpb.Set(static_cast<vislib::sys::ConsoleProgressBar::Size>(this->file->Tell()));
                        }
                }
    }
        this->setFrameCount((unsigned int)this->frameIdx.Count());
        //this->initFrameCache(1);

    return true;
}
*/

/*
 * io::VTFResDataSource::getDataCallback
 */
bool io::VTFResDataSource::getDataCallback(Call& caller) {
    using geocalls::MultiParticleDataCall;
    MultiParticleDataCall* c2 = dynamic_cast<MultiParticleDataCall*>(&caller);

    Frame* f = NULL;
    if (c2 != NULL) {
        f = dynamic_cast<Frame*>(this->requestLockedFrame(c2->FrameID()));
        if (f == NULL)
            return false;
        c2->SetUnlocker(new Unlocker(*f));

        // If the 'force' flag is set, check whether the frame number is correct,
        // if not re-request the frame
        if (c2->IsFrameForced()) {
            while (c2->FrameID() != f->FrameNumber()) {
                c2->Unlock();
                int frameBefore = ((static_cast<int>(c2->FrameID() - 1) + static_cast<int>(this->FrameCount()))) %
                                  static_cast<int>(this->FrameCount());
                //                printf("PDBLoader (while frame %i) %u requested, frame count %u\n",
                //                        frameBefore, dc->FrameID(), this->FrameCount()); // DEBUG

                // scharnkn:
                // Request the frame before the actual requested frame (modulo
                // framenumber) to trigger loading of the actually requested frame
                f = dynamic_cast<io::VTFResDataSource::Frame*>(this->requestLockedFrame(frameBefore));
                c2->SetUnlocker(new Unlocker(*f));
                c2->Unlock();
                //printf("PDBLoader: (while frame loaded): %u (%u requested)\n",
                //        fr->FrameNumber(), frameBefore); // DEBUG
                f = dynamic_cast<io::VTFResDataSource::Frame*>(this->requestLockedFrame(c2->FrameID()));
                c2->SetUnlocker(new Unlocker(*f));
                //printf("PDBLoader: (while frame loaded): %u (%u requested)\n",
                //        fr->FrameNumber(), dc->FrameID()); // DEBUG
                if (f == NULL) {
                    return false;
                }
            }
        }


        //        if (f->FrameNumber() != c2->FrameID())
        //            c2->SetFrameID(f->FrameNumber());


        c2->SetDataHash((this->file == NULL) ? 0 : this->datahash);
        c2->SetUnlocker(new Unlocker(*f));
        c2->SetParticleListCount((unsigned int)this->types.Count());
        for (unsigned int i = 0; i < this->types.Count(); i++) {
            c2->AccessParticles(i).SetGlobalRadius(this->types[i].Radius() /* / this->boxScaling*/);
            c2->AccessParticles(i).SetGlobalColour(this->types[i].Red(), this->types[i].Green(), this->types[i].Blue());
            c2->AccessParticles(i).SetCount(f->PartCnt(i));
            const float* vd = f->PartPoss(i);
            c2->AccessParticles(i).SetVertexData((vd == NULL) ? MultiParticleDataCall::Particles::VERTDATA_NONE
                                                              : MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZ,
                vd, sizeof(float) * 3);
            const float* cd = f->PartCols(i);
            c2->AccessParticles(i).SetColourData((cd == NULL) ? MultiParticleDataCall::Particles::COLDATA_NONE
                                                              : MultiParticleDataCall::Particles::COLDATA_FLOAT_I,
                cd, sizeof(float) * 4);

            c2->AccessParticles(i).SetColourMapIndexValues(1.0f, 10.0f);

            c2->AccessParticles(i).SetClusterInfos(f->GetClusterInfos());
        }
        return true;
    }
    return false;
}


/*
 * io::VTFResDataSource::getExtentCallback
 */
bool io::VTFResDataSource::getExtentCallback(Call& caller) {
    using geocalls::MultiParticleDataCall;
    MultiParticleDataCall* c2 = dynamic_cast<MultiParticleDataCall*>(&caller);
    float border = 0.0f;

    if (c2 != NULL) {
        for (unsigned int i = 0; i < this->types.Count(); i++) {
            float r = this->types[i].Radius(); // / this->boxScaling;
            if (r > border)
                border = r;
        }

        c2->SetDataHash((this->file == NULL) ? 0 : this->datahash);
        c2->SetFrameCount(this->FrameCount());
        c2->AccessBoundingBoxes().Clear();
        c2->AccessBoundingBoxes().SetObjectSpaceBBox(
            0, 0, 0, this->extents.GetX(), this->extents.GetY(), this->extents.GetZ());
        c2->AccessBoundingBoxes().SetObjectSpaceClipBox(-border, -border, -border, this->extents.GetX() + border,
            this->extents.GetY() + border, this->extents.GetZ() + border);
        return true;
    }

    return false;
}
