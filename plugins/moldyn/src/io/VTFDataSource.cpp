/*
 * VTFDataSource.cpp
 *
 * Copyright (C) 2014 by Universitaet Stuttgart (VISUS).
 * Alle Rechte vorbehalten.
 */

#include "io/VTFDataSource.h"
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
#include "vislib/sys/BufferedFile.h"
#include "vislib/sys/FastFile.h"
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
 * io::VTFDataSource::Frame::Frame
 */

io::VTFDataSource::Frame::Frame(view::AnimDataModule& owner)
        : view::AnimDataModule::Frame(owner)
        , typeCnt(0)
        , partCnt()
        , pos()
        , col()
        , particleGrid() {}


/*
 * io::VTFDataSource::Frame::~Frame
 */
io::VTFDataSource::Frame::~Frame() {
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
 * io::VTFDataSource::Frame::Clear
 */
void io::VTFDataSource::Frame::Clear(void) {
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
 * io::VTFDataSource::Frame::LoadFrame
 */
bool io::VTFDataSource::Frame::LoadFrame(vislib::sys::File* file, unsigned int idx, vislib::Array<SimpleType>& types) {
    /*
            timestep indexed
            0 -1 88.08974923911063 93.53975290469917 41.0842180843088940
            1 -1 50.542528784672555 69.2565090323587 62.71274455546361
            2 -1 62.747125087753524 49.00074973246766 75.57611795542917
            3 -1 9.46248516175452 43.90389079646931 43.07396560057581
            4 -1 0.32858672109087456 58.02125782527474 64.42774367401746
    */

    this->frame = idx;
    this->partCnt.Resize(types.Count());
    this->pos[0].EnforceSize(sizeof(float) * 3 * types[0].GetCount());
    this->col[0].EnforceSize(sizeof(float) * 4 * types[0].GetCount());

    unsigned int id = 0;

    while (!file->IsEOF()) {
        vislib::StringA line = vislib::sys::ReadLineFromFileA(*file);
        line.TrimSpaces();

        vislib::Array<vislib::StringA> shreds = vislib::StringTokeniserA::Split(line, ' ', true);

        if (line.IsEmpty() || line.StartsWithInsensitive("time")) {
            break;
        }

        int clusterId = (int)vislib::CharTraitsA::ParseInt(shreds[1]);
        this->clusterInfos.data[clusterId].Append(id);

        vislib::math::Vector<float, 3> pos;
        pos.Set((float)vislib::CharTraitsA::ParseDouble(shreds[2]), (float)vislib::CharTraitsA::ParseDouble(shreds[3]),
            (float)vislib::CharTraitsA::ParseDouble(shreds[4]));

        vislib::math::Vector<float, 4> col;
        col.Set(0.0f, // type
            static_cast<float>(clusterId), 0.0f, 0.0f);


        //this->pos[0].Append(pos.PeekComponents(), 3 * sizeof(float));

        memcpy(this->pos[0].At(4 * 3 * this->partCnt[0]), pos.PeekComponents(), 3 * sizeof(float));
        memcpy(this->col[0].At(4 * 4 * this->partCnt[0]), col.PeekComponents(), 4 * sizeof(float));

        ++this->partCnt[0];
        ++id;
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
 * io::VTFDataSource::Frame::PartPoss
 */
const float* io::VTFDataSource::Frame::PartPoss(unsigned int type) const {
    ASSERT(type < this->typeCnt);
    return this->pos[type].As<float>();
}

/*
 * io::VTFDataSource::Frame::PartCols
 */
const float* io::VTFDataSource::Frame::PartCols(unsigned int type) const {
    ASSERT(type < this->typeCnt);
    return this->col[type].As<float>();
}

/*
 * io::VTFDataSource::Frame::UpdatePartColor
 */
void io::VTFDataSource::Frame::UpdatePartColor(
    unsigned int type, unsigned int idx, vislib::math::Vector<float, 4> color) {
    *this->col[type].AsAt<float>(4 * (4 * idx + 0)) = color.GetX();
    *this->col[type].AsAt<float>(4 * (4 * idx + 1)) = color.GetY();
    *this->col[type].AsAt<float>(4 * (4 * idx + 2)) = color.GetZ();
    *this->col[type].AsAt<float>(4 * (4 * idx + 3)) = color.GetW();
}


/*
 * io::VTFDataSource::Frame::SizeOf
 */
SIZE_T io::VTFDataSource::Frame::SizeOf(void) const {
    SIZE_T size = 0;
    for (unsigned int i = 0; i < this->typeCnt; i++) {
        size += this->pos[i].GetSize();
        size += this->col[i].GetSize();
    }
    return size;
}


/*
 * io::VTFDataSource::Frame::MakeInterpolationFrame
 */
const io::VTFDataSource::Frame* io::VTFDataSource::Frame::MakeInterpolationFrame(
    float alpha, const io::VTFDataSource::Frame& a, const io::VTFDataSource::Frame& b) {
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
 * io::VTFDataSource::Frame::parseParticleLine
 */
void io::VTFDataSource::Frame::parseParticleLine(
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
 * io::VTFDataSource::Frame::SetTypeCount
 */
void io::VTFDataSource::Frame::SetTypeCount(unsigned int cnt) {
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

void io::VTFDataSource::Frame::initParticleGrid(unsigned int N1, unsigned int N2, unsigned int N3) {
    this->particleGridDim1 = N1;
    this->particleGridDim2 = N2;
    this->particleGridDim3 = N3;
    this->particleGrid.SetCount(particleGridDim1 * particleGridDim2 * particleGridDim3);
}

vislib::Array<int>& io::VTFDataSource::Frame::particleGridCell(unsigned int N1, unsigned int N2, unsigned int N3) {
    N1 = vislib::math::Min<unsigned int>(0, vislib::math::Max<unsigned int>(N1, particleGridDim1 - 1));
    N2 = vislib::math::Min<unsigned int>(0, vislib::math::Max<unsigned int>(N2, particleGridDim2 - 1));
    N3 = vislib::math::Min<unsigned int>(0, vislib::math::Max<unsigned int>(N3, particleGridDim3 - 1));
    return this->particleGrid[N3 * particleGridDim1 * particleGridDim2 + N2 * particleGridDim1 + N3];
}


/*****************************************************************************/


/*
 * io::VTFDataSource::VTFDataSource
 */
io::VTFDataSource::VTFDataSource(void)
        : view::AnimDataModule()
        , filename("filename", "The path to the trisoup file to load.")
        , getData("getdata", "Slot to request data from this data source.")
        , preprocessSlot("preprocess", "aggregation preprocessing")
        , types()
        , frameIdx()
        , file(NULL)
        , datahash(0) {

    this->filename.SetParameter(new param::FilePathParam(""));
    this->filename.SetUpdateCallback(&VTFDataSource::filenameChanged);
    this->MakeSlotAvailable(&this->filename);

    preprocessSlot << new param::BoolParam(false);
    this->MakeSlotAvailable(&this->preprocessSlot);

    this->getData.SetCallback("MultiParticleDataCall", "GetData", &VTFDataSource::getDataCallback);
    this->getData.SetCallback("MultiParticleDataCall", "GetExtent", &VTFDataSource::getExtentCallback);
    this->MakeSlotAvailable(&this->getData);

    this->setFrameCount(1);
    this->initFrameCache(1);
}


/*
 * io::VTFDataSource::~VTFDataSource
 */
io::VTFDataSource::~VTFDataSource(void) {
    this->Release(); // implicitly calls 'release'
}


/*
 * io::VTFDataSource::constructFrame
 */
view::AnimDataModule::Frame* io::VTFDataSource::constructFrame(void) const {
    Frame* f = new Frame(*const_cast<io::VTFDataSource*>(this));
    f->SetTypeCount((unsigned int)this->types.Count());
    return f;
}

/*
 * io::VTFDataSource::create
 */
bool io::VTFDataSource::create(void) {
    return true;
}


/*
 * io::VTFDataSource::loadFrame
 */
void io::VTFDataSource::loadFrame(view::AnimDataModule::Frame* frame, unsigned int idx) {
    Frame* f = dynamic_cast<Frame*>(frame);
    if (f == NULL)
        return;
    if (this->file == NULL) {
        f->Clear();
        return;
    }
    ASSERT(idx < this->FrameCount());

    this->file->Seek(this->frameIdx[idx]);
    f->LoadFrame(this->file, idx, this->types);

    if (this->preprocessSlot.Param<param::BoolParam>()->Value())
        preprocessFrame(*f);
}

void io::VTFDataSource::preprocessFrame(Frame& frame) {
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
 * io::VTFDataSource::release
 */
void io::VTFDataSource::release(void) {
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
 * io::VTFDataSource::filenameChanged
 */
bool io::VTFDataSource::filenameChanged(param::ParamSlot& slot) {

    this->types.Clear();
    this->frameIdx.Clear();
    this->resetFrameCache();

    this->datahash++;

    if (this->file == NULL) {
        //this->file = new vislib::sys::FastFile();
        this->file = new vislib::sys::BufferedFile();
        dynamic_cast<vislib::sys::BufferedFile*>(this->file)->SetBufferSize(2 << 30);
    } else {
        this->file->Close();
    }
    ASSERT(this->filename.Param<param::FilePathParam>() != NULL);

    if (!this->file->Open(this->filename.Param<param::FilePathParam>()->Value().native().c_str(),
            vislib::sys::File::READ_ONLY, vislib::sys::File::SHARE_READ, vislib::sys::File::OPEN_ONLY)) {
        vislib::sys::SystemMessage err(::GetLastError());
        megamol::core::utility::log::Log::DefaultLog.WriteMsg(megamol::core::utility::log::Log::LEVEL_ERROR,
            "Unable to open VTF-File \"%s\": %s",
            this->filename.Param<param::FilePathParam>()->Value().generic_u8string().c_str(),
            static_cast<const char*>(err));

        SAFE_DELETE(this->file);
        this->setFrameCount(1);
        this->initFrameCache(1);

        return true;
    }

    if (!this->parseHeaderAndFrameIndices(
            this->filename.Param<param::FilePathParam>()->Value().generic_u8string().c_str())) {
        megamol::core::utility::log::Log::DefaultLog.WriteMsg(megamol::core::utility::log::Log::LEVEL_ERROR,
            "Unable to read VTF-Header from file \"%s\". Wrong format?",
            this->filename.Param<param::FilePathParam>()->Value().generic_u8string().c_str());

        this->file->Close();
        SAFE_DELETE(this->file);
        this->setFrameCount(1);
        this->initFrameCache(1);

        return true;
    }

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

/*
 * io::VTFDataSource::readHeader
 */
bool io::VTFDataSource::parseHeaderAndFrameIndices(const vislib::TString& filename) {

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
    */

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

        if (!haveBoundingBox) {
            if (shreds[0].Compare("pbc", false)) {
                extents.Set((float)vislib::CharTraitsA::ParseDouble(shreds[1]),
                    (float)vislib::CharTraitsA::ParseDouble(shreds[2]),
                    (float)vislib::CharTraitsA::ParseDouble(shreds[3]));

                haveBoundingBox = true;
                continue;
            }
        }
        if (!haveAtomType) {
            if (shreds[0].Compare("atom", false)) {
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

        if (haveBoundingBox && haveAtomType) {
            if (shreds[0].Compare("time", false) && shreds[1].Compare("index", false)) {
                this->frameIdx.Append(this->file->Tell());
                cpb.Set(static_cast<vislib::sys::ConsoleProgressBar::Size>(this->file->Tell()));
            }
        }

        /*
vislib::StringA line = vislib::sys::ReadLineFromFileA(*this->file);
line.TrimSpaces();

if (line.IsEmpty()) {
    continue;

} else if (line[0] == '~') {
    SimpleType *element = NULL;
    try {// a type line!
        int type;
        element = this->parseTypeLine(line, type);
        if (element == NULL) {
            throw 0;
        }

        for (unsigned int i = 0; i < types.Count(); i++) {
            if (types[i].ID() == static_cast<unsigned int>(type)) {
                if (types[i].Radius() < element->Radius()) {
                    types[i].SetRadius(element->Radius());
                }
                element = NULL;
                break;
            }
        }
        if (element != NULL) {
            //type = types.Count;
            types.Append(*element);
            types.Last() = *element;
        }
        element = NULL;
    } catch(...) {
        megamol::core::utility::log::Log::DefaultLog.WriteMsg(50, "Error parsing type line.");
    }
    SAFE_DELETE(element);
} else if (line[0] == '>') {
    // very extream file redirection
    vislib::StringW vnfn(vislib::StringA(line.PeekBuffer() + 1));
    vnfn.TrimSpaces();
    vislib::StringW base = vislib::sys::Path::Resolve(vislib::StringW(filename));
    base.Truncate(base.FindLast(vislib::sys::Path::SEPARATOR_W) + 1);
    vnfn = vislib::sys::Path::Resolve(vnfn, base);

    this->file->Close();
    if (!this->file->Open(vnfn, vislib::sys::File::READ_ONLY,
            vislib::sys::File::SHARE_READ,
            vislib::sys::File::OPEN_ONLY)) {

        SAFE_DELETE(this->file);
        return false;
    }

    this->buildFrameTable();
    break;

} else if (line[0] == '#') {
    // beginning of the body!
    break;
}
        */
    }
    this->setFrameCount((unsigned int)this->frameIdx.Count());
    //this->initFrameCache(1);

    return true;
}


/*
 * io::VTFDataSource::getDataCallback
 */
bool io::VTFDataSource::getDataCallback(Call& caller) {
    using geocalls::MultiParticleDataCall;
    MultiParticleDataCall* c2 = dynamic_cast<MultiParticleDataCall*>(&caller);

    Frame* f = NULL;
    if (c2 != NULL) {
        f = dynamic_cast<Frame*>(this->requestLockedFrame(c2->FrameID()));
        if (f == NULL)
            return false;
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
                                                              : MultiParticleDataCall::Particles::COLDATA_FLOAT_RGBA,
                cd, sizeof(float) * 4);
            c2->AccessParticles(i).SetClusterInfos(f->GetClusterInfos());
        }
        return true;
    }

    return false;
}


/*
 * io::VTFDataSource::getExtentCallback
 */
bool io::VTFDataSource::getExtentCallback(Call& caller) {
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
