/*
 * MMPLDDataSource.cpp
 *
 * Copyright (C) 2010 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "mmcore/moldyn/MMPLDDataSource.h"
#include "mmcore/param/FilePathParam.h"
#include "mmcore/param/BoolParam.h"
#include "mmcore/param/IntParam.h"
#include "mmcore/moldyn/MultiParticleDataCall.h"
#include "mmcore/CoreInstance.h"
#include "vislib/sys/Log.h"
#include "vislib/sys/FastFile.h"
#include "vislib/String.h"
#include "vislib/sys/SystemInformation.h"

using namespace megamol::core;


/* defines for the frame cache size */
// minimum number of frames in the cache (2 for interpolation; 1 for loading)
#define CACHE_SIZE_MIN 3
// maximum number of frames in the cache (just a nice number)
#define CACHE_SIZE_MAX 100000
// factor multiplied to the frame size for estimating the overhead to the pure data.
#define CACHE_FRAME_FACTOR 1.15f

/*****************************************************************************/

/*
 * moldyn::MMPLDDataSource::Frame::Frame
 */
moldyn::MMPLDDataSource::Frame::Frame(view::AnimDataModule& owner)
        : view::AnimDataModule::Frame(owner), dat() {
    // intentionally empty
}


/*
 * moldyn::MMPLDDataSource::Frame::~Frame
 */
moldyn::MMPLDDataSource::Frame::~Frame() {
    this->dat.EnforceSize(0);
}


/*
 * moldyn::MMPLDDataSource::Frame::LoadFrame
 */
bool moldyn::MMPLDDataSource::Frame::LoadFrame(vislib::sys::File *file, unsigned int idx, UINT64 size, unsigned int version) {
    this->frame = idx;
    this->fileVersion = version;
    this->dat.EnforceSize(static_cast<SIZE_T>(size));
    return (file->Read(this->dat, size) == size);
}


/*
 * moldyn::MMPLDDataSource::Frame::SetData
 */
void moldyn::MMPLDDataSource::Frame::SetData(MultiParticleDataCall& call, vislib::math::Cuboid<float> const& bbox, bool overrideBBox) {
    if (this->dat.IsEmpty()) {
        call.SetParticleListCount(0);
        return;
    }

    SIZE_T p = 0;
    float timestamp = static_cast<float>(call.FrameID());
    // HAZARD for megamol up to fc4e784dae531953ad4cd3180f424605474dd18b this reads == 102
    // which means that many MMPLDs out there with version 103 are written wrongly (no timestamp)!
    if (this->fileVersion >= 102) {
        timestamp = *this->dat.AsAt<float>(p);
        p += sizeof(float);
    }
    UINT32 plc = *this->dat.AsAt<UINT32>(p);
    p += sizeof(UINT32);
    call.SetParticleListCount(plc);
    for (UINT32 i = 0; i < plc; i++) {
        MultiParticleDataCall::Particles &pts = call.AccessParticles(i);

        UINT8 vrtType = *this->dat.AsAt<UINT8>(p); p += 1;
        UINT8 colType = *this->dat.AsAt<UINT8>(p); p += 1;
        MultiParticleDataCall::Particles::VertexDataType vrtDatType;
        MultiParticleDataCall::Particles::ColourDataType colDatType;
        SIZE_T vrtSize = 0;
        SIZE_T colSize = 0;

        switch (vrtType) {
            case 0: vrtSize = 0; vrtDatType = MultiParticleDataCall::Particles::VERTDATA_NONE; break;
            case 1: vrtSize = 12; vrtDatType = MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZ; break;
            case 2: vrtSize = 16; vrtDatType = MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZR; break;
            case 3: vrtSize = 6; vrtDatType = MultiParticleDataCall::Particles::VERTDATA_SHORT_XYZ; break;
            case 4: vrtSize = 24; vrtDatType = MultiParticleDataCall::Particles::VERTDATA_DOUBLE_XYZ; break;
            default: vrtSize = 0; vrtDatType = MultiParticleDataCall::Particles::VERTDATA_NONE; break;
        }
        if (vrtType != 0) {
            switch (colType) {
                case 0: colSize = 0; colDatType = MultiParticleDataCall::Particles::COLDATA_NONE; break;
                case 1: colSize = 3; colDatType = MultiParticleDataCall::Particles::COLDATA_UINT8_RGB; break;
                case 2: colSize = 4; colDatType = MultiParticleDataCall::Particles::COLDATA_UINT8_RGBA; break;
                case 3: colSize = 4; colDatType = MultiParticleDataCall::Particles::COLDATA_FLOAT_I; break;
                case 4: colSize = 12; colDatType = MultiParticleDataCall::Particles::COLDATA_FLOAT_RGB; break;
                case 5: colSize = 16; colDatType = MultiParticleDataCall::Particles::COLDATA_FLOAT_RGBA; break;
                case 6: colSize = 8; colDatType = MultiParticleDataCall::Particles::COLDATA_USHORT_RGBA; break;
                case 7: colSize = 8; colDatType = MultiParticleDataCall::Particles::COLDATA_DOUBLE_I; break;
                default: colSize = 0; colDatType = MultiParticleDataCall::Particles::COLDATA_NONE; break;
            }
        } else {
            colDatType = MultiParticleDataCall::Particles::COLDATA_NONE;
            colSize = 0;
        }
        unsigned int stride = static_cast<unsigned int>(vrtSize + colSize);

        if ((vrtType == 1) || (vrtType == 3) || (vrtType == 4)) {
            pts.SetGlobalRadius(*this->dat.AsAt<float>(p)); p += 4;
        } else {
            pts.SetGlobalRadius(0.05f);
        }

        if (colType == 0) {
            pts.SetGlobalColour(*this->dat.AsAt<UINT8>(p),
                *this->dat.AsAt<UINT8>(p + 1),
                *this->dat.AsAt<UINT8>(p + 2));
            p += 4;
        } else {
            pts.SetGlobalColour(192, 192, 192);
            if (colType == 3 || colType == 7) {
                pts.SetColourMapIndexValues(
                    *this->dat.AsAt<float>(p),
                    *this->dat.AsAt<float>(p + 4));
                p += 8;
            } else {
                pts.SetColourMapIndexValues(0.0f, 1.0f);
            }
        }

        pts.SetCount(*this->dat.AsAt<UINT64>(p)); p += 8;

        if (this->fileVersion >= 103) {
            auto const box = this->dat.AsAt<float>(p);
            vislib::math::Cuboid<float> bbox;
            bbox.Set(box[0], box[1], box[2], box[3], box[4], box[5]);
            pts.SetBBox(bbox);
            p += 24;
        }
        if (overrideBBox) {
            pts.SetBBox(bbox);
        }

        pts.SetVertexData(vrtDatType, this->dat.At(p), stride);
        pts.SetColourData(colDatType, this->dat.At(p + vrtSize), stride);

        p += static_cast<SIZE_T>(stride * pts.GetCount());

        if (this->fileVersion == 101) {
            // TODO: who deletes this?
            SimpleSphericalParticles::ClusterInfos *ci = new SimpleSphericalParticles::ClusterInfos();
            ci->numClusters = *this->dat.AsAt<unsigned int>(p); p += sizeof(unsigned int);
            ci->sizeofPlainData = *this->dat.AsAt<size_t>(p); p += sizeof(size_t);
            ci->plainData = (unsigned int*)malloc(ci->sizeofPlainData);
            memcpy(ci->plainData, this->dat.At(p), ci->sizeofPlainData); p += ci->sizeofPlainData;
            pts.SetClusterInfos(ci);
        }
    }

}

/*****************************************************************************/


/*
 * moldyn::MMPLDDataSource::MMPLDDataSource
 */
moldyn::MMPLDDataSource::MMPLDDataSource(void) : view::AnimDataModule(),
        filename("filename", "The path to the MMPLD file to load."),
        limitMemorySlot("limitMemory", "Limits the memory cache size"),
        limitMemorySizeSlot("limitMemorySize", "Specifies the size limit (in MegaBytes) of the memory cache"),
        overrideBBoxSlot("overrideLocalBBox", "Override local bbox"),
        getData("getdata", "Slot to request data from this data source."),
        file(NULL), frameIdx(NULL), bbox(-1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f),
        clipbox(-1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f), data_hash(0) {

    this->filename.SetParameter(new param::FilePathParam(""));
    this->filename.SetUpdateCallback(&MMPLDDataSource::filenameChanged);
    this->MakeSlotAvailable(&this->filename);

    this->limitMemorySlot << new param::BoolParam(
#if defined(_WIN64) || defined(LIN64)
        false
#else
        true
#endif
        );
    this->MakeSlotAvailable(&this->limitMemorySlot);

    this->limitMemorySizeSlot << new param::IntParam(2 * 1024, 1);
    this->MakeSlotAvailable(&this->limitMemorySizeSlot);

    this->overrideBBoxSlot << new param::BoolParam(false);
    this->MakeSlotAvailable(&this->overrideBBoxSlot);

    this->getData.SetCallback("MultiParticleDataCall", "GetData", &MMPLDDataSource::getDataCallback);
    this->getData.SetCallback("MultiParticleDataCall", "GetExtent", &MMPLDDataSource::getExtentCallback);
    this->MakeSlotAvailable(&this->getData);

    this->setFrameCount(1);
    this->initFrameCache(1);
}


/*
 * moldyn::MMPLDDataSource::~MMPLDDataSource
 */
moldyn::MMPLDDataSource::~MMPLDDataSource(void) {
    this->Release();
}


/*
 * moldyn::MMPLDDataSource::constructFrame
 */
view::AnimDataModule::Frame* moldyn::MMPLDDataSource::constructFrame(void) const {
    Frame *f = new Frame(*const_cast<moldyn::MMPLDDataSource*>(this));
    return f;
}


/*
 * moldyn::MMPLDDataSource::create
 */
bool moldyn::MMPLDDataSource::create(void) {
    return true;
}


/*
 * moldyn::MMPLDDataSource::loadFrame
 */
void moldyn::MMPLDDataSource::loadFrame(view::AnimDataModule::Frame *frame,
        unsigned int idx) {
    using vislib::sys::Log;
    Frame *f = dynamic_cast<Frame*>(frame);
    if (f == NULL) return;
    if (this->file == NULL) {
        f->Clear();
        return;
    }
    //printf("Requesting frame %u of %u frames\n", idx, this->FrameCount());
    //printf("Requesting frame %u of %u frames\n", idx, this->FrameCount());
    //Log::DefaultLog.WriteMsg(Log::LEVEL_INFO, "Requesting frame %u of %u frames\n", idx, this->FrameCount());
    ASSERT(idx < this->FrameCount());
    this->file->Seek(this->frameIdx[idx]);
    if (!f->LoadFrame(this->file, idx, this->frameIdx[idx + 1] - this->frameIdx[idx], this->fileVersion)) {
        // failed
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Unable to read frame %d from MMPLD file\n", idx);
    }
}


/*
 * moldyn::MMPLDDataSource::release
 */
void moldyn::MMPLDDataSource::release(void) {
    this->resetFrameCache();
    if (this->file != NULL) {
        vislib::sys::File *f = this->file;
        this->file = NULL;
        f->Close();
        delete f;
    }
    ARY_SAFE_DELETE(this->frameIdx);
}



/*
 * moldyn::MMPLDDataSource::filenameChanged
 */
bool moldyn::MMPLDDataSource::filenameChanged(param::ParamSlot& slot) {
    using vislib::sys::Log;
    using vislib::sys::File;
    this->resetFrameCache();
    this->bbox.Set(-1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f);
    this->clipbox = this->bbox;
    this->data_hash++;

    if (this->file == NULL) {
        this->file = new vislib::sys::FastFile();
    } else {
        this->file->Close();
    }
    ASSERT(this->filename.Param<param::FilePathParam>() != NULL);

    if (!this->file->Open(this->filename.Param<param::FilePathParam>()->Value(), File::READ_ONLY, File::SHARE_READ, File::OPEN_ONLY)) {
        this->GetCoreInstance()->Log().WriteMsg(Log::LEVEL_ERROR, "Unable to open MMPLD-File \"%s\".", vislib::StringA(
            this->filename.Param<param::FilePathParam>()->Value()).PeekBuffer());

        SAFE_DELETE(this->file);
        this->setFrameCount(1);
        this->initFrameCache(1);

        return true;
    }

#define _ERROR_OUT(MSG) Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, MSG); \
        SAFE_DELETE(this->file); \
        this->setFrameCount(1); \
        this->initFrameCache(1); \
        this->bbox.Set(-1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f); \
        this->clipbox = this->bbox; \
        return true;
#define _ASSERT_READFILE(BUFFER, BUFFERSIZE) if (this->file->Read((BUFFER), (BUFFERSIZE)) != (BUFFERSIZE)) { \
        _ERROR_OUT("Unable to read MMPLD file header"); \
    }

    char magicid[6];
    _ASSERT_READFILE(magicid, 6);
    if (::memcmp(magicid, "MMPLD", 6) != 0) {
        _ERROR_OUT("MMPLD file header id wrong");
    }
    unsigned short ver;
    _ASSERT_READFILE(&ver, 2);
    if (ver < 100 || ver > 103) {
        _ERROR_OUT("MMPLD file header version wrong");
    }
    this->fileVersion = ver;

    UINT32 frmCnt = 0;
    _ASSERT_READFILE(&frmCnt, 4);
    if (frmCnt == 0) {
        _ERROR_OUT("MMPLD file does not contain any frame information");
    }

    float box[6];
    _ASSERT_READFILE(box, 4 * 6);
    this->bbox.Set(box[0], box[1], box[2], box[3], box[4], box[5]);
    _ASSERT_READFILE(box, 4 * 6);
    this->clipbox.Set(box[0], box[1], box[2], box[3], box[4], box[5]);

    delete[] this->frameIdx;
    this->frameIdx = new UINT64[frmCnt + 1];
    _ASSERT_READFILE(this->frameIdx, 8 * (frmCnt + 1));
    double size = 0.0;
    for (UINT32 i = 0; i < frmCnt; i++) {
        size += static_cast<double>(this->frameIdx[i + 1] - this->frameIdx[i]);
    }
    size /= static_cast<double>(frmCnt);
    size *= CACHE_FRAME_FACTOR;

    UINT64 mem = vislib::sys::SystemInformation::AvailableMemorySize();
    if (this->limitMemorySlot.Param<param::BoolParam>()->Value()) {
        mem = vislib::math::Min(mem, 
            (UINT64)(this->limitMemorySizeSlot.Param<param::IntParam>()->Value())
            * (UINT64)(1024u * 1024u));
    }
    unsigned int cacheSize = static_cast<unsigned int>(mem / size);

    if (cacheSize > CACHE_SIZE_MAX) {
        cacheSize = CACHE_SIZE_MAX;
    }
    if (cacheSize < CACHE_SIZE_MIN) {
        vislib::StringA msg;
        msg.Format("Frame cache size forced to %i. Calculated size was %u.\n",
            CACHE_SIZE_MIN, cacheSize);
        this->GetCoreInstance()->Log().WriteMsg(vislib::sys::Log::LEVEL_WARN, msg);
        cacheSize = CACHE_SIZE_MIN;
    } else {
        vislib::StringA msg;
        msg.Format("Frame cache size set to %i.\n", cacheSize);
        this->GetCoreInstance()->Log().WriteMsg(vislib::sys::Log::LEVEL_INFO, msg);
    }

    this->setFrameCount(frmCnt);
    this->initFrameCache(cacheSize);

#undef _ASSERT_READFILE
#undef _ERROR_OUT

    return true;
}


/*
 * moldyn::MMPLDDataSource::getDataCallback
 */
bool moldyn::MMPLDDataSource::getDataCallback(Call& caller) {
    MultiParticleDataCall *c2 = dynamic_cast<MultiParticleDataCall*>(&caller);
    if (c2 == NULL) return false;

    Frame *f = NULL;
    if (c2 != NULL) {
        f = dynamic_cast<Frame *>(this->requestLockedFrame(c2->FrameID(), c2->IsFrameForced()));
        if (f == NULL) return false;
        c2->SetUnlocker(new Unlocker(*f));
        c2->SetFrameID(f->FrameNumber());
        c2->SetDataHash(this->data_hash);
        auto overrideBBox = this->overrideBBoxSlot.Param<param::BoolParam>()->Value();
        f->SetData(*c2, this->bbox, overrideBBox);
    }

    return true;
}


/*
 * moldyn::MMPLDDataSource::getExtentCallback
 */
bool moldyn::MMPLDDataSource::getExtentCallback(Call& caller) {
    MultiParticleDataCall *c2 = dynamic_cast<MultiParticleDataCall*>(&caller);

    if (c2 != NULL) {
        c2->SetFrameCount(this->FrameCount());
        c2->AccessBoundingBoxes().Clear();
        c2->AccessBoundingBoxes().SetObjectSpaceBBox(this->bbox);
        c2->AccessBoundingBoxes().SetObjectSpaceClipBox(this->clipbox);
        c2->SetDataHash(this->data_hash);
        return true;
    }

    return false;
}
