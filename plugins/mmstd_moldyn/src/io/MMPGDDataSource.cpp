/*
 * MMPGDDataSource.cpp
 *
 * Copyright (C) 2010 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "MMPGDDataSource.h"
#include "mmcore/param/FilePathParam.h"
#include "mmcore/moldyn/MultiParticleDataCall.h"
#include "rendering/ParticleGridDataCall.h"
#include "mmcore/CoreInstance.h"
#include "vislib/sys/Log.h"
#include "vislib/sys/FastFile.h"
#include "vislib/String.h"
#include "vislib/sys/SystemInformation.h"

using namespace megamol::stdplugin::moldyn::io;
using namespace megamol::stdplugin::moldyn;


/* defines for the frame cache size */
// minimum number of frames in the cache (2 for interpolation; 1 for loading)
#define CACHE_SIZE_MIN 3
// maximum number of frames in the cache (just a nice number)
#define CACHE_SIZE_MAX 100000
// factor multiplied to the frame size for estimating the overhead to the pure data.
#define CACHE_FRAME_FACTOR 1.15f

/*****************************************************************************/

/*
 * MMPGDDataSource::Frame::Frame
 */
MMPGDDataSource::Frame::Frame(view::AnimDataModule& owner)
        : view::AnimDataModule::Frame(owner), dat(), types(NULL), cells(NULL) {
    // intentionally empty
}


/*
 * MMPGDDataSource::Frame::~Frame
 */
MMPGDDataSource::Frame::~Frame() {
    ARY_SAFE_DELETE(this->cells);
    ARY_SAFE_DELETE(this->types);
    this->dat.EnforceSize(0);
}


/*
 * MMPGDDataSource::Frame::LoadFrame
 */
bool MMPGDDataSource::Frame::LoadFrame(vislib::sys::File *file, unsigned int idx, UINT64 size) {
    this->frame = idx;
    ARY_SAFE_DELETE(this->cells);
    ARY_SAFE_DELETE(this->types);
    this->dat.EnforceSize(static_cast<SIZE_T>(size));
    return (file->Read(this->dat, size) == size);
}


/*
 * MMPGDDataSource::Frame::SetData
 */
void MMPGDDataSource::Frame::SetData(rendering::ParticleGridDataCall& call) {
    if (this->dat.IsEmpty()) {
        call.SetFrameID(0);
        call.SetTypeDataRef(0, NULL);
        call.SetGridDataRef(0, 0, 0, NULL);
        return;
    }

    UINT32 *headerdat = this->dat.As<UINT32>();
    UINT32 &typeCnt = headerdat[0];
    UINT32 &cellX = headerdat[1];
    UINT32 &cellY = headerdat[2];
    UINT32 &cellZ = headerdat[3];

    SIZE_T pos = 4 * 4;
    if ((this->types == NULL) || (this->cells == NULL)) {
        // also do this if 'cells' is NULL to get the right 'pos'
        if (this->types == NULL) {
            this->types = new rendering::ParticleGridDataCall::ParticleType[typeCnt];
        }
        for (UINT32 i = 0; i < typeCnt; i++) {
            UINT8 vt = *this->dat.AsAt<UINT8>(pos); pos++;
            UINT8 ct = *this->dat.AsAt<UINT8>(pos); pos++;

            switch(vt) {
                case 0: this->types[i].SetVertexDataType(core::moldyn::MultiParticleDataCall::Particles::VERTDATA_NONE); break;
                case 1: this->types[i].SetVertexDataType(core::moldyn::MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZ); break;
                case 2: this->types[i].SetVertexDataType(core::moldyn::MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZR); break;
                case 3: this->types[i].SetVertexDataType(core::moldyn::MultiParticleDataCall::Particles::VERTDATA_SHORT_XYZ); break;
                default: this->types[i].SetVertexDataType(core::moldyn::MultiParticleDataCall::Particles::VERTDATA_NONE); break;
            }
            if (vt == 0) {
                ct = 0;
            }
            switch(ct) {
                case 0: this->types[i].SetColourDataType(core::moldyn::MultiParticleDataCall::Particles::COLDATA_NONE); break;
                case 1: this->types[i].SetColourDataType(core::moldyn::MultiParticleDataCall::Particles::COLDATA_UINT8_RGB); break;
                case 2: this->types[i].SetColourDataType(core::moldyn::MultiParticleDataCall::Particles::COLDATA_UINT8_RGBA); break;
                case 3: this->types[i].SetColourDataType(core::moldyn::MultiParticleDataCall::Particles::COLDATA_FLOAT_I); break;
                case 4: this->types[i].SetColourDataType(core::moldyn::MultiParticleDataCall::Particles::COLDATA_FLOAT_RGB); break;
                case 5: this->types[i].SetColourDataType(core::moldyn::MultiParticleDataCall::Particles::COLDATA_FLOAT_RGBA); break;
                default: this->types[i].SetColourDataType(core::moldyn::MultiParticleDataCall::Particles::COLDATA_NONE); break;
            }

            if ((vt == 1) || (vt == 3)) {
                this->types[i].SetGlobalRadius(*this->dat.AsAt<float>(pos)); pos += 4;
            } else {
                this->types[i].SetGlobalRadius(0.05f);
            }
            if (ct == 0) {
                this->types[i].SetGlobalColour(this->dat.AsAt<unsigned char>(pos)); pos += 4;
            } else {
                this->types[i].SetGlobalColour(192, 192, 192);
            }
            if (ct == 3) {
                this->types[i].SetColourMapIndexValues(
                    this->dat.AsAt<float>(pos)[0],
                    this->dat.AsAt<float>(pos)[1]);
                pos += 8;
            } else {
                this->types[i].SetColourMapIndexValues(0.0f, 1.0f);
            }
        }
    }
    call.SetTypeDataRef(typeCnt, this->types);

    if (this->cells == NULL) {
        this->cells = new rendering::ParticleGridDataCall::GridCell[cellX * cellY * cellZ];
        for (UINT32 i = 0; i < cellX * cellY * cellZ; i++) {
            rendering::ParticleGridDataCall::GridCell& cell = this->cells[i];
            const float *bbox = this->dat.AsAt<float>(pos);
            pos += 6 * 4;
            cell.AllocateParticleLists(typeCnt);
            cell.SetBoundingBox(vislib::math::Cuboid<float>(bbox[0], bbox[1], bbox[2], bbox[3], bbox[4], bbox[5]));
            for (UINT32 t = 0; t < typeCnt; t++) {
				rendering::ParticleGridDataCall::ParticleType& type = this->types[t];
				rendering::ParticleGridDataCall::Particles& points = cell.AccessParticleLists()[t];

                unsigned int vs = 0, cs = 0;
                switch(type.GetVertexDataType()) {
                    case core::moldyn::MultiParticleDataCall::Particles::VERTDATA_NONE: vs = 0; break;
                    case core::moldyn::MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZ: vs = 12; break;
                    case core::moldyn::MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZR: vs = 16; break;
                    case core::moldyn::MultiParticleDataCall::Particles::VERTDATA_SHORT_XYZ: vs = 6; break;
                    default: vs = 0; break;
                }
                if (vs != 0) {
                    switch(type.GetColourDataType()) {
                        case core::moldyn::MultiParticleDataCall::Particles::COLDATA_NONE: cs = 0; break;
                        case core::moldyn::MultiParticleDataCall::Particles::COLDATA_UINT8_RGB: cs = 3; break;
                        case core::moldyn::MultiParticleDataCall::Particles::COLDATA_UINT8_RGBA: cs = 4; break;
                        case core::moldyn::MultiParticleDataCall::Particles::COLDATA_FLOAT_I: cs = 4; break;
                        case core::moldyn::MultiParticleDataCall::Particles::COLDATA_FLOAT_RGB: cs = 12; break;
                        case core::moldyn::MultiParticleDataCall::Particles::COLDATA_FLOAT_RGBA: cs = 16; break;
                        default: cs = 0; break;
                    }
                } else {
                    cs = 0;
                }

                UINT64& cnt = *this->dat.AsAt<UINT64>(pos);
                pos += 8;
                points.SetCount(static_cast<SIZE_T>(cnt));
                points.SetMaxRadius(*this->dat.AsAt<float>(pos));
                pos += 4;
                points.SetVertexData(this->dat.At(pos), static_cast<unsigned int>(vs + cs));
                points.SetColourData(this->dat.At(pos + vs), static_cast<unsigned int>(vs + cs));
                pos += static_cast<SIZE_T>(cnt) * (cs + vs);
            }
        }
    }
    call.SetGridDataRef(cellX, cellY, cellZ, this->cells);

}

/*****************************************************************************/


/*
 * MMPGDDataSource::MMPGDDataSource
 */
MMPGDDataSource::MMPGDDataSource(void) : view::AnimDataModule(),
        filename("filename", "The path to the MMPGD file to load."),
        getData("getdata", "Slot to request data from this data source."),
        file(NULL), frameIdx(NULL), bbox(-1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f),
        clipbox(-1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f) {

    this->filename.SetParameter(new param::FilePathParam(""));
    this->filename.SetUpdateCallback(&MMPGDDataSource::filenameChanged);
    this->MakeSlotAvailable(&this->filename);

    this->getData.SetCallback("ParticleGridDataCall", "GetData", &MMPGDDataSource::getDataCallback);
    this->getData.SetCallback("ParticleGridDataCall", "GetExtent", &MMPGDDataSource::getExtentCallback);
    this->MakeSlotAvailable(&this->getData);

    this->setFrameCount(1);
    this->initFrameCache(1);
}


/*
 * MMPGDDataSource::~MMPGDDataSource
 */
MMPGDDataSource::~MMPGDDataSource(void) {
    this->Release();
}


/*
 * MMPGDDataSource::constructFrame
 */
view::AnimDataModule::Frame* MMPGDDataSource::constructFrame(void) const {
    Frame *f = new Frame(*const_cast<MMPGDDataSource*>(this));
    return f;
}


/*
 * MMPGDDataSource::create
 */
bool MMPGDDataSource::create(void) {
    return true;
}


/*
 * MMPGDDataSource::loadFrame
 */
void MMPGDDataSource::loadFrame(view::AnimDataModule::Frame *frame,
        unsigned int idx) {
    using vislib::sys::Log;
    Frame *f = dynamic_cast<Frame*>(frame);
    if (f == NULL) return;
    if (this->file == NULL) {
        f->Clear();
        return;
    }
    ASSERT(idx < this->FrameCount());
    this->file->Seek(this->frameIdx[idx]);
    if (!f->LoadFrame(this->file, idx, this->frameIdx[idx + 1] - this->frameIdx[idx])) {
        // failed
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Unable to read frame %d from MMPGD file\n", idx);
    }
}


/*
 * MMPGDDataSource::release
 */
void MMPGDDataSource::release(void) {
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
 * MMPGDDataSource::filenameChanged
 */
bool MMPGDDataSource::filenameChanged(param::ParamSlot& slot) {
    using vislib::sys::Log;
    using vislib::sys::File;
    this->resetFrameCache();
    this->bbox.Set(-1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f);
    this->clipbox = this->bbox;

    if (this->file == NULL) {
        this->file = new vislib::sys::FastFile();
    } else {
        this->file->Close();
    }
    ASSERT(this->filename.Param<param::FilePathParam>() != NULL);

    if (!this->file->Open(this->filename.Param<param::FilePathParam>()->Value(), File::READ_ONLY, File::SHARE_READ, File::OPEN_ONLY)) {
        this->GetCoreInstance()->Log().WriteMsg(Log::LEVEL_ERROR, "Unable to open MMPGD-File \"%s\".", vislib::StringA(
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
        _ERROR_OUT("Unable to read MMPGD file header"); \
    }

    char magicid[6];
    _ASSERT_READFILE(magicid, 6);
    if (::memcmp(magicid, "MMPGD", 6) != 0) {
        _ERROR_OUT("MMPGD file header id wrong");
    }
    unsigned short ver;
    _ASSERT_READFILE(&ver, 2);
    if (ver != 100) {
        _ERROR_OUT("MMPGD file header version wrong");
    }

    UINT32 frmCnt = 0;
    _ASSERT_READFILE(&frmCnt, 4);
    if (frmCnt == 0) {
        _ERROR_OUT("MMPGD file does not contain any frame information");
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
 * MMPGDDataSource::getDataCallback
 */
bool MMPGDDataSource::getDataCallback(Call& caller) {
	rendering::ParticleGridDataCall *c2 = dynamic_cast<rendering::ParticleGridDataCall*>(&caller);
    if (c2 == NULL) return false;

    Frame *f = NULL;
    if (c2 != NULL) {
        f = dynamic_cast<Frame *>(this->requestLockedFrame(c2->FrameID()));
        if (f == NULL) return false;
        c2->SetUnlocker(new Unlocker(*f));
        c2->SetFrameID(f->FrameNumber());
        c2->SetDataHash(0);
        f->SetData(*c2);
    }

    return true;
}


/*
 * MMPGDDataSource::getExtentCallback
 */
bool MMPGDDataSource::getExtentCallback(Call& caller) {
	rendering::ParticleGridDataCall *c2 = dynamic_cast<rendering::ParticleGridDataCall*>(&caller);

    if (c2 != NULL) {
        c2->SetFrameCount(this->FrameCount());
        c2->AccessBoundingBoxes().Clear();
        c2->AccessBoundingBoxes().SetObjectSpaceBBox(this->bbox);
        c2->AccessBoundingBoxes().SetObjectSpaceClipBox(this->clipbox);
        return true;
    }

    return false;
}
