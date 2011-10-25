/*
 * MMSPDDataSource.cpp
 *
 * Copyright (C) 2011 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "MMSPDDataSource.h"
#include "param/FilePathParam.h"
#include "MultiParticleDataCall.h"
#include "CoreInstance.h"
//#include "vislib/Log.h"
#include "vislib/File.h"
//#include "vislib/String.h"
//#include "vislib/SystemInformation.h"
#include "vislib/mathfunctions.h"

using namespace megamol::core;


/* defines for the frame cache size */
// minimum number of frames in the cache (2 for interpolation; 1 for loading)
#define CACHE_SIZE_MIN 3
// maximum number of frames in the cache (just a nice number)
#define CACHE_SIZE_MAX 1000
// factor multiplied to the frame size for estimating the overhead to the pure data.
#define CACHE_FRAME_FACTOR 1.2f

/*****************************************************************************/

/*
 * moldyn::MMSPDDataSource::Frame::Frame
 */
moldyn::MMSPDDataSource::Frame::Frame(view::AnimDataModule& owner)
        : moldyn::MMSPDFrameData(), view::AnimDataModule::Frame(owner) {
    // intentionally empty
}


/*
 * moldyn::MMSPDDataSource::Frame::~Frame
 */
moldyn::MMSPDDataSource::Frame::~Frame() {
    // intentionally empty
}


///*
// * moldyn::MMPLDDataSource::Frame::LoadFrame
// */
//bool moldyn::MMPLDDataSource::Frame::LoadFrame(vislib::sys::File *file, unsigned int idx, UINT64 size) {
//    this->frame = idx;
//    this->dat.EnforceSize(static_cast<SIZE_T>(size));
//    return (file->Read(this->dat, size) == size);
//}
//
//
///*
// * moldyn::MMPLDDataSource::Frame::SetData
// */
//void moldyn::MMPLDDataSource::Frame::SetData(MultiParticleDataCall& call) {
//    if (this->dat.IsEmpty()) {
//        call.SetParticleListCount(0);
//        return;
//    }
//
//    SIZE_T p = sizeof(UINT32);
//    UINT32 plc = *this->dat.As<UINT32>();
//    call.SetParticleListCount(plc);
//    for (UINT32 i = 0; i < plc; i++) {
//        MultiParticleDataCall::Particles &pts = call.AccessParticles(i);
//
//        UINT8 vrtType = *this->dat.AsAt<UINT8>(p); p += 1;
//        UINT8 colType = *this->dat.AsAt<UINT8>(p); p += 1;
//        MultiParticleDataCall::Particles::VertexDataType vrtDatType;
//        MultiParticleDataCall::Particles::ColourDataType colDatType;
//        SIZE_T vrtSize = 0;
//        SIZE_T colSize = 0;
//
//        switch (vrtType) {
//            case 0: vrtSize = 0; vrtDatType = MultiParticleDataCall::Particles::VERTDATA_NONE; break;
//            case 1: vrtSize = 12; vrtDatType = MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZ; break;
//            case 2: vrtSize = 16; vrtDatType = MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZR; break;
//            case 3: vrtSize = 6; vrtDatType = MultiParticleDataCall::Particles::VERTDATA_SHORT_XYZ; break;
//            default: vrtSize = 0; vrtDatType = MultiParticleDataCall::Particles::VERTDATA_NONE; break;
//        }
//        if (vrtType != 0) {
//            switch (colType) {
//                case 0: colSize = 0; colDatType = MultiParticleDataCall::Particles::COLDATA_NONE; break;
//                case 1: colSize = 3; colDatType = MultiParticleDataCall::Particles::COLDATA_UINT8_RGB; break;
//                case 2: colSize = 4; colDatType = MultiParticleDataCall::Particles::COLDATA_UINT8_RGBA; break;
//                case 3: colSize = 4; colDatType = MultiParticleDataCall::Particles::COLDATA_FLOAT_I; break;
//                case 4: colSize = 12; colDatType = MultiParticleDataCall::Particles::COLDATA_FLOAT_RGB; break;
//                case 5: colSize = 16; colDatType = MultiParticleDataCall::Particles::COLDATA_FLOAT_RGBA; break;
//                default: colSize = 0; colDatType = MultiParticleDataCall::Particles::COLDATA_NONE; break;
//            }
//        } else {
//            colDatType = MultiParticleDataCall::Particles::COLDATA_NONE;
//            colSize = 0;
//        }
//        unsigned int stride = static_cast<unsigned int>(vrtSize + colSize);
//
//        if ((vrtType == 1) || (vrtType == 3)) {
//            pts.SetGlobalRadius(*this->dat.AsAt<float>(p)); p += 4;
//        } else {
//            pts.SetGlobalRadius(0.05f);
//        }
//
//        if (colType == 0) {
//            pts.SetGlobalColour(*this->dat.AsAt<UINT8>(p),
//                *this->dat.AsAt<UINT8>(p + 1),
//                *this->dat.AsAt<UINT8>(p + 2));
//            p += 4;
//        } else {
//            pts.SetGlobalColour(192, 192, 192);
//            if (colType == 3) {
//                pts.SetColourMapIndexValues(
//                    *this->dat.AsAt<float>(p),
//                    *this->dat.AsAt<float>(p + 4));
//                p += 8;
//            } else {
//                pts.SetColourMapIndexValues(0.0f, 1.0f);
//            }
//        }
//
//        pts.SetCount(*this->dat.AsAt<UINT64>(p)); p += 8;
//
//        pts.SetVertexData(vrtDatType, this->dat.At(p), stride);
//        pts.SetColourData(colDatType, this->dat.At(p + vrtSize), stride);
//
//    }
//
//}

/*****************************************************************************/

/*
 * moldyn::MMSPDDataSource::FileFormatAutoDetect
 */
float moldyn::MMSPDDataSource::FileFormatAutoDetect(const unsigned char* data, SIZE_T dataSize) {
    return (((dataSize >= 6)
        && ((::memcmp(data, "MMSPDb", 6) == 0)
            || (::memcmp(data, "MMSPDa", 6) == 0)
            || (::memcmp(data, "MMSPDu", 6) == 0)))
        || ((dataSize >= 9)
        && (::memcmp(data, "\xEF\xBB\xBFMMSPDu", 9) == 0))) ? 1.0f : 0.0f;
}


/*
 * moldyn::MMSPDDataSource::MMSPDDataSource
 */
moldyn::MMSPDDataSource::MMSPDDataSource(void) : view::AnimDataModule(),
        filename("filename", "The path to the MMSPD file to load."),
        getData("getdata", "Slot to request data from this data source."),
        dataHeader(), file(NULL), frameIdx(NULL),
        clipbox(-1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f) {

    this->filename.SetParameter(new param::FilePathParam(""));
    this->filename.SetUpdateCallback(&MMSPDDataSource::filenameChanged);
    this->MakeSlotAvailable(&this->filename);

    this->getData.SetCallback("MultiParticleDataCall", "GetData", &MMSPDDataSource::getDataCallback);
    this->getData.SetCallback("MultiParticleDataCall", "GetExtent", &MMSPDDataSource::getExtentCallback);
    this->MakeSlotAvailable(&this->getData);

    this->setFrameCount(1);
    this->initFrameCache(1);
}


/*
 * moldyn::MMSPDDataSource::~MMSPDDataSource
 */
moldyn::MMSPDDataSource::~MMSPDDataSource(void) {
    this->Release();
}


/*
 * moldyn::MMSPDDataSource::constructFrame
 */
view::AnimDataModule::Frame* moldyn::MMSPDDataSource::constructFrame(void) const {
    Frame *f = new Frame(*const_cast<moldyn::MMSPDDataSource*>(this));
    return f;
}


/*
 * moldyn::MMSPDDataSource::create
 */
bool moldyn::MMSPDDataSource::create(void) {
    return true;
}


/*
 * moldyn::MMSPDDataSource::loadFrame
 */
void moldyn::MMSPDDataSource::loadFrame(view::AnimDataModule::Frame *frame,
        unsigned int idx) {
    using vislib::sys::Log;
    Frame *f = dynamic_cast<Frame*>(frame);
    if (f == NULL) return;
    if (this->file == NULL) {
        //f->Clear();
        return;
    }
    //printf("Requesting frame %u of %u frames\n", idx, this->FrameCount());
    ASSERT(idx < this->FrameCount());
    this->file->Seek(this->frameIdx[idx]);
    //if (!f->LoadFrame(this->file, idx, this->frameIdx[idx + 1] - this->frameIdx[idx])) {
    //    // failed
    //    Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Unable to read frame %d from MMPLD file\n", idx);
    //}
}


/*
 * moldyn::MMSPDDataSource::release
 */
void moldyn::MMSPDDataSource::release(void) {
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
 * moldyn::MMSPDDataSource::filenameChanged
 */
bool moldyn::MMSPDDataSource::filenameChanged(param::ParamSlot& slot) {
    using vislib::sys::Log;
    using vislib::sys::File;
    this->resetFrameCache();
    this->dataHeader.SetParticleCount(0);
    this->dataHeader.SetTimeCount(1);
    this->dataHeader.BoundingBox().Set(-1.0, -1.0, -1.0, 1.0, 1.0, 1.0);
    this->clipbox = this->dataHeader.GetBoundingBox();

    if (this->file == NULL) {
        this->file = new vislib::sys::File();
    } else {
        this->file->Close();
    }
    ASSERT(this->filename.Param<param::FilePathParam>() != NULL);

    if (!this->file->Open(this->filename.Param<param::FilePathParam>()->Value(), File::READ_ONLY, File::SHARE_READ, File::OPEN_ONLY)) {
        this->GetCoreInstance()->Log().WriteMsg(Log::LEVEL_ERROR, "Unable to open MMSPD-File \"%s\".", vislib::StringA(
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
        this->dataHeader.SetParticleCount(0); \
        this->dataHeader.SetTimeCount(1); \
        this->dataHeader.BoundingBox().Set(-1.0, -1.0, -1.0, 1.0, 1.0, 1.0); \
        this->clipbox = this->dataHeader.GetBoundingBox(); \
        return true;
#define _ASSERT_READFILE(BUFFER, BUFFERSIZE) if (this->file->Read((BUFFER), (BUFFERSIZE)) != (BUFFERSIZE)) { \
        _ERROR_OUT("Unable to read MMSPD file: seems truncated"); \
    }

    // reading format marker
    BYTE headerID[9];
    _ASSERT_READFILE(headerID, 9);
    bool jmpBk, text, unicode;
    if ((text = (::memcmp(headerID, "MMSPDb", 6) != 0))
            && (unicode = (::memcmp(headerID, "MMSPDa", 6) != 0))
            && (::memcmp(headerID, "MMSPDu", 6) != 0)
            && (jmpBk = (::memcmp(headerID, "\xEF\xBB\xBFMMSPDu", 9) != 0))) {
        _ERROR_OUT("MMSPD format marker not found");
    }
    if (jmpBk) {
        this->file->Seek(-3, vislib::sys::File::CURRENT);
    }
    // TODO: Version number

    // reading header line

    // reading particle types

    // reading frames
    //  index generation and size estimation


/*

    char magicid[6];
    _ASSERT_READFILE(magicid, 6);
    if (::memcmp(magicid, "MMPLD", 6) != 0) {
        _ERROR_OUT("MMPLD file header id wrong");
    }
    unsigned short ver;
    _ASSERT_READFILE(&ver, 2);
    if (ver != 100) {
        _ERROR_OUT("MMPLD file header version wrong");
    }

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
    */

#undef _ASSERT_READFILE
#undef _ERROR_OUT

    return true;
}


/*
 * moldyn::MMSPDDataSource::getDataCallback
 */
bool moldyn::MMSPDDataSource::getDataCallback(Call& caller) {
    MultiParticleDataCall *c2 = dynamic_cast<MultiParticleDataCall*>(&caller);
    /*if (c2 == NULL)*/ return false;

    //Frame *f = NULL;
    //if (c2 != NULL) {
    //    f = dynamic_cast<Frame *>(this->requestLockedFrame(c2->FrameID()));
    //    if (f == NULL) return false;
    //    c2->SetUnlocker(new Unlocker(*f));
    //    c2->SetFrameID(f->FrameNumber());
    //    c2->SetDataHash(0);
    //    f->SetData(*c2);
    //}

    //return true;
}


/*
 * moldyn::MMSPDDataSource::getExtentCallback
 */
bool moldyn::MMSPDDataSource::getExtentCallback(Call& caller) {
    MultiParticleDataCall *c2 = dynamic_cast<MultiParticleDataCall*>(&caller);

    if (c2 != NULL) {
        c2->SetFrameCount(vislib::math::Max(1u, this->dataHeader.GetTimeCount()));
        c2->AccessBoundingBoxes().Clear();
        c2->AccessBoundingBoxes().SetObjectSpaceBBox(this->dataHeader.GetBoundingBox());
        c2->AccessBoundingBoxes().SetObjectSpaceClipBox(this->clipbox);
        c2->SetUnlocker(NULL);
        return true;
    }

    return false;
}
