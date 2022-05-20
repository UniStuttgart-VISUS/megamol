/*
 * VIMDataSource.cpp
 *
 * Copyright (C) 2009 by Universitaet Stuttgart (VISUS).
 * Alle Rechte vorbehalten.
 */

#include "io/VIMDataSource.h"
#include "geometry_calls/EllipsoidalDataCall.h"
#include "mmcore/CoreInstance.h"
#include "mmcore/param/FilePathParam.h"
#include "mmcore/param/StringParam.h"
#include "mmcore/utility/log/Log.h"
#include "stdafx.h"
#include "vislib/PtrArray.h"
#include "vislib/RawStorageWriter.h"
#include "vislib/String.h"
#include "vislib/StringTokeniser.h"
#include "vislib/Trace.h"
#include "vislib/math/ShallowPoint.h"
#include "vislib/math/ShallowQuaternion.h"
#include "vislib/sys/Path.h"
#include "vislib/sys/SystemInformation.h"
#include "vislib/sys/error.h"
#include "vislib/sys/sysfunctions.h"
#include <cassert>

using namespace megamol;
using namespace megamol::moldyn::io;


/* defines for the frame cache size */
// minimum number of frames in the cache (2 for interpolation; 1 for loading)
#define CACHE_SIZE_MIN 3
// maximum number of frames in the cache (just a nice number)
#define CACHE_SIZE_MAX 100000
// factor multiplied to the frame size for estimating the overhead to the pure data.
#define CACHE_FRAME_FACTOR 1.15f

/*****************************************************************************/

/*
 * VIMDataSource::Frame::Frame
 */
VIMDataSource::Frame::Frame(core::view::AnimDataModule& owner)
        : core::view::AnimDataModule::Frame(owner)
        , typeCnt(0)
        , partCnt(NULL)
        , pos(NULL)
        , quat(NULL)
        , radii() {}


/*
 * VIMDataSource::Frame::~Frame
 */
VIMDataSource::Frame::~Frame() {
    this->typeCnt = 0;
    ARY_SAFE_DELETE(this->partCnt);
    ARY_SAFE_DELETE(this->pos);
    ARY_SAFE_DELETE(this->quat);
}


/*
 * VIMDataSource::Frame::Clear
 */
void VIMDataSource::Frame::Clear(void) {
    for (unsigned int i = 0; i < this->typeCnt; i++) {
        this->pos[i].EnforceSize(0);
        this->quat[i].EnforceSize(0);
    }
    this->radii.clear();
}


/*
 * VIMDataSource::Frame::LoadFrame
 */
bool VIMDataSource::Frame::LoadFrame(
    vislib::sys::File* file, unsigned int idx, VIMDataSource::SimpleType* types, float /*scaling*/) {

    float lScale = 1.0f;

    // clear frame by resetting the counters
    for (unsigned int i = 0; i < this->typeCnt; i++) {
        this->partCnt[i] = 0;
    }
    this->radii.clear();

    // read frame
    vislib::StringA startLine = vislib::sys::ReadLineFromFileA(*file);

    if (startLine[0] != '#') {
        megamol::core::utility::log::Log::DefaultLog.WriteMsg(
            megamol::core::utility::log::Log::LEVEL_ERROR, "Invalid Start Line Parsed");
        return false;
    }

    if (startLine[1] == '#') {
        megamol::core::utility::log::Log::DefaultLog.WriteMsg(
            megamol::core::utility::log::Log::LEVEL_WARN, "Unexpected End of Data");
        return false;
    }

    startLine.Remove(0, 1);
    startLine.TrimSpacesBegin();

    try {
        lScale = static_cast<float>(vislib::CharTraitsA::ParseDouble(startLine.PeekBuffer()));
    } catch (...) { lScale = 1.0f; }

    this->frame = idx;

    vislib::PtrArray<vislib::RawStorageWriter> posWrtr(this->typeCnt);
    vislib::PtrArray<vislib::RawStorageWriter> quatWrtr(this->typeCnt);
    posWrtr.SetCount(this->typeCnt);
    quatWrtr.SetCount(this->typeCnt);
    for (unsigned int i = 0; i < this->typeCnt; i++) {
        posWrtr[i] = new vislib::RawStorageWriter(this->pos[i]);
        quatWrtr[i] = new vislib::RawStorageWriter(this->quat[i]);
    }

    // TODO: Implement a faster code (here it matters!);
    // however, it is VIM and thus not very important.
    while (!file->IsEOF()) {
        vislib::StringA line = vislib::sys::ReadLineFromFileA(*file);
        line.TrimSpaces();

        if (line.IsEmpty()) {
            continue;
        } else if (line[0] == '!') {
            // a type line!
            int type;
            float x, y, z, qx, qy, qz, qw;
            try {
                this->parseParticleLine(line, type, x, y, z, qx, qy, qz, qw);
            } catch (...) {
                megamol::core::utility::log::Log::DefaultLog.WriteMsg(50, "Unable to parse particle line");
                continue;
            }
            unsigned int idx = 0;
            for (unsigned int i = 0; i < this->typeCnt; i++) {
                if (types[i].ID() == static_cast<unsigned int>(type)) {
                    idx = i;
                    break;
                }
            }
            unsigned int h = this->partCnt[idx] + 1;
            if (h % 50) {
                h += 50 - h % 50;
            }

            int oldTraceLvl = vislib::Trace::GetInstance().GetLevel();
            vislib::Trace::GetInstance().SetLevel(vislib::Trace::LEVEL_NONE);
            (*posWrtr[idx]) << x << y << z;
            (*quatWrtr[idx]) << qx << qy << qz << qw;
            vislib::Trace::GetInstance().SetLevel(oldTraceLvl);

            this->partCnt[idx]++;

        } else if (line[0] == '#') {
            // frame done
            break;
        }
    }

    for (unsigned int i = 0; i < this->typeCnt; i++) {
        this->pos[i].EnforceSize(posWrtr[i]->End(), true);
        this->quat[i].EnforceSize(quatWrtr[i]->End(), true);

        // de-quantize positions
        float maxVal = 0.0f;
        for (SIZE_T j = 0; j < this->pos[i].GetSize(); j += sizeof(float)) {
            if (maxVal < *this->pos[i].AsAt<float>(j))
                maxVal = *this->pos[i].AsAt<float>(j);
        }
        float deQuant = lScale;
        if (maxVal >= 1000) {
            deQuant /= 10000.0f;
        } else {
            deQuant /= 1000.0f;
        }
        for (SIZE_T j = 0; j < this->pos[i].GetSize(); j += sizeof(float)) {
            *this->pos[i].AsAt<float>(j) *= deQuant;
        }
    }
    VLTRACE(VISLIB_TRCELVL_INFO, "Frame %u loaded\n", this->frame);

    return true;
}


/*
 * VIMDataSource::Frame::SetTypeCount
 */
void VIMDataSource::Frame::SetTypeCount(unsigned int cnt) {
    this->typeCnt = cnt;
    delete[] this->partCnt;
    delete[] this->pos;
    delete[] this->quat;
    this->partCnt = new unsigned int[cnt];
    this->pos = new vislib::RawStorage[cnt];
    this->quat = new vislib::RawStorage[cnt];
    this->radii.clear();
    for (unsigned int i = 0; i < cnt; i++) {
        this->partCnt[i] = 0;
    }
}


/*
 * VIMDataSource::Frame::PartPoss
 */
const float* VIMDataSource::Frame::PartPoss(unsigned int type) const {
    ASSERT(type < this->typeCnt);
    return this->pos[type].As<float>();
}


/*
 * VIMDataSource::Frame::PartQuats
 */
const float* VIMDataSource::Frame::PartQuats(unsigned int type) const {
    ASSERT(type < this->typeCnt);
    return this->quat[type].As<float>();
}


/*
 * VIMDataSource::Frame::PartRadii
 */
const float* VIMDataSource::Frame::PartRadii(unsigned int type, SimpleType& t) const {
    if (this->radii.size() <= type) {
        // must reconstruct radii data
        this->radii.resize(this->typeCnt);
        for (unsigned int type_i = 0; type_i < this->typeCnt; ++type_i) {
            this->radii[type_i].EnforceSize(0);
        }
    }
    if (this->radii[type].GetSize() < sizeof(float) * 3 * this->partCnt[type]) {
        this->radii[type].EnforceSize(sizeof(float) * 3 * this->partCnt[type]);

        float radi[3];

        radi[0] = t.Radius();
        radi[1] = 0.666f * t.Radius();
        radi[2] = 0.333f * t.Radius();

        for (unsigned int p_i = 0; p_i < this->partCnt[type]; ++p_i) {
            float* f = this->radii[type].AsAt<float>(sizeof(float) * 3 * p_i);
            memcpy(f, radi, sizeof(float) * 3);
        }
    }

    return this->radii[type].As<const float>();
}


/*
 * VIMDataSource::Frame::SizeOf
 */
SIZE_T VIMDataSource::Frame::SizeOf(void) const {
    SIZE_T size = 0;
    for (unsigned int i = 0; i < this->typeCnt; i++) {
        size += this->pos[i].GetSize();
        size += this->quat[i].GetSize();
    }
    return size;
}


/*
 * VIMDataSource::Frame::MakeInterpolationFrame
 */
const VIMDataSource::Frame* VIMDataSource::Frame::MakeInterpolationFrame(
    float alpha, const VIMDataSource::Frame& a, const VIMDataSource::Frame& b) {
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
        this->quat[t].AssertSize(this->partCnt[t] * 4 * sizeof(float));
    }

    for (unsigned int t = 0; t < a.typeCnt; t++) {
        for (unsigned int i = 0; i < this->partCnt[t]; i++) {
            vislib::math::ShallowPoint<float, 3> av(a.pos[t].As<float>() + i * 3);
            vislib::math::ShallowPoint<float, 3> bv(b.pos[t].As<float>() + i * 3);
            vislib::math::ShallowPoint<float, 3> tv(this->pos[t].As<float>() + i * 3);

            if (av.SquareDistance(bv) > 0.01) {
                tv = (alpha < 0.5f) ? av : bv;
            } else {
                tv.Set(av.X() * beta + bv.X() * alpha, av.Y() * beta + bv.Y() * alpha, av.Z() * beta + bv.Z() * alpha);
            }
        }
        for (unsigned int i = 0; i < this->partCnt[t]; i++) {
            vislib::math::ShallowQuaternion<float>(this->quat[t].As<float>() + i * 4)
                .Slerp(alpha, vislib::math::ShallowQuaternion<float>(a.quat[t].As<float>() + i * 4),
                    vislib::math::ShallowQuaternion<float>(b.quat[t].As<float>() + i * 4));
        }
    }

    return this;
}


/*
 * VIMDataSource::Frame::parseParticleLine
 */
void VIMDataSource::Frame::parseParticleLine(vislib::StringA& line, int& outType, float& outX, float& outY, float& outZ,
    float& outQX, float& outQY, float& outQZ, float& outQW) {
#define MAPPED_Q1 outQW
#define MAPPED_Q2 outQX
#define MAPPED_Q3 outQY
#define MAPPED_Q4 outQZ

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
    if (shreds.Count() == 9) {
        MAPPED_Q1 = float(
            vislib::CharTraitsA::ParseInt(shreds[5])); // quaternions are automatically de-quantized thru normalization
        MAPPED_Q2 = float(vislib::CharTraitsA::ParseInt(shreds[6]));
        MAPPED_Q3 = float(vislib::CharTraitsA::ParseInt(shreds[7]));
        MAPPED_Q4 = float(vislib::CharTraitsA::ParseInt(shreds[8]));

        // normalize quaternion
        double ql = sqrt(double(outQX) * double(outQX) + double(outQY) * double(outQY) + double(outQZ) * double(outQZ) +
                         double(outQW) * double(outQW));
        if (fabs(ql) < 0.001) {
            outQX = 0.0f;
            outQY = 0.0f;
            outQZ = 0.0f;
            outQW = 1.0f;
        } else {
            outQX = float(double(outQX) / ql); // qi x*sin(a/2)
            outQY = float(double(outQY) / ql); // qj y*sin(a/2)
            outQZ = float(double(outQZ) / ql); // qk z*sin(a/2)
            outQW = float(double(outQW) / ql); // qr   cos(a/2)
        }
    } else {
        outQX = 0.0f;
        outQY = 0.0f;
        outQZ = 0.0f;
        outQW = 1.0f;
    }

#undef MAPPED_Q1
#undef MAPPED_Q2
#undef MAPPED_Q3
#undef MAPPED_Q4
}

/*****************************************************************************/


/*
 * VIMDataSource::VIMDataSource
 */
VIMDataSource::VIMDataSource(void)
        : core::view::AnimDataModule()
        , filename("filename", "The path to the trisoup file to load.")
        , getData("getdata", "Slot to request data from this data source.")
        , file(NULL)
        , typeCnt(0)
        , types(NULL)
        , frameIdx(NULL)
        , boxScaling(1.0f)
        , datahash(0) {

    this->filename.SetParameter(new core::param::FilePathParam(""));
    this->filename.SetUpdateCallback(&VIMDataSource::filenameChanged);
    this->MakeSlotAvailable(&this->filename);

    this->getData.SetCallback("MultiParticleDataCall", "GetData", &VIMDataSource::getDataCallback);
    this->getData.SetCallback("MultiParticleDataCall", "GetExtent", &VIMDataSource::getExtentCallback);
    this->getData.SetCallback(
        geocalls::EllipsoidalParticleDataCall::ClassName(), "GetData", &VIMDataSource::getDataCallback);
    this->getData.SetCallback(
        geocalls::EllipsoidalParticleDataCall::ClassName(), "GetExtent", &VIMDataSource::getExtentCallback);
    this->MakeSlotAvailable(&this->getData);

    this->setFrameCount(1);
    this->initFrameCache(1);
}


/*
 * VIMDataSource::~VIMDataSource
 */
VIMDataSource::~VIMDataSource(void) {
    this->Release(); // implicitly calls 'release'
}


/*
 * VIMDataSource::constructFrame
 */
core::view::AnimDataModule::Frame* VIMDataSource::constructFrame(void) const {
    Frame* f = new Frame(*const_cast<VIMDataSource*>(this));
    f->SetTypeCount(this->typeCnt);
    return f;
}


/*
 * VIMDataSource::create
 */
bool VIMDataSource::create(void) {
    return true;
}


/*
 * VIMDataSource::loadFrame
 */
void VIMDataSource::loadFrame(core::view::AnimDataModule::Frame* frame, unsigned int idx) {
    Frame* f = dynamic_cast<Frame*>(frame);
    if (f == NULL)
        return;
    if (this->file == NULL) {
        f->Clear();
        return;
    }
    ASSERT(idx < this->FrameCount());

    this->file->Seek(this->frameIdx[idx]);
    f->LoadFrame(this->file, idx, this->types, this->boxScaling);
}


/*
 * VIMDataSource::release
 */
void VIMDataSource::release(void) {
    this->resetFrameCache();
    if (this->file != NULL) {
        vislib::sys::File* f = this->file;
        this->file = NULL;
        f->Close();
        delete f;
    }
    this->typeCnt = 0;
    ARY_SAFE_DELETE(this->types);
    ARY_SAFE_DELETE(this->frameIdx);
}


/*
 * VIMDataSource::buildFrameTable
 */
void VIMDataSource::buildFrameTable(void) {
    ASSERT(this->file != NULL);

    vislib::SingleLinkedList<vislib::sys::File::FileSize> framePoss;
    const unsigned int bufSize = 1024 * 1024;
    char* buf = new char[bufSize];
    unsigned int size = 1;
    char lCh1 = 0, lCh2 = 0;
    vislib::sys::File::FileSize pos = 0;
    unsigned int i;

    unsigned int frameCnt = 0;
    ARY_SAFE_DELETE(this->frameIdx);

    while (!this->file->IsEOF()) {
        size = static_cast<unsigned int>(this->file->Read(buf, bufSize));
        if (size == 0) {
            break;
        }

        if (lCh1 == '#') {
            if (buf[0] == '#') {
                break; // end of data
            }
            if ((lCh2 == 0x0D) || (lCh2 == 0x0A)) {
                framePoss.Add(pos - 1);
                frameCnt++;
            }
        }

        for (i = 0; i < size - 1; i++) {
            if (buf[i] == '#') {
                if (buf[i + 1] == '#') {
                    break; // end of data
                }
                if (((i == 0) && ((lCh1 == 0x0D) || (lCh1 == 0x0A))) ||
                    ((i > 0) && ((buf[i - 1] == 0x0D) || (buf[i - 1] == 0x0A)))) {
                    framePoss.Add(pos + i);
                    frameCnt++;
                }
            }
        }
        if ((i < size - 1) && (buf[i] == '#') && (buf[i + 1] == '#')) {
            break; // end of data
        }

        if (size > 1) {
            lCh2 = buf[size - 2];
            lCh1 = buf[size - 1];
        } else if (size == 1) {
            lCh2 = lCh1;
            lCh1 = buf[0];
        }

        pos += size;
    }

    this->file->SeekToBegin(); // seek back to the beginning of the file for the real loading
    this->file->Read(buf, 1);  // paranoia for fixing IsEOF under Linux
    this->file->SeekToBegin();

    delete[] buf;

    this->frameIdx = new vislib::sys::File::FileSize[frameCnt];
    frameCnt = 0;
    vislib::SingleLinkedList<vislib::sys::File::FileSize>::Iterator iter = framePoss.GetIterator();
    while (iter.HasNext()) {
        this->frameIdx[frameCnt++] = iter.Next();
    }
    if (frameCnt > 0) {
        this->setFrameCount(frameCnt);
    }

    this->file->SeekToBegin();
}


/*
 * VIMDataSource::calcBoundingBox
 */
void VIMDataSource::calcBoundingBox(void) {
    float scale;
    this->boxScaling = 0.0f;

    for (unsigned int i = 0; i < this->FrameCount(); i++) {
        this->file->Seek(this->frameIdx[i]);
        vislib::StringA line = vislib::sys::ReadLineFromFileA(*this->file).Substring(1);
        line.TrimSpacesBegin();
        try {
            scale = float(vislib::CharTraitsA::ParseDouble(line.PeekBuffer()));
            if (scale > this->boxScaling) {
                this->boxScaling = scale;
            }
        } catch (...) {}
    }

    if (vislib::math::IsEqual(this->boxScaling, 0.0f)) {
        this->boxScaling = 1.0f;
    }
}


/*
 * VIMDataSource::filenameChanged
 */
bool VIMDataSource::filenameChanged(core::param::ParamSlot& slot) {
    this->resetFrameCache();
    this->datahash++;

    if (this->file == NULL) {
        this->file = new vislib::sys::FastFile();
    } else {
        this->file->Close();
    }
    ASSERT(this->filename.Param<core::param::FilePathParam>() != NULL);

    if (!this->file->Open(this->filename.Param<core::param::FilePathParam>()->Value().native().c_str(),
            vislib::sys::File::READ_ONLY, vislib::sys::File::SHARE_READ, vislib::sys::File::OPEN_ONLY)) {
        vislib::sys::SystemMessage err(::GetLastError());
        megamol::core::utility::log::Log::DefaultLog.WriteMsg(megamol::core::utility::log::Log::LEVEL_ERROR,
            "Unable to open VIM-File \"%s\": %s",
            this->filename.Param<core::param::FilePathParam>()->Value().generic_u8string().c_str(),
            static_cast<const char*>(err));

        SAFE_DELETE(this->file);
        this->setFrameCount(1);
        this->initFrameCache(1);

        return true;
    }

    this->buildFrameTable();
    if (!this->readHeader(this->filename.Param<core::param::FilePathParam>()->Value().generic_u8string().c_str())) {
        megamol::core::utility::log::Log::DefaultLog.WriteMsg(megamol::core::utility::log::Log::LEVEL_ERROR,
            "Unable to read VIM-Header from file \"%s\". Wrong format?",
            this->filename.Param<core::param::FilePathParam>()->Value().generic_u8string().c_str());

        this->file->Close();
        SAFE_DELETE(this->file);
        this->setFrameCount(1);
        this->initFrameCache(1);

        return true;
    }
    this->calcBoundingBox();

    Frame tmpFrame(*this);
    tmpFrame.SetTypeCount(this->typeCnt);

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
 * VIMDataSource::parseTypeLine
 */
VIMDataSource::SimpleType* VIMDataSource::parseTypeLine(vislib::StringA& line, int& outType) {
    enum ElementType { TYPE_UNKNOWN, TYPE_SPHERE, TYPE_CYLINDER };
    const UINT32 ittColourTable[] = {
        0xFFB2B2B2, //  0 weisz
        0xFF191919, //  1 schwarz
        0xFF0000FF, //  2 rot
        0xFF0080FF, //  3 orange
        0xFF00FFFF, //  4 gelb
        0xFF00FFB2, //  5 zitron
        0xFF00FF00, //  6 gruen
        0xFF80FF00, //  7 tuerkis
        0xFFFFFF00, //  8 cyan
        0xFFFF8000, //  9 wasser
        0xFFFF0000, // 10 blau
        0xFFFF0080, // 11 lila
        0xFFFF00FF, // 12 magenta
        0xFF8000FF, // 13 kirsch
        0xFFC0C0C0, // 14 gray
        0xFF404040, 0xFF404040, 0xFF404040, 0xFF404040, 0xFF404040, 0xFF404040, 0xFF404040, 0xFF404040, 0xFF404040,
        0xFF404040, 0xFF404040, 0xFF404040, 0xFF404040, 0xFF404040, 0xFF404040, 0xFF404040, 0xFF404040, 0xFF404040,
        0xFF404040, 0xFF404040, 0xFF404040, 0xFF404040, 0xFF404040, 0xFF404040, 0xFF404040, 0xFF404040, 0xFF404040,
        0xFFFFC080, // 42 MegaMol-blue
        0xFF404040  // default gray
    };

    vislib::Array<vislib::StringA> shreds = vislib::StringTokeniserA::Split(line, ' ', true);
    ElementType type = TYPE_UNKNOWN;
    SimpleType* retval = NULL;

    if (shreds.Count() < 3) {
        throw 0; // far too few elements
    }
    if (!shreds[0].Equals("~")) {
        throw 0; // wrong line marker
    }

    outType = vislib::CharTraitsA::ParseInt(shreds[1]);

    if (shreds[2].Equals("LJ", false)) {
        type = TYPE_SPHERE;
    } else if (shreds[2].Equals("Sphere", false)) {
        type = TYPE_SPHERE;
    } else if (shreds[2].Equals("Cylinder", false)) {
        type = TYPE_CYLINDER;
    }

    switch (type) {
    case TYPE_UNKNOWN:
        throw 0; // unknown element type.
    case TYPE_SPHERE: {
        if ((shreds.Count() != 8) && (shreds.Count() != 10)) {
            throw 0; // wrong substring count
        }
        SimpleType* sphere = new SimpleType();
        sphere->SetID(outType);
        //sphere->SetPosition(
        //    float(vislib::CharTraitsA::ParseDouble(shreds[3])),
        //    float(vislib::CharTraitsA::ParseDouble(shreds[4])),
        //    float(vislib::CharTraitsA::ParseDouble(shreds[5])));
        sphere->SetRadius(0.5f * float(vislib::CharTraitsA::ParseDouble(shreds[6])));
        if (shreds.Count() == 8) {
            int idx = vislib::CharTraitsA::ParseInt(shreds[7]);
            if (idx < 0)
                idx = 0;
            int colours = sizeof(ittColourTable) / sizeof(UINT32);
            if (idx >= colours)
                idx = colours - 1;
            sphere->SetColour(ittColourTable[idx]);
        } else {
            sphere->SetColour(static_cast<unsigned char>(vislib::CharTraitsA::ParseInt(shreds[7])),
                static_cast<unsigned char>(vislib::CharTraitsA::ParseInt(shreds[8])),
                static_cast<unsigned char>(vislib::CharTraitsA::ParseInt(shreds[9])));
        }
        retval = sphere;
    } break;
    case TYPE_CYLINDER: {
        if ((shreds.Count() != 11) && (shreds.Count() != 13)) {
            throw 0; // wrong substring count
        }
        SimpleType* cyl = new SimpleType();
        cyl->SetID(outType);
        //cyl->SetPosition(
        //    float(vislib::CharTraitsA::ParseDouble(shreds[3])),
        //    float(vislib::CharTraitsA::ParseDouble(shreds[4])),
        //    float(vislib::CharTraitsA::ParseDouble(shreds[5])));
        //cyl->SetSecondPosition(
        //    float(vislib::CharTraitsA::ParseDouble(shreds[6])),
        //    float(vislib::CharTraitsA::ParseDouble(shreds[7])),
        //    float(vislib::CharTraitsA::ParseDouble(shreds[8])));
        cyl->SetRadius(0.5f * float(vislib::CharTraitsA::ParseDouble(shreds[9])));
        if (shreds.Count() == 11) {
            int idx = vislib::CharTraitsA::ParseInt(shreds[10]);
            if (idx < 0)
                idx = 0;
            int colours = sizeof(ittColourTable) / sizeof(UINT32);
            if (idx >= colours)
                idx = colours - 1;
            cyl->SetColour(ittColourTable[idx]);
        } else {
            cyl->SetColour(static_cast<unsigned char>(vislib::CharTraitsA::ParseInt(shreds[10])),
                static_cast<unsigned char>(vislib::CharTraitsA::ParseInt(shreds[11])),
                static_cast<unsigned char>(vislib::CharTraitsA::ParseInt(shreds[12])));
        }
        retval = cyl;
    } break;
    }

    return retval;
}


/*
 * VIMDataSource::readHeader
 */
bool VIMDataSource::readHeader(const vislib::TString& filename) {

    ASSERT(this->file->Tell() == 0);

    vislib::Array<SimpleType> types;

    // read the header
    while (!this->file->IsEOF()) {
        vislib::StringA line = vislib::sys::ReadLineFromFileA(*this->file);
        line.TrimSpaces();

        if (line.IsEmpty()) {
            continue;

        } else if (line[0] == '~') {
            SimpleType* element = NULL;
            try { // a type line!
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
            } catch (...) { megamol::core::utility::log::Log::DefaultLog.WriteMsg(50, "Error parsing type line."); }
            SAFE_DELETE(element);
        } else if (line[0] == '>') {
            // very extream file redirection
            vislib::StringW vnfn(vislib::StringA(line.PeekBuffer() + 1));
            vnfn.TrimSpaces();
            vislib::StringW base = vislib::sys::Path::Resolve(vislib::StringW(filename));
            base.Truncate(base.FindLast(vislib::sys::Path::SEPARATOR_W) + 1);
            vnfn = vislib::sys::Path::Resolve(vnfn, base);

            this->file->Close();
            if (!this->file->Open(
                    vnfn, vislib::sys::File::READ_ONLY, vislib::sys::File::SHARE_READ, vislib::sys::File::OPEN_ONLY)) {

                SAFE_DELETE(this->file);
                return false;
            }

            this->buildFrameTable();
            break;

        } else if (line[0] == '#') {
            // beginning of the body!
            break;
        }
    }

    this->typeCnt = static_cast<unsigned int>(types.Count());
    this->types = new SimpleType[this->typeCnt];
    for (unsigned int i = 0; i < this->typeCnt; i++) {
        this->types[i] = types[i];
    }

    return true;
}


/*
 * VIMDataSource::getDataCallback
 */
bool VIMDataSource::getDataCallback(core::Call& caller) {
    geocalls::MultiParticleDataCall* c2 = dynamic_cast<geocalls::MultiParticleDataCall*>(&caller);
    geocalls::EllipsoidalParticleDataCall* c3 = dynamic_cast<geocalls::EllipsoidalParticleDataCall*>(&caller);
    if ((c2 == nullptr) && (c3 == nullptr))
        return false;

    unsigned int fid = (c2 != nullptr) ? c2->FrameID() : c3->FrameID();
    Frame* f = dynamic_cast<Frame*>(this->requestLockedFrame(fid));
    if (f == NULL)
        return false;

    if (c2 != nullptr) {
        c2->SetDataHash((this->file == NULL) ? 0 : this->datahash);
        c2->SetUnlocker(new Unlocker(*f));
        c2->SetParticleListCount(this->typeCnt);
        for (unsigned int i = 0; i < this->typeCnt; i++) {
            c2->AccessParticles(i).SetGlobalRadius(this->types[i].Radius() /* / this->boxScaling*/);
            c2->AccessParticles(i).SetGlobalColour(this->types[i].Red(), this->types[i].Green(), this->types[i].Blue());
            c2->AccessParticles(i).SetCount(f->PartCnt(i));
            c2->AccessParticles(i).SetColourData(geocalls::MultiParticleDataCall::Particles::COLDATA_NONE, NULL);
            const float* vd = f->PartPoss(i);
            c2->AccessParticles(i).SetVertexData((vd == NULL)
                                                     ? geocalls::MultiParticleDataCall::Particles::VERTDATA_NONE
                                                     : geocalls::MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZ,
                vd);
        }
    } else {
        assert(c3 != nullptr);
        c3->SetDataHash((this->file == NULL) ? 0 : this->datahash);
        c3->SetUnlocker(new Unlocker(*f));
        c3->SetParticleListCount(this->typeCnt);
        for (unsigned int i = 0; i < this->typeCnt; i++) {

            //            c3->AccessParticles(i).SetGlobalRadius(this->types[i].Radius()/* / this->boxScaling*/);

            c3->AccessParticles(i).SetCount(f->PartCnt(i));

            c3->AccessParticles(i).SetGlobalColour(this->types[i].Red(), this->types[i].Green(), this->types[i].Blue());
            c3->AccessParticles(i).SetColourData(geocalls::MultiParticleDataCall::Particles::COLDATA_NONE, NULL);

            const float* vd = f->PartPoss(i);
            const float* qd = f->PartQuats(i);
            const float* rd = f->PartRadii(i, this->types[i]);
            c3->AccessParticles(i).SetVertexData((vd == NULL)
                                                     ? geocalls::MultiParticleDataCall::Particles::VERTDATA_NONE
                                                     : geocalls::MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZ,
                vd);
            if (qd)
                c3->AccessParticles(i).SetQuatData(qd);
            if (rd)
                c3->AccessParticles(i).SetRadData(rd);
        }
    }

    return true;
}


/*
 * VIMDataSource::getExtentCallback
 */
bool VIMDataSource::getExtentCallback(core::Call& caller) {
    core::AbstractGetData3DCall* c2 = dynamic_cast<core::AbstractGetData3DCall*>(&caller);
    float border = 0.0f;

    if (c2 != NULL) {
        for (unsigned int i = 0; i < this->typeCnt; i++) {
            float r = this->types[i].Radius(); // / this->boxScaling;
            if (r > border)
                border = r;
        }

        c2->SetDataHash((this->file == NULL) ? 0 : this->datahash);
        c2->SetFrameCount(this->FrameCount());
        c2->AccessBoundingBoxes().Clear();
        c2->AccessBoundingBoxes().SetObjectSpaceBBox(0, 0, 0, this->boxScaling, this->boxScaling, this->boxScaling);
        c2->AccessBoundingBoxes().SetObjectSpaceClipBox(
            -border, -border, -border, this->boxScaling + border, this->boxScaling + border, this->boxScaling + border);
        return true;
    }

    return false;
}
