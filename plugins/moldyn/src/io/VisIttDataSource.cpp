/*
 * VisIttDataSource.cpp
 *
 * Copyright (C) 2009-2011 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#include "io/VisIttDataSource.h"
#include "mmcore/CoreInstance.h"
#include "mmcore/param/BoolParam.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/param/FilePathParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/StringParam.h"
#include "mmcore/utility/log/Log.h"
#include "mmcore/utility/sys/SystemInformation.h"
#include "vislib/String.h"
#include "vislib/graphics/ColourRGBAu8.h"
#include "vislib/sys/FastFile.h"
#include "vislib/sys/sysfunctions.h"
#include <algorithm>
#include <climits>

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
 * VisIttDataSource::Frame::Frame
 */
VisIttDataSource::Frame::Frame(core::view::AnimDataModule& owner)
        : core::view::AnimDataModule::Frame(owner)
        , size(0)
        , dat() {
    // intentionally empty
}


/*
 * VisIttDataSource::Frame::~Frame
 */
VisIttDataSource::Frame::~Frame() {
    // intentionally empty
}

/*****************************************************************************/


/*
 * VisIttDataSource::VisIttDataSource
 */
VisIttDataSource::VisIttDataSource(void)
        : core::view::AnimDataModule()
        , filename("filename", "The path to the trisoup file to load.")
        , radius("radius", "The radius to be assumed for the particles")
        , filter("filter::type", "The filter to be applied")
        , filterColumn("filter::column", "The filter column to be applied")
        , filterValue("filter::value", "The filter value to be applied")
        , getData("getdata", "Slot to request data from this data source.")
        , file(NULL)
        , dataHash(0)
        , frameTable()
        , header()
        , headerIdx()
        , filterIndex(UINT_MAX)
        , sortPartIdSlots("sortById", "Sorts particles by their IDs")
        , idIndex(UINT_MAX)
        , splitTypesSlots("splitTypes", "Activates splitting based on types")
        , splitTypesNameSlots("splitTypesName", "Split types based on this data column")
        , typeIndex(UINT_MAX) {

    this->filename.SetParameter(new core::param::FilePathParam(""));
    this->filename.SetUpdateCallback(&VisIttDataSource::filenameChanged);
    this->MakeSlotAvailable(&this->filename);

    this->radius << new core::param::FloatParam(0.5f, 0.00001f);
    this->MakeSlotAvailable(&this->radius);

    core::param::EnumParam* filterTypes = new core::param::EnumParam(0);
    filterTypes->SetTypePair(0, "none");
    filterTypes->SetTypePair(1, "=");
    filterTypes->SetTypePair(6, "!=");
    filterTypes->SetTypePair(2, "<");
    filterTypes->SetTypePair(3, ">");
    filterTypes->SetTypePair(4, "<=");
    filterTypes->SetTypePair(5, ">=");
    this->filter.SetParameter(filterTypes);
    this->filter.SetUpdateCallback(&VisIttDataSource::filterChanged);
    this->MakeSlotAvailable(&this->filter);

    this->filterColumn << new core::param::StringParam("");
    this->filterColumn.SetUpdateCallback(&VisIttDataSource::filterChanged);
    this->MakeSlotAvailable(&this->filterColumn);

    this->filterValue << new core::param::FloatParam(0.0f);
    this->filterValue.SetUpdateCallback(&VisIttDataSource::filterChanged);
    this->MakeSlotAvailable(&this->filterValue);

    this->getData.SetCallback("MultiParticleDataCall", "GetData", &VisIttDataSource::getDataCallback);
    this->getData.SetCallback("MultiParticleDataCall", "GetExtent", &VisIttDataSource::getExtentCallback);
    this->MakeSlotAvailable(&this->getData);

    this->sortPartIdSlots.SetParameter(new core::param::BoolParam(true));
    this->sortPartIdSlots.SetUpdateCallback(&VisIttDataSource::filterChanged);
    this->MakeSlotAvailable(&this->sortPartIdSlots);

    this->splitTypesSlots.SetParameter(new core::param::BoolParam(false));
    this->splitTypesSlots.SetUpdateCallback(&VisIttDataSource::filterChanged);
    this->MakeSlotAvailable(&this->splitTypesSlots);

    this->splitTypesNameSlots.SetParameter(new core::param::StringParam("t"));
    this->splitTypesNameSlots.SetUpdateCallback(&VisIttDataSource::filterChanged);
    this->MakeSlotAvailable(&this->splitTypesNameSlots);

    this->setFrameCount(1);
    this->initFrameCache(1);
}


/*
 * VisIttDataSource::~VisIttDataSource
 */
VisIttDataSource::~VisIttDataSource(void) {
    this->Release(); // implicitly calls 'release'
}


/*
 * VisIttDataSource::constructFrame
 */
core::view::AnimDataModule::Frame* VisIttDataSource::constructFrame(void) const {
    Frame* f = new Frame(*const_cast<VisIttDataSource*>(this));
    return f;
}


/*
 * VisIttDataSource::create
 */
bool VisIttDataSource::create(void) {
    return true;
}


/*
 * VisIttDataSource::loadFrame
 */
void VisIttDataSource::loadFrame(core::view::AnimDataModule::Frame* frame, unsigned int idx) {
    Frame* f = dynamic_cast<Frame*>(frame);
    if (f == NULL)
        return;
    f->Clear();
    if (this->file == NULL) {
        return;
    }
    ASSERT(idx < this->FrameCount());
    ASSERT(this->headerIdx.Count() >= 3);

    unsigned int typeId = 0;
    std::map<unsigned int, std::vector<std::pair<unsigned int, unsigned int>>> pids;

    SIZE_T len =
        static_cast<SIZE_T>(((idx + 1 < this->frameTable.Count()) ? this->frameTable[idx + 1] : this->file->GetSize()) -
                            this->frameTable[idx]);
    this->file->Seek(this->frameTable[idx]);
    char* buf = new char[len + 2];
    len = static_cast<SIZE_T>(this->file->Read(buf, len));
    buf[len] = '\n';
    buf[len + 1] = 0;
    int filterType = this->filter.Param<core::param::EnumParam>()->Value();
    float filterVal = this->filterValue.Param<core::param::FloatParam>()->Value();
    for (SIZE_T i = 0; i <= len; i++) {
        char* line = buf + i;
        if (line[0] == '#')
            break; // end of frame
        while ((i <= len) && (buf[i] != '\n'))
            i++;
        if (i > len)
            break;
        buf[i] = '0';

        if ((filterType > 0) && (this->filterIndex < this->header.Count())) {
            unsigned int start = 0;
            for (unsigned int j = 0; j < this->filterIndex; j++) {
                start += this->header[j].Second();
            }
            unsigned int end = start + this->header[this->filterIndex].Second();
            char endChar = line[end];
            line[end] = 0;
            bool doFilter = false;
            try {
                float v = static_cast<float>(vislib::CharTraitsA::ParseDouble(line + start));
                switch (filterType) {
                case 1: // =
                    doFilter = vislib::math::IsEqual(v, filterVal);
                    break;
                case 6: // !=
                    doFilter = !vislib::math::IsEqual(v, filterVal);
                    break;
                case 2: // <
                    doFilter = (v < filterVal);
                    break;
                case 3: // >
                    doFilter = (v > filterVal);
                    break;
                case 4: // <=
                    doFilter = (v <= filterVal);
                    break;
                case 5: // >=
                    doFilter = (v >= filterVal);
                    break;
                default: // do not filter
                    break;
                }
            } catch (...) {}
            line[end] = endChar;
            if (doFilter)
                continue;
        }

        if (this->typeIndex != UINT_MAX) {

            unsigned int hidx = this->typeIndex;
            unsigned int start = 0;
            for (unsigned int k = 0; k < hidx; k++) {
                start += this->header[k].Second();
            }
            unsigned int end = start + this->header[hidx].Second();
            char endChar = line[end];
            line[end] = 0;
            try {
                typeId = vislib::CharTraitsA::ParseInt(line + start);
            } catch (...) { typeId = 0; }
            line[end] = endChar;
        }

        if (this->idIndex != UINT_MAX) {
            unsigned int pid;
            unsigned int hidx = this->idIndex;
            unsigned int start = 0;
            for (unsigned int k = 0; k < hidx; k++) {
                start += this->header[k].Second();
            }
            unsigned int end = start + this->header[hidx].Second();
            char endChar = line[end];
            line[end] = 0;
            try {
                pid = static_cast<unsigned int>(vislib::CharTraitsA::ParseInt(line + start));
            } catch (...) { pid = static_cast<unsigned int>(pids[typeId].size()); }
            line[end] = endChar;
            pids[typeId].push_back(
                std::pair<unsigned int, unsigned int>(static_cast<unsigned int>(pids[typeId].size()), pid));
        }

        std::vector<float>& vec = f->AccessParticleData(typeId);
        size_t ovs = vec.size();
        vec.resize(ovs + 3);
        float* pos = vec.data() + ovs;

        for (unsigned int j = 0; j < 3; j++) {
            unsigned int hidx = this->headerIdx[j];
            unsigned int start = 0;
            for (unsigned int k = 0; k < hidx; k++) {
                start += this->header[k].Second();
            }
            unsigned int end = start + this->header[hidx].Second();
            char endChar = line[end];
            line[end] = 0;
            try {
                pos[j] = static_cast<float>(vislib::CharTraitsA::ParseDouble(line + start));
            } catch (...) { pos[j] = 0.0f; }
            line[end] = endChar;
        }
    }
    delete[] buf;
    f->SetFrameNumber(idx);

    if (this->idIndex != UINT_MAX) {
        // now sort particles
        std::vector<unsigned int> keys = f->ParticleTypes();
        for (unsigned int key : keys) {
            std::vector<float>& poss = f->AccessParticleData(key);
            std::vector<std::pair<unsigned int, unsigned int>>& ids = pids[key];
            assert(ids.size() * 3 == poss.size());
            std::sort(ids.begin(), ids.end(),
                [](const std::pair<unsigned int, unsigned int>& a,
                    const std::pair<unsigned int, unsigned int>& b) -> bool { return a.second < b.second; });
            std::vector<float> pos_copy(poss);
            unsigned int cnt = static_cast<unsigned int>(ids.size());
            for (unsigned int i = 0; i < cnt; ++i) {
                for (unsigned int c = 0; c < 3; ++c) {
                    poss[i * 3 + c] = pos_copy[ids[i].first * 3 + c];
                }
            }
        }
    }

    megamol::core::utility::log::Log::DefaultLog.WriteInfo(100, "Frame %u loaded", idx);
}


/*
 * VisIttDataSource::release
 */
void VisIttDataSource::release(void) {
    this->resetFrameCache();
    if (this->file != NULL) {
        vislib::sys::File* f = this->file;
        this->file = NULL;
        f->Close();
        delete f;
    }
    this->frameTable.Clear();
    this->header.Clear();
    this->headerIdx.Clear();
}


/*
 * VisIttDataSource::buildFrameTable
 */
void VisIttDataSource::buildFrameTable(void) {
    ASSERT(this->file != NULL);

    const unsigned int bufSize = 1024 * 1024;
    char* buf = new char[bufSize];
    unsigned int size = 1;
    char lCh1 = 0, lCh2 = 0;
    vislib::sys::File::FileSize pos = 0;
    unsigned int i;

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
                this->frameTable.Add(pos - 1);
            }
        }

        for (i = 0; i < size - 1; i++) {
            if (buf[i] == '#') {
                if (buf[i + 1] == '#') {
                    break; // end of data
                }
                if (((i == 0) && ((lCh1 == 0x0D) || (lCh1 == 0x0A))) ||
                    ((i > 0) && ((buf[i - 1] == 0x0D) || (buf[i - 1] == 0x0A)))) {
                    this->frameTable.Add(pos + i);
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

    for (SIZE_T i = 1; i < this->frameTable.Count(); i++) {
        this->frameTable[i] += this->frameTable[0] + 2; // header offset
    }
}


/*
 * VisIttDataSource::filenameChanged
 */
bool VisIttDataSource::filenameChanged(core::param::ParamSlot& slot) {
    this->resetFrameCache();
    this->dataHash++;

    if (this->file == NULL) {
        this->file = new vislib::sys::FastFile();
    } else {
        this->file->Close();
    }
    ASSERT(this->filename.Param<core::param::FilePathParam>() != NULL);

    if (!this->file->Open(this->filename.Param<core::param::FilePathParam>()->Value().native().c_str(),
            vislib::sys::File::READ_ONLY, vislib::sys::File::SHARE_READ, vislib::sys::File::OPEN_ONLY)) {
        megamol::core::utility::log::Log::DefaultLog.WriteError("Unable to open VisItt-File \"%s\".",
            this->filename.Param<core::param::FilePathParam>()->Value().generic_u8string().c_str());

        SAFE_DELETE(this->file);
        this->setFrameCount(1);
        this->initFrameCache(1);

        return true;
    }

    this->frameTable.Clear();
    this->header.Clear();
    this->headerIdx.Clear();

    // read header line
    this->file->SeekToBegin();
    vislib::StringA header = vislib::sys::ReadLineFromFileA(*this->file);
    if (!this->parseHeader(header)) {
        megamol::core::utility::log::Log::DefaultLog.WriteError("Unable to parse VisItt-file header line");

        this->file->Close();
        SAFE_DELETE(this->file);
        this->setFrameCount(1);
        this->initFrameCache(1);

        return true;
    }

    if (this->sortPartIdSlots.Param<core::param::BoolParam>()->Value()) {
        this->findIdColumn();
    }
    if (this->splitTypesSlots.Param<core::param::BoolParam>()->Value()) {
        this->findTypeColumn();
    }
    this->findFilterColumn();

    this->frameTable.Add(this->file->Tell());
    this->buildFrameTable();
    this->setFrameCount(static_cast<unsigned int>(this->frameTable.Count()));
    Frame tmpFrame(*this);
    this->loadFrame(&tmpFrame, 0);

    // calculating the bounding box from frame 0
    this->bbox.Set(-1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0);
    if (tmpFrame.GetFrameSize() > sizeof(float) * 3) {
        std::vector<unsigned int> keys = tmpFrame.ParticleTypes();

        const float* pos = tmpFrame.ParticleData(keys[0]);
        this->bbox.Set(pos[0], pos[1], pos[2], pos[0], pos[1], pos[2]);

        for (unsigned int key : keys) {
            const float* pos = tmpFrame.ParticleData(key);
            unsigned int cnt = tmpFrame.ParticleCount(key);
            for (unsigned int i = 0; i < cnt; i++, pos += 3) {
                this->bbox.GrowToPoint(pos[0], pos[1], pos[2]);
            }
        }
    }

    //tmpFrame.SetTypeCount(this->typeCnt);
    // use frame zero to estimate the frame size in memory to calculate the
    // frame cache size
    SIZE_T frameSize = tmpFrame.GetFrameSize();
    tmpFrame.Clear();
    frameSize = static_cast<SIZE_T>(float(frameSize) * CACHE_FRAME_FACTOR);
    UINT64 mem = vislib::sys::SystemInformation::AvailableMemorySize();
    unsigned int cacheSize = static_cast<unsigned int>(mem / frameSize);
    if (cacheSize > static_cast<unsigned int>(this->frameTable.Count())) {
        cacheSize = static_cast<unsigned int>(this->frameTable.Count());
    }

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
    if (this->frameTable.Count() > 0) {
        // refine bounding box using more frames
        this->loadFrame(&tmpFrame, static_cast<unsigned int>(this->frameTable.Count() - 1));
        std::vector<unsigned int> keys = tmpFrame.ParticleTypes();
        for (unsigned int key : keys) {
            const float* pos = tmpFrame.ParticleData(key);
            unsigned int cnt = tmpFrame.ParticleCount(key);
            for (unsigned int i = 0; i < cnt; i++, pos += 3) {
                this->bbox.GrowToPoint(pos[0], pos[1], pos[2]);
            }
        }

        this->loadFrame(&tmpFrame, static_cast<unsigned int>(this->frameTable.Count() / 2));
        keys = tmpFrame.ParticleTypes();
        for (unsigned int key : keys) {
            const float* pos = tmpFrame.ParticleData(key);
            unsigned int cnt = tmpFrame.ParticleCount(key);
            for (unsigned int i = 0; i < cnt; i++, pos += 3) {
                this->bbox.GrowToPoint(pos[0], pos[1], pos[2]);
            }
        }
    }
    this->initFrameCache(cacheSize);

    return true; // to reset the dirty flag of the param slot
}


/*
 * VisIttDataSource::filterChanged
 */
bool VisIttDataSource::filterChanged(core::param::ParamSlot& slot) {
    // Power Dovel: On!
    unsigned int cs = this->CacheSize();
    unsigned int fc = this->FrameCount();
    this->resetFrameCache();
    this->setFrameCount(fc);
    this->findFilterColumn();
    this->initFrameCache(cs);
    return true;
}


/*
 * VisIttDataSource::parseHeader
 */
bool VisIttDataSource::parseHeader(const vislib::StringA& header) {
    ASSERT(this->header.Count() == 0);
    ASSERT(this->headerIdx.Count() == 0);
    unsigned int len = header.Length();
    if (len == 0)
        return false;
    for (unsigned int p = 0; p < len;) {
        unsigned int start = p;
        while ((p < len) && vislib::CharTraitsA::IsSpace(header[p]))
            p++;
        while ((p < len) && !vislib::CharTraitsA::IsSpace(header[p]))
            p++;
        if ((p - start) > 0) {
            vislib::StringA label = header.Substring(start, p - start);
            label.TrimSpaces();
            if (label.Length() > 0) {
                this->header.Add(vislib::Pair<vislib::StringA, unsigned int>(label, (p - start)));
            }
        }
    }
    if (this->header.Count() == 0)
        return false;

    for (SIZE_T i = 0; i < this->header.Count(); i++) {
        if (this->header[i].First().Equals("x", false)) {
            this->headerIdx.Add(static_cast<unsigned int>(i));
        }
    }
    for (SIZE_T i = 0; i < this->header.Count(); i++) {
        if (this->header[i].First().Equals("y", false)) {
            this->headerIdx.Add(static_cast<unsigned int>(i));
        }
    }
    for (SIZE_T i = 0; i < this->header.Count(); i++) {
        if (this->header[i].First().Equals("z", false)) {
            this->headerIdx.Add(static_cast<unsigned int>(i));
        }
    }
    if (this->headerIdx.Count() == 0)
        return false;

    while (this->headerIdx.Count() < 3) { // stupid, but makes things easier for now
        this->headerIdx.Add(this->headerIdx[0]);
    }

    return true;
}


/*
 * VisIttDataSource::getDataCallback
 */
bool VisIttDataSource::getDataCallback(core::Call& caller) {
    geocalls::MultiParticleDataCall* c2 = dynamic_cast<geocalls::MultiParticleDataCall*>(&caller);

    Frame* f = NULL;
    if (c2 != NULL) {
        f = dynamic_cast<Frame*>(this->requestLockedFrame(c2->FrameID()));
        if (f == NULL)
            return false;

        c2->SetDataHash((this->file == NULL) ? 0 : this->dataHash);
        c2->SetUnlocker(new Unlocker(*f));
        c2->SetFrameID(f->FrameNumber());

        std::vector<unsigned int> keys = f->ParticleTypes();
        std::vector<vislib::graphics::ColourRGBAu8> cols;
        cols.resize(keys.size(), vislib::graphics::ColourRGBAu8(192, 192, 192, 255));

        if (cols.size() > 1) {
            // some colors
            cols[0] = vislib::graphics::ColourRGBAu8(255, 0, 0, 255);
            cols[1] = vislib::graphics::ColourRGBAu8(0, 255, 0, 255);
            if (cols.size() > 2)
                cols[2] = vislib::graphics::ColourRGBAu8(0, 0, 255, 255);
            // not bright, but ok for now
        }

        c2->SetParticleListCount(static_cast<unsigned int>(keys.size()));
        unsigned int ti = 0;
        for (unsigned int key : keys) {
            c2->AccessParticles(ti).SetGlobalRadius(this->radius.Param<core::param::FloatParam>()->Value());
            c2->AccessParticles(ti).SetGlobalColour(cols[ti].R(), cols[ti].G(), cols[ti].B());
            c2->AccessParticles(ti).SetCount(f->ParticleCount(key));
            c2->AccessParticles(ti).SetVertexData(
                geocalls::MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZ, f->ParticleData(key));
            ++ti;
        }

        return true;
    }

    return false;
}


/*
 * VisIttDataSource::getExtentCallback
 */
bool VisIttDataSource::getExtentCallback(core::Call& caller) {
    geocalls::MultiParticleDataCall* c2 = dynamic_cast<geocalls::MultiParticleDataCall*>(&caller);

    if (c2 != NULL) {
        float border = this->radius.Param<core::param::FloatParam>()->Value();

        c2->SetDataHash((this->file == NULL) ? 0 : this->dataHash);
        c2->SetFrameCount(static_cast<unsigned int>(this->frameTable.Count()));
        c2->AccessBoundingBoxes().Clear();
        c2->AccessBoundingBoxes().SetObjectSpaceBBox(this->bbox);
        c2->AccessBoundingBoxes().SetObjectSpaceClipBox(this->bbox.Left() - border, this->bbox.Bottom() - border,
            this->bbox.Back() - border, this->bbox.Right() + border, this->bbox.Top() + border,
            this->bbox.Front() + border);

        return true;
    }

    return false;
}


/*
 * VisIttDataSource::findFilterColumn
 */
void VisIttDataSource::findFilterColumn(void) {

    this->filterIndex = UINT_MAX;
    vislib::StringA filtCol(this->filterColumn.Param<core::param::StringParam>()->Value().c_str());
    filtCol.TrimSpaces();
    for (SIZE_T i = 0; i < this->header.Count(); i++) {
        vislib::Pair<vislib::StringA, unsigned int>& hi = this->header[i];
        if (hi.First().Equals(filtCol)) {
            this->filterIndex = static_cast<unsigned int>(i);
            break;
        }
    }
    if (this->filterIndex == UINT_MAX) {
        for (SIZE_T i = 0; i < this->header.Count(); i++) {
            vislib::Pair<vislib::StringA, unsigned int>& hi = this->header[i];
            if (hi.First().Equals(filtCol, false)) {
                this->filterIndex = static_cast<unsigned int>(i);
                break;
            }
        }
    }
    if (this->filterIndex == UINT_MAX) {
        try {
            int idx = vislib::CharTraitsA::ParseInt(filtCol);
            if ((idx >= 0) && (idx < static_cast<int>(this->header.Count()))) {
                this->filterIndex = idx;
            }
        } catch (...) {}
    }
}


/*
 * VisIttDataSource::findIdColumn
 */
void VisIttDataSource::findIdColumn(void) {
    this->idIndex = UINT_MAX;
    for (SIZE_T i = 0; i < this->header.Count(); i++) {
        vislib::Pair<vislib::StringA, unsigned int>& hi = this->header[i];
        if (hi.First().Equals("id", false)) {
            this->idIndex = static_cast<unsigned int>(i);
            break;
        }
    }
}


/*
 * VisIttDataSource::findTypeColumn
 */
void VisIttDataSource::findTypeColumn(void) {
    this->typeIndex = UINT_MAX;
    vislib::StringA typeCol(this->splitTypesNameSlots.Param<core::param::StringParam>()->Value().c_str());
    for (SIZE_T i = 0; i < this->header.Count(); i++) {
        vislib::Pair<vislib::StringA, unsigned int>& hi = this->header[i];
        if (hi.First().Equals(typeCol)) {
            this->typeIndex = static_cast<unsigned int>(i);
            break;
        }
    }
    if (this->typeIndex == UINT_MAX) {
        for (SIZE_T i = 0; i < this->header.Count(); i++) {
            vislib::Pair<vislib::StringA, unsigned int>& hi = this->header[i];
            if (hi.First().Equals(typeCol, false)) {
                this->typeIndex = static_cast<unsigned int>(i);
                break;
            }
        }
    }
}
