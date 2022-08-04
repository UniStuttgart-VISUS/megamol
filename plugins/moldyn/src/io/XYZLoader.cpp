/*
 * XYZLoader.cpp
 *
 * Copyright (C) 2015 by MegaMol Team (TU Dresden)
 * Alle Rechte vorbehalten.
 */
#include "XYZLoader.h"
#include "geometry_calls/MultiParticleDataCall.h"
#include "mmcore/CoreInstance.h"
#include "mmcore/param/BoolParam.h"
#include "mmcore/param/FilePathParam.h"
#include "mmcore/param/FloatParam.h"
#include "vislib/StringTokeniser.h"
#include "vislib/sys/FastFile.h"
#include "vislib/sys/TextFileReader.h"
#include <deque>
#include <map>

using namespace megamol;
using namespace megamol::moldyn;


float io::XYZLoader::FileFormatAutoDetect(const unsigned char* data, SIZE_T dataSize) {
    // idea: try to find a newline char in data and be happy if you only encounter numbers before
    for (SIZE_T i = 0; i < dataSize; ++i) {
        char c = static_cast<char>(data[i]);
        if (c == '\n')
            return 0.3f; // could be
        if (c == ' ')
            continue;
        if (c == '\t')
            continue;
        if (c == '\r')
            continue;
        if (c >= '0' && c <= '9')
            continue;
        break;
    }
    return 0.0f;
}

io::XYZLoader::XYZLoader()
        : core::Module()
        , getDataSlot("getdata", "Access to the data")
        , filenameSlot("filename", "Path to the XYZ file to load")
        , hasCountLineSlot("hasCountLine", "If true, tries to load the number of atoms from the first line")
        , hasCommentLineSlot("hasCommentLine", "If true, skips the second line")
        , hasElementSymbolSlot("hasElementSymbol", "If true, expects an element symbol at each atom line")
        , groupByElementSlot("groupByElements", "If true, groups atoms of each element in one list (may destroy order)")
        , radiusSlot("radius", "The radius to be assumed for all atoms")
        , hash(0)
        , bbox(-1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f)
        , poss() {

    getDataSlot.SetCallback(geocalls::MultiParticleDataCall::ClassName(), "GetData", &XYZLoader::getDataCallback);
    getDataSlot.SetCallback(geocalls::MultiParticleDataCall::ClassName(), "GetExtent", &XYZLoader::getExtentCallback);
    MakeSlotAvailable(&getDataSlot);

    filenameSlot.SetParameter(new core::param::FilePathParam(""));
    MakeSlotAvailable(&filenameSlot);

    hasCountLineSlot.SetParameter(new core::param::BoolParam(true));
    MakeSlotAvailable(&hasCountLineSlot);

    hasCommentLineSlot.SetParameter(new core::param::BoolParam(true));
    MakeSlotAvailable(&hasCommentLineSlot);

    hasElementSymbolSlot.SetParameter(new core::param::BoolParam(true));
    MakeSlotAvailable(&hasElementSymbolSlot);

    groupByElementSlot.SetParameter(new core::param::BoolParam(true));
    MakeSlotAvailable(&groupByElementSlot);

    radiusSlot.SetParameter(new core::param::FloatParam(0.5));
    MakeSlotAvailable(&radiusSlot);
}

io::XYZLoader::~XYZLoader() {
    Release();
}

bool io::XYZLoader::create(void) {
    return true;
}

void io::XYZLoader::release(void) {
    clear();
    hash = 0;
}

bool io::XYZLoader::getDataCallback(core::Call& caller) {
    geocalls::MultiParticleDataCall* mpdc = dynamic_cast<geocalls::MultiParticleDataCall*>(&caller);
    if (mpdc == nullptr)
        return false;
    assertData();

    mpdc->SetDataHash(hash);
    mpdc->SetFrameCount(1);
    mpdc->AccessBoundingBoxes().Clear();
    mpdc->AccessBoundingBoxes().SetObjectSpaceBBox(bbox);
    vislib::math::Cuboid<float> cbox = bbox;
    cbox.Grow(radiusSlot.Param<core::param::FloatParam>()->Value());
    mpdc->AccessBoundingBoxes().SetObjectSpaceClipBox(cbox);
    mpdc->SetFrameID(0);
    mpdc->SetParticleListCount(static_cast<unsigned int>(poss.size()));
    for (size_t i = 0; i < poss.size(); ++i) {
        auto& p = mpdc->AccessParticles(static_cast<unsigned int>(i));
        assert((poss[i].size() % 3) == 0);
        p.SetCount(poss[i].size() / 3);
        p.SetGlobalColour(192, 192, 192);
        p.SetGlobalRadius(radiusSlot.Param<core::param::FloatParam>()->Value());
        p.SetColourData(geocalls::SimpleSphericalParticles::COLDATA_NONE, nullptr);
        if (poss[i].size() > 0) {
            p.SetVertexData(geocalls::SimpleSphericalParticles::VERTDATA_FLOAT_XYZ, poss[i].data());
        } else {
            p.SetVertexData(geocalls::SimpleSphericalParticles::VERTDATA_NONE, nullptr);
        }
    }
    mpdc->SetUnlocker(nullptr);

    return true;
}

bool io::XYZLoader::getExtentCallback(core::Call& caller) {
    geocalls::MultiParticleDataCall* mpdc = dynamic_cast<geocalls::MultiParticleDataCall*>(&caller);
    if (mpdc == nullptr)
        return false;
    assertData();

    mpdc->SetDataHash(hash);
    mpdc->SetFrameCount(1);
    mpdc->AccessBoundingBoxes().Clear();
    mpdc->AccessBoundingBoxes().SetObjectSpaceBBox(bbox);
    vislib::math::Cuboid<float> cbox = bbox;
    cbox.Grow(radiusSlot.Param<core::param::FloatParam>()->Value());
    mpdc->AccessBoundingBoxes().SetObjectSpaceClipBox(cbox);
    mpdc->SetUnlocker(nullptr);

    return true;
}

void io::XYZLoader::clear(void) {
    hash++;
    bbox.SetNull();
    poss.clear();
}

void io::XYZLoader::assertData(void) {
    if (!filenameSlot.IsDirty() && !hasCountLineSlot.IsDirty() && !hasCommentLineSlot.IsDirty() &&
        !hasElementSymbolSlot.IsDirty() && !groupByElementSlot.IsDirty())
        return;
    filenameSlot.ResetDirty();
    hasCountLineSlot.ResetDirty();
    hasCommentLineSlot.ResetDirty();
    hasElementSymbolSlot.ResetDirty();
    groupByElementSlot.ResetDirty();

    float rad = radiusSlot.Param<core::param::FloatParam>()->Value();

    vislib::sys::FastFile file;
    if (!file.Open(filenameSlot.Param<core::param::FilePathParam>()->Value().native().c_str(),
            vislib::sys::File::READ_ONLY, vislib::sys::File::SHARE_READ, vislib::sys::File::OPEN_ONLY)) {
        megamol::core::utility::log::Log::DefaultLog.WriteError("Unable to open file \"%s\"",
            filenameSlot.Param<core::param::FilePathParam>()->Value().generic_u8string().c_str());
        return;
    }
    vislib::sys::TextFileReader reader(&file);

    size_t partCnt = -1;
    vislib::StringA l;
    unsigned int lineNum = 0;

    if (hasCountLineSlot.Param<core::param::BoolParam>()->Value()) {
        lineNum++;
        reader.ReadLine(l);
        try {
            partCnt = static_cast<size_t>(vislib::CharTraitsA::ParseUInt64(l.PeekBuffer()));
        } catch (...) {
            megamol::core::utility::log::Log::DefaultLog.WriteWarn(
                "Unable to parse atom count from first line in \"%s\"",
                filenameSlot.Param<core::param::FilePathParam>()->Value().generic_u8string().c_str());
        }
    }

    if (hasCommentLineSlot.Param<core::param::BoolParam>()->Value()) {
        lineNum++;
        reader.ReadLine(l); // just skip the second line
    }

    bool hasEl = hasElementSymbolSlot.Param<core::param::BoolParam>()->Value();
    bool grpEl = groupByElementSlot.Param<core::param::BoolParam>()->Value();
    bool warning = true;
    if (!grpEl)
        poss.resize(1);
    std::map<std::string, std::deque<float>> grpDat;

    while (reader.ReadLine(l)) {
        lineNum++;
        l.Replace('\t', ' ');
        vislib::Array<vislib::StringA> parts(vislib::StringTokeniserA::Split(l, ' ', true));
        if (parts.Count() != (hasEl ? 4 : 3)) {
            if (warning) {
                megamol::core::utility::log::Log::DefaultLog.WriteWarn("Problem parsing \"%s\":",
                    filenameSlot.Param<core::param::FilePathParam>()->Value().string().c_str());
                warning = false;
            }
        }
        if (parts.Count() < (hasEl ? 4 : 3)) {
            megamol::core::utility::log::Log::DefaultLog.WriteError(
                "Line %u has too few tokens; line will be ignored", lineNum);
            continue;
        }
        if (parts.Count() > (hasEl ? 4 : 3)) {
            megamol::core::utility::log::Log::DefaultLog.WriteWarn(
                "Line %u has too many tokens; trailing tokens will be ignored", lineNum);
        }
        std::string el = (hasEl && grpEl) ? parts[0].PeekBuffer() : "";
        int o = hasEl ? 1 : 0;

        float x = NAN, y = NAN, z = NAN;
        try {
            x = static_cast<float>(vislib::CharTraitsA::ParseDouble(parts[o + 0]));
            y = static_cast<float>(vislib::CharTraitsA::ParseDouble(parts[o + 1]));
            z = static_cast<float>(vislib::CharTraitsA::ParseDouble(parts[o + 2]));
        } catch (...) {
            if (warning) {
                megamol::core::utility::log::Log::DefaultLog.WriteWarn("Problem parsing \"%s\":",
                    filenameSlot.Param<core::param::FilePathParam>()->Value().generic_u8string().c_str());
                warning = false;
            }
            megamol::core::utility::log::Log::DefaultLog.WriteError(
                "Failed to parse coordinates at line %u: (%f, %f, %f); line will be ignored", lineNum, x, y, z);
            continue;
        }

        grpDat[el].push_back(x);
        grpDat[el].push_back(y);
        grpDat[el].push_back(z);
    }

    poss.clear();
    hash++;

    if (grpDat.size() == 0) {
        bbox.Set(-1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f);
    } else {
        auto i = grpDat.begin()->second.begin();
        float x = *i;
        ++i;
        float y = *i;
        ++i;
        float z = *i;
        bbox.Set(x, y, z, x, y, z);
    }

    for (const auto& g : grpDat) {
        poss.push_back(std::vector<float>());
        auto& p = poss.back();
        p.clear();
        p.reserve(g.second.size());
        auto end = g.second.end();
        for (auto i = g.second.begin(); i != end;) {
            float x = *i;
            ++i;
            float y = *i;
            ++i;
            float z = *i;
            ++i;
            bbox.GrowToPoint(x, y, z);
            p.push_back(x);
            p.push_back(y);
            p.push_back(z);
        }
    }
}
