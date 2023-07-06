/*
 * TclMolSelectionLoader.cpp
 *
 * Copyright (C) 2015 by MegaMol Team (TU Dresden)
 * Alle Rechte vorbehalten.
 */
#include "TclMolSelectionLoader.h"
#include "mmcore/param/FilePathParam.h"
#include "vislib/sys/ASCIIFileBuffer.h"
#include <deque>
#include <vector>

using namespace megamol;
using namespace megamol::moldyn;


io::TclMolSelectionLoader::TclMolSelectionLoader()
        : core::Module()
        , getDataSlot("getdata", "Access to the data")
        , filenameSlot("filename", "Path to the Tcl file to load")
        , hash(0)
        , cnt(0)
        , data() {

    getDataSlot.SetCallback(
        geocalls::ParticleRelistCall::ClassName(), "GetData", &TclMolSelectionLoader::getDataCallback);
    getDataSlot.SetCallback(
        geocalls::ParticleRelistCall::ClassName(), "GetExtent", &TclMolSelectionLoader::getDataCallback);
    MakeSlotAvailable(&getDataSlot);

    filenameSlot.SetParameter(new core::param::FilePathParam(""));
    MakeSlotAvailable(&filenameSlot);
}

io::TclMolSelectionLoader::~TclMolSelectionLoader() {
    Release();
}

bool io::TclMolSelectionLoader::create() {
    return true;
}

void io::TclMolSelectionLoader::release() {
    clear();
    hash = 0;
}

bool io::TclMolSelectionLoader::getDataCallback(core::Call& caller) {
    geocalls::ParticleRelistCall* prc = dynamic_cast<geocalls::ParticleRelistCall*>(&caller);
    if (prc == nullptr)
        return false;
    if (filenameSlot.IsDirty())
        load();

    prc->SetDataHash(hash);
    prc->Set(cnt, data.size(), data.data());
    prc->SetExtent(1, -1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f);
    prc->SetFrameID(0, true);
    prc->SetUnlocker(nullptr);

    return true;
}

void io::TclMolSelectionLoader::clear() {
    hash++;
    cnt = 0;
    data.clear();
}

void io::TclMolSelectionLoader::load() {
    typedef geocalls::ParticleRelistCall::ListIDType ListIDType;
    filenameSlot.ResetDirty();

    vislib::sys::ASCIIFileBuffer file;
    if (!file.LoadFile(filenameSlot.Param<core::param::FilePathParam>()->Value().native().c_str(),
            vislib::sys::ASCIIFileBuffer::PARSING_WORDS)) {
        megamol::core::utility::log::Log::DefaultLog.WriteError("Unable to read file \"%s\"",
            filenameSlot.Param<core::param::FilePathParam>()->Value().generic_u8string().c_str());
        return;
    }

    // count selection particles
    uint64_t partCnt = 0;
    uint64_t maxPart = 0;
    int actCol = -1;
    std::vector<int> cols;
    ListIDType selCnt = 0;

    SIZE_T lineCount = file.Count();
    for (SIZE_T li = 0; li < lineCount; ++li) {
        const auto& line = file.Line(li);
        SIZE_T wordCount = line.Count();
        if (wordCount < 4)
            continue;
        if ((vislib::StringA(line.Word(0)).Equals("mol", false) &&
                vislib::StringA(line.Word(1)).Equals("color", false) &&
                vislib::StringA(line.Word(2)).Equals("ColorID", false))) {
            try {
                int v = vislib::CharTraitsA::ParseInt(line.Word(3));
                if (actCol != v) {
                    auto ci = std::find(cols.begin(), cols.end(), v);
                    if (ci == cols.end()) {
                        cols.push_back(v);
                    }
                    actCol = v;
                }
            } catch (...) {}
        }
        if (!(vislib::StringA(line.Word(0)).Equals("mol", false) &&
                vislib::StringA(line.Word(1)).Equals("selection", false) &&
                vislib::StringA(line.Word(2)).Equals("serial", false)))
            continue;

        for (SIZE_T wi = 3; wi < wordCount; ++wi) {
            try {
                uint64_t v = vislib::CharTraitsA::ParseUInt64(line.Word(wi)); // these are 1-based
                partCnt++;
                if (maxPart < v)
                    maxPart = v;
            } catch (...) {}
        }
    }

    bool incompleteSelection = (partCnt != maxPart);
    if (incompleteSelection)
        cols.insert(cols.begin(), -1);
    // GetCoreInstance()->Log().WriteWarn("Particle selection", vislib::StringA(filenameSlot.Param<core::param::FilePathParam>()->Value()).PeekBuffer());

    hash++;
    data.resize(maxPart);
    ::memset(data.data(), 0, sizeof(ListIDType) * maxPart);
    selCnt = 0;
    for (SIZE_T li = 0; li < lineCount; ++li) {
        const auto& line = file.Line(li);
        SIZE_T wordCount = line.Count();
        if (wordCount < 4)
            continue;
        if ((vislib::StringA(line.Word(0)).Equals("mol", false) &&
                vislib::StringA(line.Word(1)).Equals("color", false) &&
                vislib::StringA(line.Word(2)).Equals("ColorID", false))) {
            try {
                int v = vislib::CharTraitsA::ParseInt(line.Word(3));
                if (actCol != v) {
                    auto ci = std::find(cols.begin(), cols.end(), v);
                    assert(ci != cols.end());
                    actCol = v;
                    selCnt = static_cast<ListIDType>(std::distance(cols.begin(), ci));
                }
            } catch (...) {}
        }
        if (!(vislib::StringA(line.Word(0)).Equals("mol", false) &&
                vislib::StringA(line.Word(1)).Equals("selection", false) &&
                vislib::StringA(line.Word(2)).Equals("serial", false)))
            continue;

        for (SIZE_T wi = 3; wi < wordCount; ++wi) {
            try {
                uint64_t v = vislib::CharTraitsA::ParseUInt64(line.Word(wi)); // these are 1-based
                data[v - 1] = selCnt;
            } catch (...) {}
        }
    }

    cnt = static_cast<ListIDType>(cols.size());
}
