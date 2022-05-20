/*
 * QuartzCrystalDataSource.cpp
 *
 * Copyright (C) 2018 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */
#include "QuartzCrystalDataSource.h"
#include "mmcore/param/FilePathParam.h"
#include "mmcore/utility/log/Log.h"
#include "stdafx.h"
#include "vislib/CharTraits.h"
#include "vislib/math/Vector.h"
#include "vislib/math/mathfunctions.h"
#include "vislib/sys/ASCIIFileBuffer.h"


namespace megamol {
namespace demos_gl {

/*
 * CrystalDataSource::CrystalDataSource
 */
CrystalDataSource::CrystalDataSource(void)
        : core::Module()
        , filenameSlot("filename", "The path to the file to be loaded")
        , dataOutSlot("dataout", "The slot providing the loaded data")
        , datahash(0)
        , crystals() {

    this->filenameSlot << new core::param::FilePathParam("");
    this->MakeSlotAvailable(&this->filenameSlot);

    this->dataOutSlot.SetCallback(CrystalDataCall::ClassName(),
        CrystalDataCall::FunctionName(CrystalDataCall::CallForGetData), &CrystalDataSource::getData);
    this->dataOutSlot.SetCallback(CrystalDataCall::ClassName(),
        CrystalDataCall::FunctionName(CrystalDataCall::CallForGetExtent), &CrystalDataSource::getData);
    this->MakeSlotAvailable(&this->dataOutSlot);
}


/*
 * CrystalDataSource::~CrystalDataSource
 */
CrystalDataSource::~CrystalDataSource(void) {
    this->Release();
}


/*
 * CrystalDataSource::create
 */
bool CrystalDataSource::create(void) {
    // intentionally empty
    return true;
}


/*
 * CrystalDataSource::getData
 */
bool CrystalDataSource::getData(core::Call& c) {
    CrystalDataCall* cdc = dynamic_cast<CrystalDataCall*>(&c);
    if (cdc == NULL)
        return false;

    if (this->filenameSlot.IsDirty()) {
        this->filenameSlot.ResetDirty();
        this->loadFile(this->filenameSlot.Param<core::param::FilePathParam>()->Value().generic_u8string().c_str());
    }

    cdc->SetDataHash(this->datahash);
    cdc->SetCrystals(static_cast<unsigned int>(this->crystals.Count()), this->crystals.PeekElements());
    cdc->SetUnlocker(NULL);

    return true;
}


/*
 * CrystalDataSource::release
 */
void CrystalDataSource::release(void) {
    this->crystals.Clear();
}


/*
 * CrystalDataSource::loadFile
 */
void CrystalDataSource::loadFile(const vislib::TString& filename) {
    using megamol::core::utility::log::Log;
    vislib::sys::ASCIIFileBuffer file(vislib::sys::ASCIIFileBuffer::PARSING_WORDS);
    if (!file.LoadFile(filename)) {
        Log::DefaultLog.WriteError("Unable to read file \"%s\"", vislib::StringA(filename).PeekBuffer());
        return;
    }

    this->crystals.Clear();
    this->datahash++;
    CrystalDataCall::Crystal crystal;

    for (SIZE_T l = 0; l < file.Count(); l++) {
        if (file[l].Count() == 0) {
            continue;
        } else if (file[l].Count() == 3) {
            try {
                vislib::math::Vector<float, 3> v(static_cast<float>(vislib::CharTraitsA::ParseDouble(file[l].Word(0))),
                    static_cast<float>(vislib::CharTraitsA::ParseDouble(file[l].Word(1))),
                    static_cast<float>(vislib::CharTraitsA::ParseDouble(file[l].Word(2))));
                // Do not normalize! Length of v is radius of base-sphere for that plane
                crystal.AddFace(v);
            } catch (...) { Log::DefaultLog.WriteWarn("Error parsing vector line \"%d\"", static_cast<int>(l)); }
        } else if (file[l].Count() == 2) {
            try {
                crystal.SetBaseRadius(static_cast<float>(vislib::CharTraitsA::ParseDouble(file[l].Word(1))));
                crystal.SetBoundingRadius(static_cast<float>(vislib::CharTraitsA::ParseDouble(file[l].Word(0))));
            } catch (...) { Log::DefaultLog.WriteWarn("Error parsing radius line \"%d\"", static_cast<int>(l)); }
            if (!crystal.IsEmpty())
                this->crystals.Add(crystal);
            crystal.Clear();
        } else {
            Log::DefaultLog.WriteWarn("Strange line \"%d\" ignored", static_cast<int>(l));
        }
    }

    Log::DefaultLog.WriteInfo(
        "Loaded \"%u\" crystalite definitions\n", static_cast<unsigned int>(this->crystals.Count()));
}
} // namespace demos_gl
} /* end namespace megamol */
