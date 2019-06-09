/*
 * FloatTabletoADIOS.h
 *
 * Copyright (C) 2018 by Universitaet Stuttgart (VISUS).
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "FloatTabletoADIOS.h"
#include "CallADIOSData.h"
#include "mmstd_datatools/floattable/CallFloatTableData.h"
#include "mmcore/param/EnumParam.h"
#include "vislib/sys/Log.h"

namespace megamol {
namespace adios {

FloatTabletoADIOS::FloatTabletoADIOS(void)
    : core::Module()
    , ftSlot("ftSlot", "Slot to request multi particle data.")
    , adiosSlot("adiosSlot", "Slot to send ADIOS IO") {


    this->adiosSlot.SetCallback(
        CallADIOSData::ClassName(), CallADIOSData::FunctionName(0), &FloatTabletoADIOS::getDataCallback);
    this->adiosSlot.SetCallback(
        CallADIOSData::ClassName(), CallADIOSData::FunctionName(1), &FloatTabletoADIOS::getHeaderCallback);
    this->MakeSlotAvailable(&this->adiosSlot);

    this->ftSlot.SetCompatibleCall<stdplugin::datatools::floattable::CallFloatTableDataDescription>();
    this->MakeSlotAvailable(&this->ftSlot);
}

FloatTabletoADIOS::~FloatTabletoADIOS(void) { this->Release(); }

bool FloatTabletoADIOS::create(void) { return true; }

void FloatTabletoADIOS::release(void) {}

bool FloatTabletoADIOS::getDataCallback(core::Call& call) {
    CallADIOSData* cad = dynamic_cast<CallADIOSData*>(&call);
    if (cad == nullptr) return false;

    stdplugin::datatools::floattable::CallFloatTableData* cftd = this->ftSlot.CallAs<stdplugin::datatools::floattable::CallFloatTableData>();
    if (cftd == nullptr) return false;

    if (!(*cftd)(1)) return false;

    // set frame to load from view
    cftd->SetFrameID(cad->getFrameIDtoLoad());

    if (!(*cftd)(0)) return false;

    // Get list of column names


    for (size_t i = 0; i < cftd->GetColumnsCount(); i++) {
        std::string n = std::string(this->cleanUpColumnHeader(cftd->GetColumnsInfos()[i].Name().c_str()));
        columnIndex[n] = i;

        std::shared_ptr<FloatContainer> fcontainer = std::make_shared<FloatContainer>(FloatContainer());
        std::vector<float>& tmp_dataptr = fcontainer->getVec();
        tmp_dataptr.reserve(cftd->GetRowsCount());

        for (size_t j = 0; j < cftd->GetRowsCount(); j++) {
            tmp_dataptr.push_back(cftd->GetData(i, j));
        }

        dataMap[n] = std::move(fcontainer);
    }



    // set stuff in call
    cad->setData(std::make_shared<adiosDataMap>(dataMap));
    cad->setDataHash(cftd->DataHash());

    return true;
}

bool FloatTabletoADIOS::getHeaderCallback(core::Call& call) {

    CallADIOSData* cad = dynamic_cast<CallADIOSData*>(&call);
    if (cad == nullptr) return false;

    stdplugin::datatools::floattable::CallFloatTableData* cftd = this->ftSlot.CallAs<stdplugin::datatools::floattable::CallFloatTableData>();
    if (cftd == nullptr) return false;

    if (!(*cftd)(1)) return false;
    if (!(*cftd)(0)) return false;
    std::vector<std::string> availVars;
    availVars.reserve(cftd->GetColumnsCount());

    for (size_t i = 0; i < cftd->GetColumnsCount(); i++) {
        std::string n = std::string(this->cleanUpColumnHeader(cftd->GetColumnsInfos()[i].Name().c_str()));

        availVars.push_back(n);
    }
    // set available vars
    cad->setAvailableVars(availVars);

    // get total frame cound from data source
    cad->setFrameCount(cftd->GetFrameCount());

    return true;
}

std::string FloatTabletoADIOS::cleanUpColumnHeader(const vislib::TString& header) const {
    vislib::TString h(header);
    h.TrimSpaces();
    h.ToLowerCase();
    return std::string(T2A(h.PeekBuffer()));
}

} // end namespace adios
} // end namespace megamol