/*
 * TableToADIOS.h
 *
 * Copyright (C) 2018 by Universitaet Stuttgart (VISUS).
 * Alle Rechte vorbehalten.
 */

#include "TableToADIOS.h"

#include "datatools/table/TableDataCall.h"
#include "mmadios/CallADIOSData.h"
#include "mmcore/param/EnumParam.h"

#include "mmcore/utility/log/Log.h"

#include "vislib/StringConverter.h"

namespace megamol::adios {

TableToADIOS::TableToADIOS()
        : core::Module()
        , ftSlot("ftSlot", "Slot to request table data from")
        , adiosSlot("adiosSlot", "Slot to send ADIOS IO to") {


    this->adiosSlot.SetCallback(
        CallADIOSData::ClassName(), CallADIOSData::FunctionName(0), &TableToADIOS::getDataCallback);
    this->adiosSlot.SetCallback(
        CallADIOSData::ClassName(), CallADIOSData::FunctionName(1), &TableToADIOS::getHeaderCallback);
    this->MakeSlotAvailable(&this->adiosSlot);

    this->ftSlot.SetCompatibleCall<datatools::table::TableDataCallDescription>();
    this->MakeSlotAvailable(&this->ftSlot);
}

TableToADIOS::~TableToADIOS() {
    this->Release();
}

bool TableToADIOS::create() {
    return true;
}

void TableToADIOS::release() {}

bool TableToADIOS::getDataCallback(core::Call& call) {
    CallADIOSData* cad = dynamic_cast<CallADIOSData*>(&call);
    if (cad == nullptr)
        return false;

    datatools::table::TableDataCall* cftd = this->ftSlot.CallAs<datatools::table::TableDataCall>();
    if (cftd == nullptr)
        return false;

    if (!(*cftd)(1))
        return false;

    // set frame to load from view
    cftd->SetFrameID(cad->getFrameIDtoLoad());

    if (!(*cftd)(0))
        return false;

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

bool TableToADIOS::getHeaderCallback(core::Call& call) {

    CallADIOSData* cad = dynamic_cast<CallADIOSData*>(&call);
    if (cad == nullptr)
        return false;

    datatools::table::TableDataCall* cftd = this->ftSlot.CallAs<datatools::table::TableDataCall>();
    if (cftd == nullptr)
        return false;

    if (!(*cftd)(1))
        return false;
    if (!(*cftd)(0))
        return false;
    std::vector<std::string> availVars;
    availVars.reserve(cftd->GetColumnsCount());

    for (size_t i = 0; i < cftd->GetColumnsCount(); i++) {
        std::string n = std::string(this->cleanUpColumnHeader(cftd->GetColumnsInfos()[i].Name().c_str()));

        availVars.push_back(n);
    }
    // set available vars
    cad->setAvailableVars(availVars);

    // get total frame count from data source
    cad->setFrameCount(cftd->GetFrameCount());

    return true;
}

std::string TableToADIOS::cleanUpColumnHeader(const vislib::TString& header) const {
    vislib::TString h(header);
    h.TrimSpaces();
    h.ToLowerCase();
    return std::string(T2A(h.PeekBuffer()));
}

} // namespace megamol::adios
