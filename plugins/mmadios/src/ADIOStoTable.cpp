/*
 * ADIOStoTable.cpp
 * Copyright (C) 2020 by MegaMol Team
 * Alle Rechte vorbehalten.
 */

#include "ADIOStoTable.h"
#include "mmadios/CallADIOSData.h"


namespace megamol::adios {


ADIOStoTable::ADIOStoTable() : Module(), _getDataSlot("getData", ""), _deployTableSlot("deployTable", "") {

    this->_deployTableSlot.SetCallback(datatools::table::TableDataCall::ClassName(),
        datatools::table::TableDataCall::FunctionName(0), &ADIOStoTable::getData);
    this->_deployTableSlot.SetCallback(datatools::table::TableDataCall::ClassName(),
        datatools::table::TableDataCall::FunctionName(1), &ADIOStoTable::getMetaData);
    this->MakeSlotAvailable(&this->_deployTableSlot);

    this->_getDataSlot.SetCompatibleCall<adios::CallADIOSDataDescription>();
    this->MakeSlotAvailable(&this->_getDataSlot);
}

ADIOStoTable::~ADIOStoTable() {
    this->Release();
}

bool ADIOStoTable::create() {
    return true;
}

void ADIOStoTable::release() {}

bool ADIOStoTable::InterfaceIsDirty() {
    return false;
}


bool ADIOStoTable::getData(core::Call& call) {

    datatools::table::TableDataCall* ctd = dynamic_cast<datatools::table::TableDataCall*>(&call);
    if (ctd == nullptr)
        return false;

    adios::CallADIOSData* cad = this->_getDataSlot.CallAs<adios::CallADIOSData>();
    if (cad == nullptr)
        return false;

    auto availVars = cad->getAvailableVars();
    // maybe get meta data was not called jet
    if (availVars.empty()) {
        this->getMetaData(call);
        availVars = cad->getAvailableVars();
    }

    bool dathashChanged = (cad->getDataHash() != ctd->DataHash());

    if ((cad->getFrameIDtoLoad() != _currentFrame) || dathashChanged) {

        ctd->SetDataHash(cad->getDataHash());

        for (auto var : availVars) {
            if (!cad->inquireVar(var)) {
                megamol::core::utility::log::Log::DefaultLog.WriteError(
                    "[ADIOStoTable] variable \"%s\" does not exist.", var.c_str());
            }
        }


        if (!(*cad)(0)) {
            megamol::core::utility::log::Log::DefaultLog.WriteError("[ADIOStoTable] Error during GetData");
            return false;
        }

        for (auto& availVar : availVars) {
            _rows = std::max(_rows, cad->getData(availVar)->size());
        }
        availVars.erase(std::remove_if(availVars.begin(), availVars.end(),
                            [&](const std::string& var) { return cad->getData(var)->size() != _rows; }),
            availVars.end());

        _cols = availVars.size();
        _colinfo.resize(_cols);
        std::vector<std::vector<float>> raw_data(_cols);
        for (int i = 0; i < availVars.size(); ++i) {
            _rows = std::max(_rows, cad->getData(availVars[i])->size());
            raw_data[i] = cad->getData(availVars[i])->GetAsFloat();
            auto prop = cad->getVarProperties(availVars[i]);
            float min = std::numeric_limits<float>::max();
            float max = std::numeric_limits<float>::lowest();
            if (prop.find("Min") != prop.end() && prop.find("Max") != prop.end()) {
                min = std::stof(prop["Min"]);
                max = std::stof(prop["Max"]);
            } else {
                for (float& j : raw_data[i]) {
                    min = std::min(min, j);
                    max = std::max(max, j);
                }
            }
            _colinfo[i].SetName(availVars[i]);
            _colinfo[i].SetMaximumValue(max);
            _colinfo[i].SetMinimumValue(min);
            _colinfo[i].SetType(datatools::table::TableDataCall::ColumnType::QUANTITATIVE);
        }

        _floatBlob.resize(_rows * _cols);
#pragma omp parallel for
        for (int i = 0; i < _rows; ++i) {
            for (int j = 0; j < _cols; ++j) {
                if (i >= raw_data[j].size()) {
                    _floatBlob[_cols * i + j] = 0.0f;
                } else {
                    _floatBlob[_cols * i + j] = raw_data[j][i];
                }
            }
        }
    }

    if (_floatBlob.empty())
        return false;

    _currentFrame = ctd->GetFrameID();
    ctd->Set(_cols, _rows, _colinfo.data(), _floatBlob.data());
    ctd->SetDataHash(cad->getDataHash());
    return true;
}

bool ADIOStoTable::getMetaData(core::Call& call) {

    datatools::table::TableDataCall* ctd = dynamic_cast<datatools::table::TableDataCall*>(&call);
    if (ctd == nullptr)
        return false;

    adios::CallADIOSData* cad = this->_getDataSlot.CallAs<adios::CallADIOSData>();
    if (cad == nullptr)
        return false;

    // get metadata from adios
    cad->setFrameIDtoLoad(ctd->GetFrameID());

    if (!(*cad)(1))
        return false;

    // put metadata in table call
    ctd->SetFrameCount(cad->getFrameCount());
    //ctd->SetDataHash(cad->getDataHash());

    return true;
}

} // namespace megamol::adios
