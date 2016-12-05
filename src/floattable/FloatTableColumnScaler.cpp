/*
 * FloatTableColumnScaler.cpp
 *
 * Copyright (C) 2016-2016 by VISUS (University of Stuttgart)
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "FloatTableColumnScaler.h"

#include "mmcore/param/FloatParam.h"
#include "mmcore/param/StringParam.h"

#include "vislib/StringTokeniser.h"

#include "vislib/sys/Log.h"


/*
 * FloatTableColumnScaler::ModuleName
 */
std::string megamol::stdplugin::datatools::floattable::FloatTableColumnScaler::ModuleName
    = std::string("FloatTableColumnScaler");


/*
 * FloatTableColumnSelector::FloatTableColumnSelector
 */
megamol::stdplugin::datatools::floattable::FloatTableColumnScaler::FloatTableColumnScaler(void) :
    core::Module(),
    dataInSlot("dataIn", "Input"),
    dataOutSlot("dataOut", "Output"),
    scalingFactorSlot("scalingFactor", "Factor by which the selected column get scaled"),
    columnSelectorSlot("columns", "Select columns to scale separated by \";\""),
    frameID(-1),
    datahash(MAXULONG_PTR) {
    this->dataInSlot.SetCompatibleCall<CallFloatTableDataDescription>();
    this->MakeSlotAvailable(&this->dataInSlot);

    this->dataOutSlot.SetCallback(CallFloatTableData::ClassName(),
        CallFloatTableData::FunctionName(0),
        &FloatTableColumnScaler::processData);
    this->dataOutSlot.SetCallback(CallFloatTableData::ClassName(),
        CallFloatTableData::FunctionName(1),
        &FloatTableColumnScaler::getExtent);
    this->MakeSlotAvailable(&this->dataOutSlot);

    this->scalingFactorSlot << new core::param::FloatParam(1.0f);
    this->MakeSlotAvailable(&this->scalingFactorSlot);

    this->columnSelectorSlot << new core::param::StringParam("c");
    this->MakeSlotAvailable(&this->columnSelectorSlot);
}


/*
 * FloatTableColumnSelector::~FloatTableColumnSelector
 */
megamol::stdplugin::datatools::floattable::FloatTableColumnScaler::~FloatTableColumnScaler(void) {
    this->Release();
}


/*
 * FloatTableColumnSelector::create
 */
bool megamol::stdplugin::datatools::floattable::FloatTableColumnScaler::create(void) {
    return true;
}


/*
 * FloatTableColumnSelector::release
 */
void megamol::stdplugin::datatools::floattable::FloatTableColumnScaler::release(void) {

}


/*
 * FloatTableColumnSelector::processData
 */
bool megamol::stdplugin::datatools::floattable::FloatTableColumnScaler::processData(core::Call &c) {
    try {
        CallFloatTableData *outCall = dynamic_cast<CallFloatTableData *>(&c);
        if (outCall == NULL) return false;

        CallFloatTableData *inCall = this->dataInSlot.CallAs<CallFloatTableData>();
        if (inCall == NULL) return false;

        inCall->SetFrameID(outCall->GetFrameID());
        if (!(*inCall)()) return false;

        if (this->datahash != inCall->DataHash() || this->frameID != inCall->GetFrameID()) {
            this->datahash = inCall->DataHash();
            this->frameID = inCall->GetFrameID();

            auto rows_count = inCall->GetRowsCount();
            auto column_count = inCall->GetColumnsCount();
            auto in_data = inCall->GetData();
            auto column_infos = inCall->GetColumnsInfos();

            auto scalingFactor = this->scalingFactorSlot.Param<core::param::FloatParam>()->Value();

            auto selectionString = this->columnSelectorSlot.Param<core::param::StringParam>()->Value();
            selectionString.Remove(vislib::TString(" "));
            auto st = vislib::StringTokeniserW(selectionString, vislib::TString(";"));
            auto selectors = st.Split(selectionString, vislib::TString(";"));

            if (selectors.Count() == 0) {
                vislib::sys::Log::DefaultLog.WriteError(_T("%hs: No valid selectors have been given\n"),
                    ModuleName.c_str());
                return false;
            }

            std::vector<size_t> indexMask;
            indexMask.reserve(selectors.Count());
            for (size_t sel = 0; sel < selectors.Count(); sel++) {
                for (size_t col = 0; col < column_count; col++) {
                    if (selectors[sel].CompareInsensitive(vislib::TString(
                        column_infos[col].Name().c_str()))) {
                        indexMask.push_back(col);
                        break;
                    }
                }
                //// if we reach this, no match has been found
                //vislib::sys::Log::DefaultLog.WriteInfo(_T("%hs: No match has been found for selector %s\n"),
                //    ModuleName.c_str(), selectors[sel].PeekBuffer());
            }

            if (indexMask.size() == 0) {
                vislib::sys::Log::DefaultLog.WriteError(_T("%hs: No matches for selectors have been found\n"),
                    ModuleName.c_str());
                /*this->columnInfos.clear();
                this->data.clear();*/
                return false;
            }

            this->columnInfos.clear();
            this->columnInfos.reserve(column_count);
            for (size_t i = 0; i < column_count; i++) {
                this->columnInfos.push_back(column_infos[i]);
            }

            for (auto &col : indexMask) {
                this->columnInfos[col].SetMinimumValue(this->columnInfos[col].MinimumValue()*scalingFactor);
                this->columnInfos[col].SetMaximumValue(this->columnInfos[col].MaximumValue()*scalingFactor);
            }

            this->data.clear();
            this->data.resize(rows_count*column_count);
            memcpy(this->data.data(), in_data, sizeof(float)*rows_count*column_count);
            for (size_t row = 0; row < rows_count; row++) {
                for (auto &col : indexMask) {
                    this->data[col + row*column_count] *= scalingFactor;
                }
            }
        }

        outCall->SetFrameCount(inCall->GetFrameCount());
        outCall->SetFrameID(this->frameID);
        outCall->SetDataHash(this->datahash);

        if (this->data.size() != 0 && this->columnInfos.size() != 0) {
            outCall->Set(this->columnInfos.size(), this->data.size() / this->columnInfos.size(),
                this->columnInfos.data(), this->data.data());
        } else {
            outCall->Set(0, 0, NULL, NULL);
        }
    } catch (...) {
        vislib::sys::Log::DefaultLog.WriteError(_T("Failed to execute %hs::processData\n"), ModuleName.c_str());
        return false;
    }

    return true;
}


/*
 * FloatTableColumnSelector::getExtent
 */
bool megamol::stdplugin::datatools::floattable::FloatTableColumnScaler::getExtent(core::Call &c) {
    try {
        CallFloatTableData *outCall = dynamic_cast<CallFloatTableData *>(&c);
        if (outCall == NULL) return false;

        CallFloatTableData *inCall = this->dataInSlot.CallAs<CallFloatTableData>();
        if (inCall == NULL) return false;

        inCall->SetFrameID(outCall->GetFrameID());
        if (!(*inCall)(1)) return false;

        outCall->SetFrameCount(inCall->GetFrameCount());
        //outCall->SetFrameID(this->frameID);
        outCall->SetDataHash(this->datahash);
    } catch (...) {
        vislib::sys::Log::DefaultLog.WriteError(_T("Failed to execute %hs::getExtent\n"), ModuleName.c_str());
        return false;
    }

    return true;
}