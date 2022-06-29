/*
 * TableColumnFilter.cpp
 *
 * Copyright (C) 2016-2017 by VISUS (University of Stuttgart)
 * Alle Rechte vorbehalten.
 */

#include "TableColumnFilter.h"

#include "mmcore/param/StringParam.h"

#include "mmcore/utility/log/Log.h"
#include "vislib/StringTokeniser.h"
#include <limits>

using namespace megamol::datatools;
using namespace megamol::datatools::table;
using namespace megamol;

std::string TableColumnFilter::ModuleName = std::string("TableColumnFilter");

TableColumnFilter::TableColumnFilter(void)
        : core::Module()
        , dataOutSlot("dataOut", "Ouput")
        , dataInSlot("dataIn", "Input")
        , selectionStringSlot("selection", "Select columns by name separated by \";\"")
        , frameID(-1)
        , inDatahash(std::numeric_limits<unsigned long>::max())
        , datahash(std::numeric_limits<unsigned long>::max()) {

    this->dataInSlot.SetCompatibleCall<TableDataCallDescription>();
    this->MakeSlotAvailable(&this->dataInSlot);

    this->dataOutSlot.SetCallback(
        TableDataCall::ClassName(), TableDataCall::FunctionName(0), &TableColumnFilter::processData);
    this->dataOutSlot.SetCallback(
        TableDataCall::ClassName(), TableDataCall::FunctionName(1), &TableColumnFilter::getExtent);
    this->MakeSlotAvailable(&this->dataOutSlot);

    this->selectionStringSlot << new core::param::StringParam("x; y; z");
    this->MakeSlotAvailable(&this->selectionStringSlot);
}

TableColumnFilter::~TableColumnFilter(void) {
    this->Release();
}

bool TableColumnFilter::create(void) {
    return true;
}

void TableColumnFilter::release(void) {}

bool TableColumnFilter::processData(core::Call& c) {
    try {
        TableDataCall* outCall = dynamic_cast<TableDataCall*>(&c);
        if (outCall == NULL)
            return false;

        TableDataCall* inCall = this->dataInSlot.CallAs<TableDataCall>();
        if (inCall == NULL)
            return false;

        inCall->SetFrameID(outCall->GetFrameID());
        if (!(*inCall)())
            return false;

        if (this->inDatahash != inCall->DataHash() || this->frameID != inCall->GetFrameID() ||
            this->selectionStringSlot.IsDirty()) {
            this->inDatahash = inCall->DataHash();
            this->datahash++;
            this->selectionStringSlot.ResetDirty();
            this->frameID = inCall->GetFrameID();

            auto column_count = inCall->GetColumnsCount();
            auto column_infos = inCall->GetColumnsInfos();
            auto rows_count = inCall->GetRowsCount();
            auto in_data = inCall->GetData();

            auto selectionString =
                vislib::TString(this->selectionStringSlot.Param<core::param::StringParam>()->Value().c_str());
            //selectionString.Remove(vislib::TString(" "));
            auto st = vislib::StringTokeniserW(selectionString, vislib::TString(";"));
            auto selectors = st.Split(selectionString, vislib::TString(";"));

            if (selectors.Count() == 0) {
                megamol::core::utility::log::Log::DefaultLog.WriteError(
                    _T("%hs: No valid selectors have been given\n"), ModuleName.c_str());
                return false;
            }

            this->columnInfos.clear();
            this->columnInfos.reserve(selectors.Count());

            std::vector<size_t> indexMask;
            indexMask.reserve(selectors.Count());
            for (size_t sel = 0; sel < selectors.Count(); sel++) {
                for (size_t col = 0; col < column_count; col++) {
                    if (selectors[sel].CompareInsensitive(vislib::TString(column_infos[col].Name().c_str()))) {
                        indexMask.push_back(col);
                        this->columnInfos.push_back(column_infos[col]);
                        break;
                    }
                }
                //// if we reach this, no match has been found
                //megamol::core::utility::log::Log::DefaultLog.WriteInfo(_T("%hs: No match has been found for selector %s\n"),
                //    ModuleName.c_str(), selectors[sel].PeekBuffer());
            }

            if (indexMask.size() == 0) {
                megamol::core::utility::log::Log::DefaultLog.WriteError(
                    _T("%hs: No matches for selectors have been found\n"), ModuleName.c_str());
                this->columnInfos.clear();
                this->data.clear();
                return false;
            }

            this->data.clear();
            this->data.reserve(rows_count * this->columnInfos.size());

            for (size_t row = 0; row < rows_count; row++) {
                for (auto& cidx : indexMask) {
                    this->data.push_back(in_data[cidx + row * column_count]);
                }
            }
        }

        outCall->SetFrameCount(inCall->GetFrameCount());
        outCall->SetFrameID(this->frameID);
        outCall->SetDataHash(this->datahash);

        if (this->columnInfos.size() != 0) {
            outCall->Set(this->columnInfos.size(), this->data.size() / this->columnInfos.size(),
                this->columnInfos.data(), this->data.data());
        } else {
            outCall->Set(0, 0, NULL, NULL);
        }
    } catch (...) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            _T("Failed to execute %hs::processData\n"), ModuleName.c_str());
        return false;
    }

    return true;
}

bool TableColumnFilter::getExtent(core::Call& c) {
    try {
        TableDataCall* outCall = dynamic_cast<TableDataCall*>(&c);
        if (outCall == NULL)
            return false;

        TableDataCall* inCall = this->dataInSlot.CallAs<TableDataCall>();
        if (inCall == NULL)
            return false;

        inCall->SetFrameID(outCall->GetFrameID());
        if (!(*inCall)(1))
            return false;

        outCall->SetFrameCount(inCall->GetFrameCount());
        outCall->SetDataHash(this->datahash);
    } catch (...) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            _T("Failed to execute %hs::getExtent\n"), ModuleName.c_str());
        return false;
    }

    return true;
}
