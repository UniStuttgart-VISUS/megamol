/*
 * FloatTableColumnFilter.cpp
 *
 * Copyright (C) 2016-2017 by VISUS (University of Stuttgart)
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "FloatTableColumnFilter.h"

#include "mmcore/param/StringParam.h"
#include "vislib/StringTokeniser.h"
#include "vislib/sys/Log.h"


std::string megamol::stdplugin::datatools::floattable::FloatTableColumnFilter::ModuleName
    = std::string("FloatTableColumnFilter");

megamol::stdplugin::datatools::floattable::FloatTableColumnFilter::FloatTableColumnFilter(void) :
	core::Module(),
	dataOutSlot("dataOut", "Ouput"),
	dataInSlot("dataIn", "Input"),
	selectionStringSlot("selection", "Select columns by name separated by \";\""),
	frameID(-1),
	datahash(MAXULONG_PTR) {
    this->dataInSlot.SetCompatibleCall<CallFloatTableDataDescription>();
    this->MakeSlotAvailable(&this->dataInSlot);

    this->dataOutSlot.SetCallback(CallFloatTableData::ClassName(),
        CallFloatTableData::FunctionName(0),
        &FloatTableColumnFilter::processData);
    this->dataOutSlot.SetCallback(CallFloatTableData::ClassName(),
        CallFloatTableData::FunctionName(1),
        &FloatTableColumnFilter::getExtent);
    this->MakeSlotAvailable(&this->dataOutSlot);

    this->selectionStringSlot << new core::param::StringParam("x; y; z");
    this->MakeSlotAvailable(&this->selectionStringSlot);
}

megamol::stdplugin::datatools::floattable::FloatTableColumnFilter::~FloatTableColumnFilter(void) {
    this->Release();
}

bool megamol::stdplugin::datatools::floattable::FloatTableColumnFilter::create(void) {
    return true;
}

void megamol::stdplugin::datatools::floattable::FloatTableColumnFilter::release(void) {
}

bool megamol::stdplugin::datatools::floattable::FloatTableColumnFilter::processData(core::Call &c) {
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

            auto column_count = inCall->GetColumnsCount();
            auto column_infos = inCall->GetColumnsInfos();
            auto rows_count = inCall->GetRowsCount();
            auto in_data = inCall->GetData();

            auto selectionString = this->selectionStringSlot.Param<core::param::StringParam>()->Value();
            selectionString.Remove(vislib::TString(" "));
            auto st = vislib::StringTokeniserW(selectionString, vislib::TString(";"));
            auto selectors = st.Split(selectionString, vislib::TString(";"));

            if (selectors.Count() == 0) {
                vislib::sys::Log::DefaultLog.WriteError(_T("%hs: No valid selectors have been given\n"),
                    ModuleName.c_str());
                return false;
            }

            this->columnInfos.clear();
            this->columnInfos.reserve(selectors.Count());

            std::vector<size_t> indexMask;
            indexMask.reserve(selectors.Count());
            for (size_t sel = 0; sel < selectors.Count(); sel++) {
                for (size_t col = 0; col < column_count; col++) {
                    if (selectors[sel].CompareInsensitive(vislib::TString(
                        column_infos[col].Name().c_str()))) {
                        indexMask.push_back(col);
                        this->columnInfos.push_back(column_infos[col]);
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
                this->columnInfos.clear();
                this->data.clear();
                return false;
            }

            this->data.clear();
            this->data.reserve(rows_count*this->columnInfos.size());

            for (size_t row = 0; row < rows_count; row++) {
                for (auto &cidx : indexMask) {
                    this->data.push_back(in_data[cidx + row*column_count]);
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
        vislib::sys::Log::DefaultLog.WriteError(_T("Failed to execute %hs::processData\n"),
            ModuleName.c_str());
        return false;
    }

    return true;
}

bool megamol::stdplugin::datatools::floattable::FloatTableColumnFilter::getExtent(core::Call &c) {
	try {
		CallFloatTableData *outCall = dynamic_cast<CallFloatTableData *>(&c);
		if (outCall == NULL) return false;

		CallFloatTableData *inCall = this->dataInSlot.CallAs<CallFloatTableData>();
		if (inCall == NULL) return false;

		inCall->SetFrameID(outCall->GetFrameID());
		if (!(*inCall)(1)) return false;

		outCall->SetFrameCount(inCall->GetFrameCount());
		outCall->SetDataHash(this->datahash);
	}
	catch (...) {
		vislib::sys::Log::DefaultLog.WriteError(_T("Failed to execute %hs::getExtent\n"), ModuleName.c_str());
		return false;
	}

	return true;
}
