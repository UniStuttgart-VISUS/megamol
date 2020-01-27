/*
 * TableFlagFilter.cpp
 *
 * Copyright (C) 2020 by VISUS (University of Stuttgart)
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "TableFlagFilter.h"

#include "vislib/sys/Log.h"

using namespace megamol::stdplugin::datatools;
using namespace megamol::stdplugin::datatools::table;
using namespace megamol;

TableFlagFilter::TableFlagFilter()
    : core::Module()
    , tableInSlot("getDataIn", "Float table input")
    , flagStorageInSlot("readFlagStorage", "Flag storage read input")
    , tableOutSlot("getDataOut", "Float table output")
    , frameCount(0)
    , dataHash(0)
    , colCount(0)
    , rowCount(0) {

    this->tableInSlot.SetCompatibleCall<TableDataCallDescription>();
    this->MakeSlotAvailable(&this->tableInSlot);

    this->flagStorageInSlot.SetCompatibleCall<core::FlagCallRead_GLDescription>();
    this->MakeSlotAvailable(&this->flagStorageInSlot);

    this->tableOutSlot.SetCallback(TableDataCall::ClassName(),
        TableDataCall::FunctionName(0),
        &TableFlagFilter::getData);
    this->tableOutSlot.SetCallback(TableDataCall::ClassName(),
        TableDataCall::FunctionName(1),
        &TableFlagFilter::getHash);
    this->MakeSlotAvailable(&this->tableOutSlot);
}

TableFlagFilter::~TableFlagFilter() {
    this->Release();
}

bool TableFlagFilter::create() {
    return true;
}

void TableFlagFilter::release() {
}

bool TableFlagFilter::getData(core::Call &call) {
    if (!this->handleCall(call)) {
        return false;
    }

    auto *tableOutCall = dynamic_cast<TableDataCall *>(&call);
    tableOutCall->SetFrameCount(this->frameCount);
    tableOutCall->SetDataHash(this->dataHash);
    tableOutCall->Set(this->colCount, this->rowCount, this->colInfos.data(), this->data.data());

    return true;
}

bool TableFlagFilter::getHash(core::Call &call) {
    if (!this->handleCall(call)) {
        return false;
    }

    auto *tableOutCall = dynamic_cast<TableDataCall *>(&call);
    tableOutCall->SetFrameCount(this->frameCount);
    tableOutCall->SetDataHash(this->dataHash);

    return true;
}

bool TableFlagFilter::handleCall(core::Call &call) {
    auto *tableOutCall = dynamic_cast<TableDataCall *>(&call);
    auto *tableInCall = this->tableInSlot.CallAs<TableDataCall>();
    auto *flagsInCall = this->flagStorageInSlot.CallAs<core::FlagCallRead_GL>();

    if (tableOutCall == nullptr) {
        return false;
    }

    if (tableInCall == nullptr) {
        vislib::sys::Log::DefaultLog.WriteMsg(
            vislib::sys::Log::LEVEL_ERROR, "TableFlagFilter requires a table!");
        return false;
    }

    if (flagsInCall == nullptr) {
        vislib::sys::Log::DefaultLog.WriteMsg(
            vislib::sys::Log::LEVEL_ERROR, "TableFlagFilter requires a flag storage!");
        return false;
    }

    tableInCall->SetFrameID(tableOutCall->GetFrameID());
    (*tableInCall)(1);
    (*tableInCall)(0);

    if (this->frameCount != tableInCall->GetFrameCount() || this->dataHash != tableInCall->DataHash()) {
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_INFO, "TableFlagFilter: Filter table.");
        this->frameCount = tableInCall->GetFrameCount();
        this->dataHash = tableInCall->DataHash();
        this->colCount = tableInCall->GetColumnsCount();
        this->rowCount = tableInCall->GetRowsCount();
        this->colInfos.resize(this->colCount);
        for (size_t i = 0; i < this->colCount; ++i) {
            this->colInfos[i] = tableInCall->GetColumnsInfos()[i];
        }
        tableInCall->GetColumnsInfos();
        this->data.resize(this->colCount * this->rowCount);
        std::memcpy(this->data.data(), tableInCall->GetData(), this->colCount * this->rowCount * sizeof(float));
    }

    return true;
}
