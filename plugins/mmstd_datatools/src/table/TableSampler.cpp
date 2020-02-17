/*
 * TableSampler.cpp
 *
 * Copyright (C) 2020 by VISUS (University of Stuttgart)
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "TableSampler.h"

#include "mmcore/param/ButtonParam.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/IntParam.h"
#include "vislib/sys/Log.h"

#include <algorithm>
#include <numeric>
#include <random>
#include <vector>

using namespace megamol::stdplugin::datatools;
using namespace megamol::stdplugin::datatools::table;
using namespace megamol;

TableSampler::TableSampler()
    : core::Module()
    , tableInSlot("getDataIn", "Float table input")
    , tableOutSlot("getDataOut", "Float table output")
    , sampleNumberModeParam("sampleNumberMode", "sample number mode")
    , sampleNumberAbsoluteParam("sampleNumberAbsolute", "number of samples")
    , sampleNumberRelativeParam("sampleNumberRelative", "percentage of samples")
    , resampleParam("resample", "resample")
    , tableInFrameCount(0)
    , tableInDataHash(0)
    , tableInColCount(0)
    , dataHash(0)
    , rowCount(0)
    , doResampling(false) {

    this->tableInSlot.SetCompatibleCall<TableDataCallDescription>();
    this->MakeSlotAvailable(&this->tableInSlot);

    this->tableOutSlot.SetCallback(TableDataCall::ClassName(),
        TableDataCall::FunctionName(0),
        &TableSampler::getData);
    this->tableOutSlot.SetCallback(TableDataCall::ClassName(),
        TableDataCall::FunctionName(1),
        &TableSampler::getHash);
    this->MakeSlotAvailable(&this->tableOutSlot);

    auto *snp = new core::param::EnumParam(SampleNumberMode::ABSOLUTE);
    snp->SetTypePair(SampleNumberMode::ABSOLUTE, "Number of samples");
    snp->SetTypePair(SampleNumberMode::RELATIVE, "Percentage");
    this->sampleNumberModeParam << snp;
    this->sampleNumberModeParam.SetUpdateCallback(this, &TableSampler::numberModeCallback);
    this->MakeSlotAvailable(&this->sampleNumberModeParam);

    this->sampleNumberAbsoluteParam << new core::param::IntParam(100, 0);
    this->sampleNumberAbsoluteParam.SetUpdateCallback(this, &TableSampler::resampleCallback);
    this->MakeSlotAvailable(&this->sampleNumberAbsoluteParam);

    auto *snrp = new core::param::FloatParam(10.0, 0.0, 100.0);
    snrp->SetGUIVisible(false);
    this->sampleNumberRelativeParam << snrp;
    this->sampleNumberRelativeParam.SetUpdateCallback(this, &TableSampler::resampleCallback);
    this->MakeSlotAvailable(&this->sampleNumberRelativeParam);

    this->resampleParam << new core::param::ButtonParam();
    this->resampleParam.SetUpdateCallback(this, &TableSampler::resampleCallback);
    this->MakeSlotAvailable(&resampleParam);
}

TableSampler::~TableSampler() {
    this->Release();
}

bool TableSampler::create() {
    return true;
}

void TableSampler::release() {
}

bool TableSampler::getData(core::Call &call) {
    if (!this->handleCall(call)) {
        return false;
    }

    auto *tableOutCall = dynamic_cast<TableDataCall *>(&call);
    tableOutCall->SetFrameCount(this->tableInFrameCount);
    tableOutCall->SetDataHash(this->dataHash);
    tableOutCall->Set(this->tableInColCount, this->rowCount, this->colInfos.data(), this->data.data());

    return true;
}

bool TableSampler::getHash(core::Call &call) {
    if (!this->handleCall(call)) {
        return false;
    }

    auto *tableOutCall = dynamic_cast<TableDataCall *>(&call);
    tableOutCall->SetFrameCount(this->tableInFrameCount);
    tableOutCall->SetDataHash(this->dataHash);

    return true;
}

bool TableSampler::handleCall(core::Call &call) {
    auto *tableOutCall = dynamic_cast<TableDataCall *>(&call);
    auto *tableInCall = this->tableInSlot.CallAs<TableDataCall>();

    if (tableOutCall == nullptr) {
        return false;
    }

    if (tableInCall == nullptr) {
        vislib::sys::Log::DefaultLog.WriteMsg(
            vislib::sys::Log::LEVEL_ERROR, "TableSampler requires a table!");
        return false;
    }

    tableInCall->SetFrameID(tableOutCall->GetFrameID());
    (*tableInCall)(1);
    (*tableInCall)(0);

    if (this->tableInFrameCount != tableInCall->GetFrameCount() || this->tableInDataHash != tableInCall->DataHash() || doResampling) {
        //vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_INFO, "TableSampler: Sample table.");

        this->dataHash++;
        this->doResampling = false;

        this->tableInFrameCount = tableInCall->GetFrameCount();
        this->tableInDataHash = tableInCall->DataHash();
        this->tableInColCount = tableInCall->GetColumnsCount();
        size_t tableInRowCount = tableInCall->GetRowsCount();

        // copy column infos
        this->colInfos.resize(this->tableInColCount);
        for (size_t i = 0; i < this->tableInColCount; ++i) {
            this->colInfos[i] = tableInCall->GetColumnsInfos()[i];
            this->colInfos[i].SetMinimumValue(std::numeric_limits<float>::max());
            this->colInfos[i].SetMaximumValue(std::numeric_limits<float>::lowest());
        }

        size_t numberOfSamples = 0;
        switch (static_cast<SampleNumberMode>(this->sampleNumberModeParam.Param<core::param::EnumParam>()->Value())) {
        case SampleNumberMode::ABSOLUTE:
            numberOfSamples = static_cast<size_t>(this->sampleNumberAbsoluteParam.Param<core::param::IntParam>()->Value());
            break;
        case SampleNumberMode::RELATIVE:
            numberOfSamples = tableInRowCount * this->sampleNumberRelativeParam.Param<core::param::FloatParam>()->Value() / 100.0;
            break;
        }

        std::vector<size_t> indexList(tableInRowCount);
        std::iota(indexList.begin(), indexList.end(), 0);

        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(indexList.begin(), indexList.end(), g);

        this->rowCount = std::min(numberOfSamples, indexList.size());
        indexList.resize(this->rowCount);

        this->data.resize(this->tableInColCount * this->rowCount);

        const float *tableInData = tableInCall->GetData();
        for (size_t r = 0; r < this->rowCount; ++r) {
            for (size_t c = 0; c < this->tableInColCount; ++c) {
                float val = tableInData[this->tableInColCount * indexList[r] + c];
                this->data[this->tableInColCount * r + c] = val;
                if (val < this->colInfos[c].MinimumValue()) {
                    this->colInfos[c].SetMinimumValue(val);
                }
                if (val > this->colInfos[c].MaximumValue()) {
                    this->colInfos[c].SetMaximumValue(val);
                }
            }
        }

        // nicer output
        if (this->rowCount == 0) {
            for (size_t i = 0; i < this->tableInColCount; ++i) {
                this->colInfos[i].SetMinimumValue(0.0);
                this->colInfos[i].SetMaximumValue(0.0);
            }
        }
    }

    return true;
}

bool TableSampler::resampleCallback(core::param::ParamSlot &caller) {
    this->doResampling = true;
    return true;
}

bool TableSampler::numberModeCallback(core::param::ParamSlot &caller) {
    this->doResampling = true;
    this->sampleNumberAbsoluteParam.Param<core::param::IntParam>()->SetGUIVisible(false);
    this->sampleNumberRelativeParam.Param<core::param::FloatParam>()->SetGUIVisible(false);

    switch (static_cast<SampleNumberMode>(this->sampleNumberModeParam.Param<core::param::EnumParam>()->Value())) {
    case SampleNumberMode::ABSOLUTE:
        this->sampleNumberAbsoluteParam.Param<core::param::IntParam>()->SetGUIVisible(true);
        break;
    case SampleNumberMode::RELATIVE:
        this->sampleNumberRelativeParam.Param<core::param::FloatParam>()->SetGUIVisible(true);
        break;
    }
    return true;
}
