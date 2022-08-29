/*
 * CSVFileSequence.cpp
 *
 * Copyright (C) 2019 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#include "CSVFileSequence.h"
#include "PluginsResource.h"
#include "datatools/table/TableDataCall.h"
#include "mmcore/MegaMolGraph.h"
#include "mmcore/factories/CallDescriptionManager.h"
#include "mmcore/param/BoolParam.h"
#include "mmcore/param/FilePathParam.h"
#include "mmcore/param/IntParam.h"
#include "mmcore/param/StringParam.h"
#include "mmcore/utility/log/Log.h"
#include "vislib/String.h"
#include "vislib/sys/File.h"

using namespace megamol;


/*
 * CSVFileSequence::CSVFileSequence
 */
datatools::CSVFileSequence::CSVFileSequence()
        : core::Module()
        , fileNameTemplateSlot("fileNameTemplate", "The file name template"
                                                   " example: D:\\data\\Kohler\\nial\\nialout50_*5{0..599+2}*.crist "
                                                   " Syntax: *[[DIG][{MIN..MAX[+STEP]}]*] "
                                                   " Currently MIN, MAX, and STEP work globally!!! "
                                                   " Currently only ONE '*'-Sequence is allowed!!!")
        , fileNumberMinSlot("fileNumberMin", "Slot for the minimum file number")
        , fileNumberMaxSlot("fileNumberMax", "Slot for the maximum file number")
        , fileNumberStepSlot("fileNumberStep", "Slot for the file number increase step")
        , fileNameSlotNameSlot("fileNameSlotName", "The name of the data source file name parameter slot (e.g. "
                                                   "'filename', not the whole path to module and slot)")
        , useClipBoxAsBBox("useClipBoxAsBBox", "If true will use the all-data clip box as bounding box")
        , outDataSlot("outData", "The slot for publishing data to the writer")
        , inDataSlot("inData", "The slot for requesting data from the source")
        , clipbox(-1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f)
        , datahash(0)
        , fileNameTemplate(_T(""))
        , fileNumMin(0)
        , fileNumMax(0)
        , fileNumStep(1)
        , needDataUpdate(true)
        , frameCnt(1)
        , lastIdxRequested(0) {

    this->fileNameTemplateSlot << new core::param::StringParam(this->fileNameTemplate.PeekBuffer());
    this->fileNameTemplateSlot.SetUpdateCallback(&CSVFileSequence::onFileNameTemplateChanged);
    this->MakeSlotAvailable(&this->fileNameTemplateSlot);

    this->fileNumberMinSlot << new core::param::IntParam(this->fileNumMin, 0);
    this->MakeSlotAvailable(&this->fileNumberMinSlot);

    this->fileNumberMaxSlot << new core::param::IntParam(this->fileNumMax, 0);
    this->MakeSlotAvailable(&this->fileNumberMaxSlot);

    this->fileNumberStepSlot << new core::param::IntParam(this->fileNumStep, 1);
    this->MakeSlotAvailable(&this->fileNumberStepSlot);

    this->fileNameSlotNameSlot << new core::param::StringParam("");
    this->fileNameSlotNameSlot.SetUpdateCallback(&CSVFileSequence::onFileNameSlotNameChanged);
    this->MakeSlotAvailable(&this->fileNameSlotNameSlot);

    this->useClipBoxAsBBox << new core::param::BoolParam(false);
    this->MakeSlotAvailable(&this->useClipBoxAsBBox);

    this->outDataSlot.SetCallback(table::TableDataCall::ClassName(), "GetData", &CSVFileSequence::getDataCallback);
    this->outDataSlot.SetCallback(table::TableDataCall::ClassName(), "GetHash", &CSVFileSequence::getExtentCallback);
    this->MakeSlotAvailable(&this->outDataSlot);

    this->inDataSlot.SetCompatibleCall<table::TableDataCallDescription>();
    this->MakeSlotAvailable(&this->inDataSlot);

    //core::CallDescriptionManager::DescriptionIterator iter(core::CallDescriptionManager::Instance()->GetIterator());
    //const core::CallDescription *cd = NULL;
    //while ((cd = this->moveToNextCompatibleCall(iter)) != NULL) {
    //    this->outDataSlot.SetCallback(cd->ClassName(), "GetData", &CSVFileSequence::getDataCallback);
    //    this->outDataSlot.SetCallback(cd->ClassName(), "GetExtent", &CSVFileSequence::getExtentCallback);
    //    this->inDataSlot.SetCompatibleCall(*cd);
    //}
    //this->MakeSlotAvailable(&this->outDataSlot);
    //this->MakeSlotAvailable(&this->inDataSlot);
}


/*
 * CSVFileSequence::~CSVFileSequence
 */
datatools::CSVFileSequence::~CSVFileSequence() {
    this->Release(); // implicitly calls 'release'
}


/*
 * CSVFileSequence::create
 */
bool datatools::CSVFileSequence::create() {

    return true;
}


/*
 * CSVFileSequence::release
 */
void datatools::CSVFileSequence::release() {}


/*
 * datatools::CSVFileSequence::IsCallDescriptionCompatible
 */
bool datatools::CSVFileSequence::IsCallDescriptionCompatible(core::factories::CallDescription::ptr desc) {
    return (desc->FunctionCount() == 2) && vislib::StringA("GetData").Equals(desc->FunctionName(0), false) &&
           vislib::StringA("GetExtent").Equals(desc->FunctionName(1), false);
}


/*
 * CSVFileSequence::getDataCallback
 */
bool datatools::CSVFileSequence::getDataCallback(core::Call& caller) {

    this->checkParameters();
    this->assertData();

    table::TableDataCall* pgdc = dynamic_cast<table::TableDataCall*>(&caller);
    if (pgdc == NULL)
        return false;

    table::TableDataCall* ggdc = this->inDataSlot.CallAs<table::TableDataCall>();
    if (ggdc == NULL)
        return false;

    core::param::ParamSlot* fnSlot = this->findFileNameSlot();
    if (fnSlot == NULL)
        return false;

    if (this->frameCnt == 0) {
        // I don't like this, but, argl
        return true;

    } else {
        vislib::TString filename;

        unsigned int frameID = pgdc->GetFrameID();
        unsigned int idx = this->fileNumMin + this->fileNumStep * frameID;
        filename.Format(this->fileNameTemplate, idx);
        fnSlot->Parameter()->ParseValue(filename.PeekBuffer());

        if (this->lastIdxRequested != pgdc->GetFrameID()) {
            this->lastIdxRequested = pgdc->GetFrameID();
            this->datahash++;
        }

        // TODO: WTF?????? all modules behind me think time is only ever 0 and "does not move"
        //
        // YES. This is the way it should be. Because this module is ment to "concatenate" files containing single frames only.
        // A better module could be written and should be written, but this is the way it is.
        //
        ggdc->SetFrameID(0);
        //ggdc->SetFrameID(frameID);
        if (!(*ggdc)(0)) {
            return false; // unable to get data
        }
        auto const& pluginsRes = frontend_resources.get<frontend_resources::PluginsResource>();
        pluginsRes.all_call_descriptions.AssignmentCrowbar(pgdc, ggdc);
        ggdc->SetUnlocker(nullptr, false);

        pgdc->SetFrameID(frameID);
        pgdc->SetDataHash(this->datahash);
        pgdc->SetFrameCount(this->frameCnt);
    }

    return true;
}


/*
 * CSVFileSequence::getExtentCallback
 */
bool datatools::CSVFileSequence::getExtentCallback(core::Call& caller) {

    this->checkParameters();
    this->assertData();

    table::TableDataCall* pgdc = dynamic_cast<table::TableDataCall*>(&caller);
    if (pgdc == NULL)
        return false;

    table::TableDataCall* ggdc = this->inDataSlot.CallAs<table::TableDataCall>();
    if (ggdc == NULL)
        return false;

    core::param::ParamSlot* fnSlot = this->findFileNameSlot();
    if (fnSlot == NULL)
        return false;

    if (this->frameCnt == 0) {
        pgdc->SetDataHash(0);
        pgdc->SetFrameID(0);
        pgdc->SetFrameCount(1);

    } else {
        vislib::TString filename;

        unsigned int idx = this->fileNumMin + this->fileNumStep * pgdc->GetFrameID();
        filename.Format(this->fileNameTemplate, idx);
        fnSlot->Parameter()->ParseValue(filename.PeekBuffer());

        if (this->lastIdxRequested != pgdc->GetFrameID()) {
            this->lastIdxRequested = pgdc->GetFrameID();
            this->datahash++;
        }

        ggdc->SetFrameID(pgdc->GetFrameID());
        if (!(*ggdc)(1)) {
            return false; // unable to get data
        }

        pgdc->SetFrameID(pgdc->GetFrameID());
        pgdc->SetFrameCount(this->frameCnt);
        pgdc->SetDataHash(this->datahash);
    }

    return true;
}


/*
 * CSVFileSequence::checkParameters
 */
void datatools::CSVFileSequence::checkParameters() {
    if (this->fileNumberMinSlot.IsDirty()) {
        this->fileNumberMinSlot.ResetDirty();
        this->fileNumMin = static_cast<unsigned int>(
            vislib::math::Max(0, this->fileNumberMinSlot.Param<core::param::IntParam>()->Value()));
        this->needDataUpdate = true;
    }
    if (this->fileNumberMaxSlot.IsDirty()) {
        this->fileNumberMaxSlot.ResetDirty();
        this->fileNumMax = static_cast<unsigned int>(
            vislib::math::Max(0, this->fileNumberMaxSlot.Param<core::param::IntParam>()->Value()));
        this->needDataUpdate = true;
    }
    if (this->fileNumberStepSlot.IsDirty()) {
        this->fileNumberStepSlot.ResetDirty();
        this->fileNumStep = static_cast<unsigned int>(
            vislib::math::Max(1, this->fileNumberStepSlot.Param<core::param::IntParam>()->Value()));
        this->needDataUpdate = true;
    }

    if (this->fileNumMax < this->fileNumMin) {
        unsigned int i = this->fileNumMax;
        this->fileNumMax = this->fileNumMin;
        this->fileNumMin = i;
        this->fileNumberMinSlot.Param<core::param::IntParam>()->SetValue(this->fileNumMin);
        this->fileNumberMaxSlot.Param<core::param::IntParam>()->SetValue(this->fileNumMax);
    }
}


/*
 * CSVFileSequence::onFileNameTemplateChanged
 */
bool datatools::CSVFileSequence::onFileNameTemplateChanged(core::param::ParamSlot& slot) {
    using megamol::core::utility::log::Log;
    ASSERT(&slot == &this->fileNameTemplateSlot);
    // D:\data\Kohler\nial\nialout50_*5{0..599+2}*.crist
    //  Syntax: *[[DIG][{MIN..MAX[+STEP]}]*]
    // Currently MIN, MAX, and STEP work globally!!!
    // Currently only ONE '*'-Sequence is allowed!!!
    const vislib::TString& val = this->fileNameTemplateSlot.Param<core::param::StringParam>()->Value().c_str();
    vislib::TString fnt;
    unsigned int len = val.Length();
    unsigned int state = 0;
    unsigned int digs = 0;
    unsigned int minVal = 0;
    bool minSet = false;
    unsigned int maxVal = 0;
    bool maxSet = false;
    unsigned int stepVal = 0;
    bool stepSet = false;
    const char* errMsg;
    for (unsigned int i = 0; i < len; i++) {
        errMsg = NULL;
        switch (state) {
        case 0:
            if (val[i] == _T('*')) {
                state = 1;
            } else {
                fnt += val[i];
            }
            break;
        case 1:
            if ((val[i] >= _T('0')) && (val[i] <= _T('9'))) {
                digs = static_cast<unsigned int>(val[i] - _T('0'));
                state = 2;
            } else if (val[i] == _T('*')) {
                fnt += _T("%u");
                state = 0;
            } else if (val[i] == _T('{')) {
                state = 3;
            } else {
                fnt += _T("%u");
                fnt += val[i];
                state = 0;
            }
            break;
        case 2:
            if ((val[i] >= _T('0')) && (val[i] <= _T('9'))) {
                digs = digs * 10 + static_cast<unsigned int>(val[i] - _T('0'));
            } else if (val[i] == _T('*')) {
                vislib::TString s;
                if (digs == 0) {
                    s = _T("%u");
                } else {
                    s.Format(_T("%%.%uu"), digs);
                }
                fnt += s;
                state = 0;
            } else if (val[i] == _T('{')) {
                state = 3;
            } else {
                errMsg = "Unexpected character. Expected 'DIGIT', '{', or '*'";
            }
            break;
        case 3:
            if ((val[i] >= _T('0')) && (val[i] <= _T('9'))) {
                minVal = static_cast<unsigned int>(val[i] - _T('0'));
                state = 4;
            } else {
                errMsg = "Unexpected character. Expected 'DIGIT'";
            }
            break;
        case 4:
            if ((val[i] >= _T('0')) && (val[i] <= _T('9'))) {
                minVal = minVal * 10 + static_cast<unsigned int>(val[i] - _T('0'));
            } else if (val[i] == _T('.')) {
                state = 5;
            } else {
                errMsg = "Unexpected character. Expected 'DIGIT' or '.'";
            }
            break;
        case 5:
            if (val[i] == _T('.')) {
                state = 6;
            } else {
                errMsg = "Unexpected character. Expected '.'";
            }
            break;
        case 6:
            if ((val[i] >= _T('0')) && (val[i] <= _T('9'))) {
                maxVal = static_cast<unsigned int>(val[i] - _T('0'));
                state = 7;
            } else {
                errMsg = "Unexpected character. Expected 'DIGIT'";
            }
            break;
        case 7:
            if ((val[i] >= _T('0')) && (val[i] <= _T('9'))) {
                maxVal = maxVal * 10 + static_cast<unsigned int>(val[i] - _T('0'));
            } else if (val[i] == _T('}')) {
                minSet = maxSet = true;
                state = 8;
            } else if (val[i] == _T('+')) {
                state = 9;
            } else {
                errMsg = "Unexpected character. Expected 'DIGIT', '+', or '}'";
            }
            break;
        case 8:
            if (val[i] == _T('*')) {
                vislib::TString s;
                if (digs == 0) {
                    s = _T("%u");
                } else {
                    s.Format(_T("%%.%uu"), digs);
                }
                fnt += s;
                state = 0;
            } else {
                errMsg = "Unexpected character. Expected '*'";
            }
            break;
        case 9:
            if ((val[i] >= _T('0')) && (val[i] <= _T('9'))) {
                stepVal = static_cast<unsigned int>(val[i] - _T('0'));
                state = 10;
            } else {
                errMsg = "Unexpected character. Expected 'DIGIT'";
            }
            break;
        case 10:
            if ((val[i] >= _T('0')) && (val[i] <= _T('9'))) {
                stepVal = stepVal * 10 + static_cast<unsigned int>(val[i] - _T('0'));
            } else if (val[i] == _T('}')) {
                stepSet = minSet = maxSet = true;
                state = 8;
            } else {
                errMsg = "Unexpected character. Expected 'DIGIT' or '}'";
            }
            break;
        default:
            errMsg = "Internal state error";
            break;
        }
        if (errMsg != NULL) {
            Log::DefaultLog.WriteError("Parser Error at character %u: %s", i, errMsg);
            state = 0;
        }
    }

    if (minSet)
        this->fileNumberMinSlot.Param<core::param::IntParam>()->SetValue(minVal);
    if (maxSet)
        this->fileNumberMaxSlot.Param<core::param::IntParam>()->SetValue(maxVal);
    if (stepSet)
        this->fileNumberStepSlot.Param<core::param::IntParam>()->SetValue(stepVal);

    this->fileNameTemplate = fnt;
    Log::DefaultLog.WriteInfo(
        "Parsed file name template to \"%s\"", vislib::StringA(this->fileNameTemplate).PeekBuffer());
    this->needDataUpdate = true;
    return true;
}


/*
 * CSVFileSequence::onFileNameSlotNameChanged
 */
bool datatools::CSVFileSequence::onFileNameSlotNameChanged(core::param::ParamSlot& slot) {
    ASSERT(&slot == &this->fileNameSlotNameSlot);
    this->ModuleGraphLock().LockExclusive();
    core::param::StringParam* P = this->fileNameSlotNameSlot.Param<core::param::StringParam>();
    if ((P != NULL) && (this->findFileNameSlot() == NULL)) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "Unable to connect to file name parameter slot \"%s\". Parameter resetted.", P->Value().c_str());
        P->SetValue("", false);
    }
    this->ModuleGraphLock().UnlockExclusive();
    this->needDataUpdate = true;
    return true;
}


/*
 * CSVFileSequence::findFileNameSlot
 */
core::param::ParamSlot* datatools::CSVFileSequence::findFileNameSlot() {
    core::param::StringParam* P = this->fileNameSlotNameSlot.Param<core::param::StringParam>();

    // target slot has name
    if (!P)
        return nullptr;

    // there is a target module connected via call
    if (this->inDataSlot.GetStatus() != megamol::core::AbstractSlot::STATUS_CONNECTED)
        return nullptr;

    core::param::ParamSlot* slot =
        dynamic_cast<core::param::ParamSlot*>(AbstractNamedObjectContainer::dynamic_pointer_cast(
            this->inDataSlot
                .CallAs<megamol::core::Call>() // connected Call
                ->PeekCalleeSlotNoConst()      // slot of target Module
                ->Parent()                     // target Module
                ->shared_from_this())          // cast to AbstractNamedObjectContainer
                                                  // TODO: find slot by full name, not just by slot name
                                                  ->FindChild(P->Value().c_str()) // Find target Slot by name in Module
                                                  .get()                          // get value of smart ptr
        );

    return slot;
}


/*
 * CSVFileSequence::assertData
 */
void datatools::CSVFileSequence::assertData() {
    using megamol::core::utility::log::Log;
    if (!this->needDataUpdate)
        return;
    vislib::TString filename;
    this->needDataUpdate = false;

    this->frameCnt = 0;
    this->clipbox.Set(-1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f);

    core::param::ParamSlot* fnSlot = this->findFileNameSlot();
    if (fnSlot == NULL) {
        Log::DefaultLog.WriteError("Unable to connect to file name slot");
        return;
    }

    // we want to inject our variations of the file name into the referenced file path param
    auto* fileParam = fnSlot->Param<core::param::FilePathParam>();
    if (fileParam == NULL) {
        Log::DefaultLog.WriteMsg(Log::log_level::error, "file name slot is not a FilePathParam");
        return;
    }
    auto curent_file_param_value = fileParam->ValueString();

    table::TableDataCall* gdc = this->inDataSlot.CallAs<table::TableDataCall>();
    if (gdc == NULL) {
        Log::DefaultLog.WriteError("Unable to get input data call");
        return;
    }

    // count really available frames
    for (unsigned int i = this->fileNumMin; i <= this->fileNumMax; i += this->fileNumStep) {
        filename.Format(this->fileNameTemplate, i);

        // we need to use the file path param to check for existence of the file
        fileParam->ParseValue(filename.PeekBuffer());

        if (std::filesystem::exists(fileParam->Value())) {
            this->frameCnt++;
        } else {
            break;
        }
    }
    fileParam->ParseValue(curent_file_param_value);

    if (this->frameCnt == 0) {
        Log::DefaultLog.WriteError("CSVFileSequence: No data files found");
        return;
    }


    // collect heuristic approach for clipping box
    filename.Format(this->fileNameTemplate, this->fileNumMin);
    fnSlot->Parameter()->ParseValue(filename.PeekBuffer());
    gdc->SetFrameID(0);
    if (!(*gdc)(1)) {
        this->frameCnt = 0;
        Log::DefaultLog.WriteError("CSVFileSequence: Unable to clipping box of file %u (#1)", this->fileNumMin);
        return;
    }

    this->datahash++;
}
