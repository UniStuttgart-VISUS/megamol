/*
 * DataFileSequence.cpp
 *
 * Copyright (C) 2010 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#include "DataFileSequence.h"
#include "mmcore/CoreInstance.h"
#include "mmcore/factories/CallDescriptionManager.h"
#include "mmcore/param/BoolParam.h"
#include "mmcore/param/FilePathParam.h"
#include "mmcore/param/IntParam.h"
#include "mmcore/param/StringParam.h"
#include "mmcore/utility/log/Log.h"
#include "mmstd/data/AbstractGetData3DCall.h"
#include "vislib/String.h"
#include "vislib/sys/File.h"

using namespace megamol;


/*
 * moldyn::DataFileSequence::DataFileSequence
 */
datatools::DataFileSequence::DataFileSequence(void)
        : core::Module()
        , fileNameTemplateSlot("fileNameTemplate", "The file name template"
                                                   " example: D:\\data\\Kohler\\nial\\nialout50_*5{0..599+2}*.crist "
                                                   " Syntax: *[[DIG][{MIN..MAX[+STEP]}]*] "
                                                   " Currently MIN, MAX, and STEP work globally!!! "
                                                   " Currently only ONE '*'-Sequence is allowed!!!")
        , fileNumberMinSlot("fileNumberMin", "Slot for the minimum file number")
        , fileNumberMaxSlot("fileNumberMax", "Slot for the maximum file number")
        , fileNumberStepSlot("fileNumberStep", "Slot for the file number increase step")
        , fileNameSlotNameSlot("fileNameSlotName", "The name of the data source file name parameter slot")
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
    this->fileNameTemplateSlot.SetUpdateCallback(&DataFileSequence::onFileNameTemplateChanged);
    this->MakeSlotAvailable(&this->fileNameTemplateSlot);

    this->fileNumberMinSlot << new core::param::IntParam(this->fileNumMin, 0);
    this->MakeSlotAvailable(&this->fileNumberMinSlot);

    this->fileNumberMaxSlot << new core::param::IntParam(this->fileNumMax, 0);
    this->MakeSlotAvailable(&this->fileNumberMaxSlot);

    this->fileNumberStepSlot << new core::param::IntParam(this->fileNumStep, 1);
    this->MakeSlotAvailable(&this->fileNumberStepSlot);

    this->fileNameSlotNameSlot << new core::param::StringParam("");
    this->fileNameSlotNameSlot.SetUpdateCallback(&DataFileSequence::onFileNameSlotNameChanged);
    this->MakeSlotAvailable(&this->fileNameSlotNameSlot);

    this->useClipBoxAsBBox << new core::param::BoolParam(false);
    this->MakeSlotAvailable(&this->useClipBoxAsBBox);

    //core::CallDescriptionManager::DescriptionIterator iter(core::CallDescriptionManager::Instance()->GetIterator());
    //const core::CallDescription *cd = NULL;
    //while ((cd = this->moveToNextCompatibleCall(iter)) != NULL) {
    //    this->outDataSlot.SetCallback(cd->ClassName(), "GetData", &DataFileSequence::getDataCallback);
    //    this->outDataSlot.SetCallback(cd->ClassName(), "GetExtent", &DataFileSequence::getExtentCallback);
    //    this->inDataSlot.SetCompatibleCall(*cd);
    //}
    //this->MakeSlotAvailable(&this->outDataSlot);
    //this->MakeSlotAvailable(&this->inDataSlot);
}


/*
 * moldyn::DataFileSequence::~DataFileSequence
 */
datatools::DataFileSequence::~DataFileSequence(void) {
    this->Release(); // implicitly calls 'release'
}


/*
 * moldyn::DataFileSequence::create
 */
bool datatools::DataFileSequence::create(void) {
    for (auto cd : this->GetCoreInstance()->GetCallDescriptionManager()) {
        if (IsCallDescriptionCompatible(cd)) {
            this->outDataSlot.SetCallback(cd->ClassName(), "GetData", &DataFileSequence::getDataCallback);
            this->outDataSlot.SetCallback(cd->ClassName(), "GetExtent", &DataFileSequence::getExtentCallback);
            this->inDataSlot.SetCompatibleCall(cd);
        }
    }
    this->MakeSlotAvailable(&this->outDataSlot);
    this->MakeSlotAvailable(&this->inDataSlot);
    return true;
}


/*
 * moldyn::DataFileSequence::release
 */
void datatools::DataFileSequence::release(void) {}


/*
 * datatools::DataFileSequence::IsCallDescriptionCompatible
 */
bool datatools::DataFileSequence::IsCallDescriptionCompatible(core::factories::CallDescription::ptr desc) {
    return (desc->FunctionCount() == 2) && vislib::StringA("GetData").Equals(desc->FunctionName(0), false) &&
           vislib::StringA("GetExtent").Equals(desc->FunctionName(1), false);
}


/*
 * moldyn::DataFileSequence::getDataCallback
 */
bool datatools::DataFileSequence::getDataCallback(core::Call& caller) {
    if (!this->checkConnections(&caller))
        return false;
    this->checkParameters();
    this->assertData();

    core::AbstractGetData3DCall* pgdc = dynamic_cast<core::AbstractGetData3DCall*>(&caller);
    if (pgdc == NULL)
        return false;

    core::AbstractGetData3DCall* ggdc = this->inDataSlot.CallAs<core::AbstractGetData3DCall>();
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

        unsigned int frameID = pgdc->FrameID();
        unsigned int idx = this->fileNumMin + this->fileNumStep * frameID;
        filename.Format(this->fileNameTemplate, idx);
        fnSlot->Parameter()->ParseValue(filename.PeekBuffer());

        if (this->lastIdxRequested != pgdc->FrameID()) {
            this->lastIdxRequested = pgdc->FrameID();
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
        this->GetCoreInstance()->GetCallDescriptionManager().AssignmentCrowbar(pgdc, ggdc);
        ggdc->SetUnlocker(nullptr, false);

        pgdc->SetFrameID(frameID, true);
        pgdc->SetDataHash(this->datahash);
    }

    return true;
}


/*
 * moldyn::DataFileSequence::getExtentCallback
 */
bool datatools::DataFileSequence::getExtentCallback(core::Call& caller) {

    if (!this->checkConnections(&caller))
        return false;

    this->checkParameters();
    this->assertData();

    core::AbstractGetData3DCall* pgdc = dynamic_cast<core::AbstractGetData3DCall*>(&caller);
    if (pgdc == NULL)
        return false;

    core::AbstractGetData3DCall* ggdc = this->inDataSlot.CallAs<core::AbstractGetData3DCall>();
    if (ggdc == NULL)
        return false;

    core::param::ParamSlot* fnSlot = this->findFileNameSlot();
    if (fnSlot == NULL)
        return false;


    pgdc->AccessBoundingBoxes().Clear();
    if (this->frameCnt == 0) {
        pgdc->SetDataHash(0);
        pgdc->SetFrameID(0);
        pgdc->SetFrameCount(1);

    } else {
        vislib::TString filename;

        unsigned int idx = this->fileNumMin + this->fileNumStep * pgdc->FrameID();
        filename.Format(this->fileNameTemplate, idx);
        fnSlot->Parameter()->ParseValue(filename.PeekBuffer());

        if (this->lastIdxRequested != pgdc->FrameID()) {
            this->lastIdxRequested = pgdc->FrameID();
            this->datahash++;
        }

        ggdc->SetFrameID(pgdc->FrameID(), pgdc->IsFrameForced());
        if (!(*ggdc)(1)) {
            return false; // unable to get data
        }

        if (this->useClipBoxAsBBox.Param<core::param::BoolParam>()->Value()) {
            pgdc->AccessBoundingBoxes().SetObjectSpaceBBox(this->clipbox);
        } else {
            pgdc->AccessBoundingBoxes().SetObjectSpaceBBox(ggdc->AccessBoundingBoxes().ObjectSpaceBBox());
        }
        pgdc->AccessBoundingBoxes().SetObjectSpaceClipBox(this->clipbox);
        pgdc->SetFrameID(pgdc->FrameID());
        pgdc->SetFrameCount(this->frameCnt);
        pgdc->SetDataHash(this->datahash);
    }

    return true;
}


/*
 * moldyn::DataFileSequence::checkConnections
 */
bool datatools::DataFileSequence::checkConnections(core::Call* outCall) {
    if (this->inDataSlot.GetStatus() != core::AbstractSlot::STATUS_CONNECTED)
        return false;
    if (this->outDataSlot.GetStatus() != core::AbstractSlot::STATUS_CONNECTED)
        return false;
    core::Call* inCall = this->inDataSlot.CallAs<core::Call>();
    if ((inCall == NULL) || (outCall == NULL))
        return false;
    for (auto cd : this->GetCoreInstance()->GetCallDescriptionManager()) {
        if (IsCallDescriptionCompatible(cd)) {
            if (cd->IsDescribing(inCall) && cd->IsDescribing(outCall))
                return true;
        }
    }
    return false;
}


/*
 * moldyn::DataFileSequence::checkParameters
 */
void datatools::DataFileSequence::checkParameters(void) {
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
 * moldyn::DataFileSequence::onFileNameTemplateChanged
 */
bool datatools::DataFileSequence::onFileNameTemplateChanged(core::param::ParamSlot& slot) {
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
 * moldyn::DataFileSequence::onFileNameSlotNameChanged
 */
bool datatools::DataFileSequence::onFileNameSlotNameChanged(core::param::ParamSlot& slot) {
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
 * moldyn::DataFileSequence::findFileNameSlot
 */
core::param::ParamSlot* datatools::DataFileSequence::findFileNameSlot(void) {
    core::param::StringParam* P = this->fileNameSlotNameSlot.Param<core::param::StringParam>();
    if (P != NULL) {
        AbstractNamedObjectContainer::ptr_type anoc =
            AbstractNamedObjectContainer::dynamic_pointer_cast(this->shared_from_this());
        while (anoc) {
            core::param::ParamSlot* slot =
                dynamic_cast<core::param::ParamSlot*>(anoc->FindNamedObject(P->Value().c_str()).get());
            if (slot != NULL) {
                if ((slot->Param<core::param::FilePathParam>() != NULL) ||
                    (slot->Param<core::param::StringParam>() != NULL)) {
                    // everything is fine
                    return slot;
                }
            }
            anoc = AbstractNamedObjectContainer::dynamic_pointer_cast(anoc->Parent());
        }
    }
    return NULL;
}


/*
 * moldyn::DataFileSequence::assertData
 */
void datatools::DataFileSequence::assertData(void) {
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

    core::AbstractGetData3DCall* gdc = this->inDataSlot.CallAs<core::AbstractGetData3DCall>();
    if (gdc == NULL) {
        Log::DefaultLog.WriteError("Unable to get input data call");
        return;
    }

    // count really available frames
    for (unsigned int i = this->fileNumMin; i <= this->fileNumMax; i += this->fileNumStep) {
        filename.Format(this->fileNameTemplate, i);
        if (vislib::sys::File::Exists(filename)) {
            this->frameCnt++;
        } else {
            break;
        }
    }
    if (this->frameCnt == 0) {
        Log::DefaultLog.WriteError("DataFileSequence: No data files found");
        return;
    }

    // collect heuristic approach for clipping box
    filename.Format(this->fileNameTemplate, this->fileNumMin);
    fnSlot->Parameter()->ParseValue(filename.PeekBuffer());
    gdc->SetFrameID(0);
    if (!(*gdc)(1)) {
        this->frameCnt = 0;
        Log::DefaultLog.WriteError("DataFileSequence: Unable to clipping box of file %u (#1)", this->fileNumMin);
        return;
    }
    this->clipbox = gdc->AccessBoundingBoxes().ClipBox();

    if (this->frameCnt > 1) {
        unsigned int idx = this->fileNumMin + this->fileNumStep * (this->frameCnt - 1);
        filename.Format(this->fileNameTemplate, idx);
        fnSlot->Parameter()->ParseValue(filename.PeekBuffer());
        gdc->SetFrameID(0);
        if (!(*gdc)(1)) {
            this->frameCnt = 0;
            Log::DefaultLog.WriteError("DataFileSequence: Unable to clipping box of file %u (#2)", idx);
            return;
        }
        this->clipbox.Union(gdc->AccessBoundingBoxes().ClipBox());
        if (this->frameCnt > 2) {
            idx = this->fileNumMin + this->fileNumStep * ((this->frameCnt - 1) / 2);
            filename.Format(this->fileNameTemplate, idx);
            fnSlot->Parameter()->ParseValue(filename.PeekBuffer());
            gdc->SetFrameID(0);
            if (!(*gdc)(1)) {
                this->frameCnt = 0;
                Log::DefaultLog.WriteError("DataFileSequence: Unable to clipping box of file %u (#3)", idx);
                return;
            }
            this->clipbox.Union(gdc->AccessBoundingBoxes().ClipBox());
        }
    }

    this->datahash++;
}
