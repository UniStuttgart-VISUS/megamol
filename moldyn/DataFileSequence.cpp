/*
 * DataFileSequence.cpp
 *
 * Copyright (C) 2010 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "DataFileSequence.h"
#include "AbstractGetData3DCall.h"
#include "CallDescriptionManager.h"
#include "CallDescription.h"
#include "param/FilePathParam.h"
#include "param/StringParam.h"
#include "param/IntParam.h"
#include "vislib/File.h"
#include "vislib/Log.h"
//#include "vislib/MemmappedFile.h"
//#include "vislib/SingleLinkedList.h"
#include "vislib/String.h"
//#include "vislib/sysfunctions.h"
//#include "vislib/VersionNumber.h"

using namespace megamol::core;


/*
 * moldyn::DataFileSequence::DataFileSequence
 */
moldyn::DataFileSequence::DataFileSequence(void) : Module(),
        fileNameTemplateSlot("fileNameTemplate", "The file name template"),
        fileNumberMinSlot("fileNumberMin", "Slot for the minimum file number"),
        fileNumberMaxSlot("fileNumberMax", "Slot for the maximum file number"),
        fileNumberStepSlot("fileNumberStep", "Slot for the file number increase step"),
        fileNameSlotNameSlot("fileNameSlotName", "The name of the data source file name parameter slot"),
        outDataSlot("outData", "The slot for publishing data to the writer"),
        inDataSlot("inData", "The slot for requesting data from the source"),
        clipbox(-1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f), datahash(0),
        fileNameTemplate(_T("")), fileNumMin(0), fileNumMax(0), fileNumStep(1),
        needDataUpdate(true), frameCnt(1), lastIdxRequested(0) {

    this->fileNameTemplateSlot << new param::StringParam(this->fileNameTemplate);
    this->fileNameTemplateSlot.SetUpdateCallback(&DataFileSequence::onFileNameTemplateChanged);
    this->MakeSlotAvailable(&this->fileNameTemplateSlot);

    this->fileNumberMinSlot << new param::IntParam(this->fileNumMin, 0);
    this->MakeSlotAvailable(&this->fileNumberMinSlot);

    this->fileNumberMaxSlot << new param::IntParam(this->fileNumMax, 0);
    this->MakeSlotAvailable(&this->fileNumberMaxSlot);

    this->fileNumberStepSlot << new param::IntParam(this->fileNumStep, 1);
    this->MakeSlotAvailable(&this->fileNumberStepSlot);

    this->fileNameSlotNameSlot << new param::StringParam("");
    this->fileNameSlotNameSlot.SetUpdateCallback(&DataFileSequence::onFileNameSlotNameChanged);
    this->MakeSlotAvailable(&this->fileNameSlotNameSlot);

    CallDescriptionManager::DescriptionIterator iter(CallDescriptionManager::Instance()->GetIterator());
    const CallDescription *cd = NULL;
    while ((cd = this->moveToNextCompatibleCall(iter)) != NULL) {
        this->outDataSlot.SetCallback(cd->ClassName(), "GetData", &DataFileSequence::getDataCallback);
        this->outDataSlot.SetCallback(cd->ClassName(), "GetExtent", &DataFileSequence::getExtentCallback);
        this->inDataSlot.SetCompatibleCall(*cd);
    }

    this->MakeSlotAvailable(&this->outDataSlot);
    this->MakeSlotAvailable(&this->inDataSlot);
}


/*
 * moldyn::DataFileSequence::~DataFileSequence
 */
moldyn::DataFileSequence::~DataFileSequence(void) {
    this->Release(); // implicitly calls 'release'
}


/*
 * moldyn::DataFileSequence::create
 */
bool moldyn::DataFileSequence::create(void) {
    return true;
}


/*
 * moldyn::DataFileSequence::release
 */
void moldyn::DataFileSequence::release(void) {
}


/*
 * moldyn::DataFileSequence::moveToNextCompatibleCall
 */
const CallDescription* moldyn::DataFileSequence::moveToNextCompatibleCall(
        CallDescriptionManager::DescriptionIterator &iterator) const {
    while (iterator.HasNext()) {
        const CallDescription *d = iterator.Next();
        if ((d->FunctionCount() == 2)
                && vislib::StringA("GetData").Equals(d->FunctionName(0), false)
                && vislib::StringA("GetExtent").Equals(d->FunctionName(1), false)) {
            return d;
        }
    }
    return NULL;
}


/*
 * moldyn::DataFileSequence::getDataCallback
 */
bool moldyn::DataFileSequence::getDataCallback(Call& caller) {
    if (!this->checkConnections(&caller)) return false;
    this->checkParameters();
    this->assertData();

    AbstractGetData3DCall *pgdc = dynamic_cast<AbstractGetData3DCall*>(&caller);
    if (pgdc == NULL) return false;

    AbstractGetData3DCall *ggdc = this->inDataSlot.CallAs<AbstractGetData3DCall>();
    if (ggdc == NULL) return false;

    param::ParamSlot *fnSlot = this->findFileNameSlot();
    if (fnSlot == NULL) return false;

    if (this->frameCnt == 0) {
        // I don't like this, but, argl
        return true;

    } else {
        vislib::TString filename;

        unsigned int frameID = pgdc->FrameID();
        unsigned int idx = this->fileNumMin + this->fileNumStep * frameID;
        filename.Format(this->fileNameTemplate, idx);
        fnSlot->Parameter()->ParseValue(filename);

        if (this->lastIdxRequested != pgdc->FrameID()) {
            this->lastIdxRequested = pgdc->FrameID();
            this->datahash++;
        }

        ggdc->SetFrameID(0);
        if (!(*ggdc)(0)) {
            return false; // unable to get data
        }
        CallDescriptionManager::Instance()->AssignmentCrowbar(pgdc, ggdc);

        pgdc->SetFrameID(frameID);
        pgdc->SetDataHash(this->datahash);
    }

    return true;
}


/*
 * moldyn::DataFileSequence::getExtentCallback
 */
bool moldyn::DataFileSequence::getExtentCallback(Call& caller) {
    if (!this->checkConnections(&caller)) return false;
    this->checkParameters();
    this->assertData();

    AbstractGetData3DCall *pgdc = dynamic_cast<AbstractGetData3DCall*>(&caller);
    if (pgdc == NULL) return false;

    AbstractGetData3DCall *ggdc = this->inDataSlot.CallAs<AbstractGetData3DCall>();
    if (ggdc == NULL) return false;

    param::ParamSlot *fnSlot = this->findFileNameSlot();
    if (fnSlot == NULL) return false;

    pgdc->AccessBoundingBoxes().Clear();
    if (this->frameCnt == 0) {
        pgdc->SetDataHash(0);
        pgdc->SetFrameID(0);
        pgdc->SetFrameCount(1);

    } else {
        vislib::TString filename;

        unsigned int idx = this->fileNumMin + this->fileNumStep * pgdc->FrameID();
        filename.Format(this->fileNameTemplate, idx);
        fnSlot->Parameter()->ParseValue(filename);

        if (this->lastIdxRequested != pgdc->FrameID()) {
            this->lastIdxRequested = pgdc->FrameID();
            this->datahash++;
        }

        ggdc->SetFrameID(0);
        if (!(*ggdc)(1)) {
            return false; // unable to get data
        }

        pgdc->AccessBoundingBoxes().SetObjectSpaceBBox(ggdc->AccessBoundingBoxes().ObjectSpaceBBox());
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
bool moldyn::DataFileSequence::checkConnections(Call *outCall) {
    if (this->inDataSlot.GetStatus() != AbstractSlot::STATUS_CONNECTED) return false;
    if (this->outDataSlot.GetStatus() != AbstractSlot::STATUS_CONNECTED) return false;
    Call *inCall = this->inDataSlot.CallAs<Call>();
    if ((inCall == NULL) || (outCall == NULL)) return false;
    CallDescriptionManager::DescriptionIterator iter(CallDescriptionManager::Instance()->GetIterator());
    const CallDescription *cd = NULL;
    while ((cd = this->moveToNextCompatibleCall(iter)) != NULL) {
        if (cd->IsDescribing(inCall) && cd->IsDescribing(outCall)) return true;
        // both slot connected with similar calls
    }
    return false;
}


/*
 * moldyn::DataFileSequence::checkParameters
 */
void moldyn::DataFileSequence::checkParameters(void) {
    if (this->fileNumberMinSlot.IsDirty()) {
        this->fileNumberMinSlot.ResetDirty();
        this->fileNumMin = static_cast<unsigned int>(vislib::math::Max(0,
            this->fileNumberMinSlot.Param<param::IntParam>()->Value()));
        this->needDataUpdate = true;
    }
    if (this->fileNumberMaxSlot.IsDirty()) {
        this->fileNumberMaxSlot.ResetDirty();
        this->fileNumMax = static_cast<unsigned int>(vislib::math::Max(0,
            this->fileNumberMaxSlot.Param<param::IntParam>()->Value()));
        this->needDataUpdate = true;
    }
    if (this->fileNumberStepSlot.IsDirty()) {
        this->fileNumberStepSlot.ResetDirty();
        this->fileNumStep = static_cast<unsigned int>(vislib::math::Max(1,
            this->fileNumberStepSlot.Param<param::IntParam>()->Value()));
        this->needDataUpdate = true;
    }

    if (this->fileNumMax < this->fileNumMin) {
        unsigned int i = this->fileNumMax;
        this->fileNumMax = this->fileNumMin;
        this->fileNumMin = i;
        this->fileNumberMinSlot.Param<param::IntParam>()->SetValue(this->fileNumMin);
        this->fileNumberMaxSlot.Param<param::IntParam>()->SetValue(this->fileNumMax);
    }
}


/*
 * moldyn::DataFileSequence::onFileNameTemplateChanged
 */
bool moldyn::DataFileSequence::onFileNameTemplateChanged(param::ParamSlot& slot) {
    using vislib::sys::Log;
    ASSERT(&slot == &this->fileNameTemplateSlot);
    // D:\data\Kohler\nial\nialout50_*5{0..599+2}*.crist
    //  Syntax: *[[DIG][{MIN..MAX[+STEP]}]*]
    // Currently MIN, MAX, and STEP work globally!!!
    // Currently only ONE '*'-Sequence is allowed!!!
    const vislib::TString &val = this->fileNameTemplateSlot.Param<param::StringParam>()->Value();
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
    const char *errMsg;
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
            Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                "Parser Error at character %u: %s", i, errMsg);
            state = 0;
        }
    }

    if (minSet) this->fileNumberMinSlot.Param<param::IntParam>()->SetValue(minVal);
    if (maxSet) this->fileNumberMaxSlot.Param<param::IntParam>()->SetValue(maxVal);
    if (stepSet) this->fileNumberStepSlot.Param<param::IntParam>()->SetValue(stepVal);

    this->fileNameTemplate = fnt;
    Log::DefaultLog.WriteMsg(Log::LEVEL_INFO,
        "Parsed file name template to \"%s\"",
        vislib::StringA(this->fileNameTemplate).PeekBuffer());
    this->needDataUpdate = true;
    return true;
}


/*
 * moldyn::DataFileSequence::onFileNameSlotNameChanged
 */
bool moldyn::DataFileSequence::onFileNameSlotNameChanged(param::ParamSlot& slot) {
    ASSERT(&slot == &this->fileNameSlotNameSlot);
    this->LockModuleGraph(false);
    param::StringParam *P = this->fileNameSlotNameSlot.Param<param::StringParam>();
    if ((P != NULL) && (this->findFileNameSlot() == NULL)) {
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
            "Unable to connect to file name parameter slot \"%s\". Parameter resetted.",
            vislib::StringA(P->Value()).PeekBuffer());
        P->SetValue("", false);
    }
    this->UnlockModuleGraph();
    this->needDataUpdate = true;
    return true;
}


/*
 * moldyn::DataFileSequence::findFileNameSlot
 */
param::ParamSlot *moldyn::DataFileSequence::findFileNameSlot(void) {
    param::StringParam *P = this->fileNameSlotNameSlot.Param<param::StringParam>();
    if (P != NULL) {
        AbstractNamedObjectContainer *anoc = this;
        while (anoc != NULL) {
            param::ParamSlot *slot = dynamic_cast<param::ParamSlot*>(anoc->FindNamedObject(
                vislib::StringA(P->Value()).PeekBuffer()));
            if (slot != NULL) {
                if ((slot->Param<param::FilePathParam>() != NULL)
                        || (slot->Param<param::StringParam>() != NULL)) {
                    // everything is fine
                    return slot;
                }
            }
            anoc = dynamic_cast<AbstractNamedObjectContainer *>(anoc->Parent());
        }
    }
    return NULL;
}


/*
 * moldyn::DataFileSequence::assertData
 */
void moldyn::DataFileSequence::assertData(void) {
    using vislib::sys::Log;
    if (!this->needDataUpdate) return;
    vislib::TString filename;
    this->needDataUpdate = false;

    this->frameCnt = 0;
    this->clipbox.Set(-1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f);

    param::ParamSlot *fnSlot = this->findFileNameSlot();
    if (fnSlot == NULL) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Unable to connect to file name slot");
        return;
    }

    AbstractGetData3DCall *gdc = this->inDataSlot.CallAs<AbstractGetData3DCall>();
    if (gdc == NULL) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Unable to get input data call");
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
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
            "DataFileSequence: No data files found");
        return;
    }

    // collect heuristic approach for clipping box
    filename.Format(this->fileNameTemplate, this->fileNumMin);
    fnSlot->Parameter()->ParseValue(filename);
    gdc->SetFrameID(0);
    if (!(*gdc)(1)) {
        this->frameCnt = 0;
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "DataFileSequence: Unable to clipping box of file %u (#1)", this->fileNumMin);
        return;
    }
    this->clipbox = gdc->AccessBoundingBoxes().ClipBox();

    if (this->frameCnt > 1) {
        unsigned int idx = this->fileNumMin + this->fileNumStep * (this->frameCnt - 1);
        filename.Format(this->fileNameTemplate, idx);
        fnSlot->Parameter()->ParseValue(filename);
        gdc->SetFrameID(0);
        if (!(*gdc)(1)) {
            this->frameCnt = 0;
            Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "DataFileSequence: Unable to clipping box of file %u (#2)", idx);
            return;
        }
        this->clipbox.Union(gdc->AccessBoundingBoxes().ClipBox());
        if (this->frameCnt > 2) {
            idx = this->fileNumMin + this->fileNumStep * ((this->frameCnt - 1) / 2);
            filename.Format(this->fileNameTemplate, idx);
            fnSlot->Parameter()->ParseValue(filename);
            gdc->SetFrameID(0);
            if (!(*gdc)(1)) {
                this->frameCnt = 0;
                Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "DataFileSequence: Unable to clipping box of file %u (#3)", idx);
                return;
            }
            this->clipbox.Union(gdc->AccessBoundingBoxes().ClipBox());
        }
    }

    this->datahash++;

}
