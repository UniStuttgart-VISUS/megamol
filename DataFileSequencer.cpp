/*
 * DataFileSequencer.cpp
 *
 * Copyright (C) 2010 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "DataFileSequencer.h"
#include "param/ButtonParam.h"
#include "param/FilePathParam.h"
#include "param/StringParam.h"
#include "view/CallRenderView.h"
#include "vislib/File.h"
#include "vislib/Log.h"


/*
 * megamol::core::DataFileSequencer::DataFileSequencer
 */
megamol::core::DataFileSequencer::DataFileSequencer(void) : Module(),
        conSlot("connection", "Connects to a view to make this module part of the same module network."),
        filenameSlotNameSlot("filenameSlot", "String parameter identifying the filename slot to manipulate"),
        nextFileSlot("nextFile", "Button parameter to switch to the next file"),
        prevFileSlot("prevFile", "Button parameter to switch to the previous file") {

    this->conSlot.SetCompatibleCall<view::CallRenderViewDescription>();
    // TODO: Connection to a job would also be nice
    this->MakeSlotAvailable(&this->conSlot);

    this->filenameSlotNameSlot << new param::StringParam("");
    this->MakeSlotAvailable(&this->filenameSlotNameSlot);

    this->nextFileSlot << new param::ButtonParam('n');
    this->nextFileSlot.SetUpdateCallback(&DataFileSequencer::onNextFile);
    this->MakeSlotAvailable(&this->nextFileSlot);

    this->prevFileSlot << new param::ButtonParam('b');
    this->prevFileSlot.SetUpdateCallback(&DataFileSequencer::onPrevFile);
    this->MakeSlotAvailable(&this->prevFileSlot);
}


/*
 * megamol::core::DataFileSequencer::~DataFileSequencer
 */
megamol::core::DataFileSequencer::~DataFileSequencer(void) {
    this->Release();
    // Intentionally empty
}


/*
 * megamol::core::DataFileSequencer::create
 */
bool megamol::core::DataFileSequencer::create(void) {
    // Intentionally empty
    return true;
}


/*
 * megamol::core::DataFileSequencer::findFilenameSlot
 */
megamol::core::param::ParamSlot * megamol::core::DataFileSequencer::findFilenameSlot(void) {
    vislib::StringA name(this->filenameSlotNameSlot.Param<param::StringParam>()->Value());
    if (name.IsEmpty()) return NULL;

    this->LockModuleGraph(false);

    AbstractNamedObjectContainer *anoc = this;
    AbstractNamedObject *ano = NULL;
    while ((anoc != NULL) && (ano == NULL)) {
        ano = anoc->FindNamedObject(name.PeekBuffer());
        anoc = dynamic_cast<AbstractNamedObjectContainer*>(anoc->Parent());
    }

    this->UnlockModuleGraph();

    return dynamic_cast<param::ParamSlot*>(ano);
}


/*
 * megamol::core::DataFileSequencer::GetFilename
 */
bool megamol::core::DataFileSequencer::GetFilename(megamol::core::param::ParamSlot& slot, vislib::TString& outName) const {
    using vislib::sys::Log;
    param::StringParam *sp = slot.Param<param::StringParam>();
    param::FilePathParam *fpp = slot.Param<param::FilePathParam>();
    if ((sp == NULL) && (fpp == NULL)) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
            "Parameter \"%s\" is not of a compatible type.",
            vislib::StringA(this->filenameSlotNameSlot.Param<param::StringParam>()->Value()).PeekBuffer());
        return false;
    }
    outName = (sp == NULL) ? fpp->Value() : sp->Value();
    if (outName.IsEmpty()) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_WARN,
            "Value of parameter \"%s\" is empty.",
            vislib::StringA(this->filenameSlotNameSlot.Param<param::StringParam>()->Value()).PeekBuffer());
        return false;
    }
    return true;
}


/*
 * megamol::core::DataFileSequencer::GetFormat
 */
bool megamol::core::DataFileSequencer::GetFormat(vislib::TString& inoutName, int& outValue) const {
    int len = static_cast<int>(inoutName.Length());
    int p1 = -1;

    for (int pos = len -1; pos >= 0; pos--) {
        if ((inoutName[pos] >= _T('0')) && (inoutName[pos] <= _T('9'))) {
            if (p1 == -1) p1 = pos;
        } else {
            if (p1 != -1) {
                // pos + 1 .. p1 is last number sequence
                vislib::TString rv;
                rv.Format(_T("%%.%dd"), p1 - pos);
                rv.Prepend(inoutName.Substring(0, pos + 1));
                rv.Append(inoutName.Substring(p1 + 1));
                outValue = static_cast<int>(vislib::TCharTraits::ParseInt(inoutName.Substring(pos + 1, p1 - pos)));
                inoutName = rv;
                return true;
            }
        }
    }

    return false;
}


/*
 * megamol::core::DataFileSequencer::onNextFile
 */
bool megamol::core::DataFileSequencer::onNextFile(param::ParamSlot& param) {
    using vislib::sys::Log;
    param::ParamSlot *fnps = this->findFilenameSlot();
    if (fnps == NULL) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
            "Cannot find parameter slot \"%s\"",
            vislib::StringA(this->filenameSlotNameSlot.Param<param::StringParam>()->Value()).PeekBuffer());
        return true;
    }
    vislib::TString fnf;
    if (!this->GetFilename(*fnps, fnf)) return true;
    int val;
    if (!this->GetFormat(fnf, val)) return true;

    val++;
    vislib::TString fn;
    fn.Format(fnf, val);
    if (vislib::sys::File::Exists(fn)) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_INFO, "Switching to file \"%s\"", vislib::StringA(fn).PeekBuffer());
        this->SetFilename(*fnps, fn);
        return true;
    }

    Log::DefaultLog.WriteMsg(Log::LEVEL_INFO, "Next file not found. Going to start of series.");
    // search for last existing file
    while (val >= 0) {
        val--;
        fn.Format(fnf, val);
        if (vislib::sys::File::Exists(fn)) break;
    }
    if (val < 0) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "File name series \"%s\" does not seem to exist at all.",
            vislib::StringA(fnf).PeekBuffer());
        return true;
    }
    while (val >= 0) {
        val--;
        fn.Format(fnf, val);
        if (!vislib::sys::File::Exists(fn)) break;
    }
    val++;
    fn.Format(fnf, val);
    if (vislib::sys::File::Exists(fn)) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_INFO, "Switching to file \"%s\"", vislib::StringA(fn).PeekBuffer());
        this->SetFilename(*fnps, fn);
        return true;
    }

    Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "File name series \"%s\" does not seem to exist at all.",
        vislib::StringA(fnf).PeekBuffer());

    return true;
}


/*
 * megamol::core::DataFileSequencer::onPrevFile
 */
bool megamol::core::DataFileSequencer::onPrevFile(param::ParamSlot& param) {
    using vislib::sys::Log;
    const int MAX_FILE_NUM = 10000000;
    param::ParamSlot *fnps = this->findFilenameSlot();
    if (fnps == NULL) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
            "Cannot find parameter slot \"%s\"",
            vislib::StringA(this->filenameSlotNameSlot.Param<param::StringParam>()->Value()).PeekBuffer());
        return true;
    }
    vislib::TString fnf;
    if (!this->GetFilename(*fnps, fnf)) return true;
    int val;
    if (!this->GetFormat(fnf, val)) return true;

    val--;
    vislib::TString fn;
    if (val > 0) {
        fn.Format(fnf, val);
        if (vislib::sys::File::Exists(fn)) {
            Log::DefaultLog.WriteMsg(Log::LEVEL_INFO, "Switching to file \"%s\"", vislib::StringA(fn).PeekBuffer());
            this->SetFilename(*fnps, fn);
            return true;
        }
    }

    Log::DefaultLog.WriteMsg(Log::LEVEL_INFO, "Previous file not found. Going to end of series.");
    // search for first existing file
    while (val < MAX_FILE_NUM) {
        val++;
        fn.Format(fnf, val);
        if (vislib::sys::File::Exists(fn)) break;
    }
    if (val >= MAX_FILE_NUM) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "File name series \"%s\" does not seem to exist at all.",
            vislib::StringA(fnf).PeekBuffer());
        return true;
    }
    while (val < MAX_FILE_NUM) {
        val++;
        fn.Format(fnf, val);
        if (!vislib::sys::File::Exists(fn)) break;
    }
    val--;
    fn.Format(fnf, val);
    if (vislib::sys::File::Exists(fn)) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_INFO, "Switching to file \"%s\"", vislib::StringA(fn).PeekBuffer());
        this->SetFilename(*fnps, fn);
        return true;
    }

    Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "File name series \"%s\" does not seem to exist at all.",
        vislib::StringA(fnf).PeekBuffer());

    return true;}


/*
 * megamol::core::DataFileSequencer::release
 */
void megamol::core::DataFileSequencer::release(void) {
    // Intentionally empty
}


/*
 * megamol::core::DataFileSequencer::SetFilename
 */
bool megamol::core::DataFileSequencer::SetFilename(param::ParamSlot& slot, const vislib::TString& name) const {
    using vislib::sys::Log;
    param::StringParam *sp = slot.Param<param::StringParam>();
    param::FilePathParam *fpp = slot.Param<param::FilePathParam>();
    if ((sp == NULL) && (fpp == NULL)) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
            "Parameter \"%s\" is not of a compatible type.",
            vislib::StringA(this->filenameSlotNameSlot.Param<param::StringParam>()->Value()).PeekBuffer());
        return false;
    }
    if (sp == NULL) {
        fpp->SetValue(name);
    } else {
        sp->SetValue(name);
    }
    return true;
}
