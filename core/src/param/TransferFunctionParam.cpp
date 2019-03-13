/*
 * TransferFunctionParam.cpp
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "mmcore/param/TransferFunctionParam.h"


using namespace megamol::core::param;


TransferFunctionParam::TransferFunctionParam(const std::string& initVal) : val(initVal) {
    ///TODO:  Verify format of string -> JSON ...
}


TransferFunctionParam::TransferFunctionParam(const char *initVal) : val(initVal) {
    ///TODO:  Verify format of string -> JSON ...
}


TransferFunctionParam::TransferFunctionParam(const vislib::StringA& initVal) : val(initVal.PeekBuffer()) {
    ///TODO:  Verify format of string -> JSON ...
}


TransferFunctionParam::~TransferFunctionParam(void) {}


void TransferFunctionParam::Definition(vislib::RawStorage& outDef) const {
    outDef.AssertSize(6);
    memcpy(outDef.AsAt<char>(0), "MMTFFNC", 6);
}


bool TransferFunctionParam::ParseValue(vislib::TString const& v) {

    ///TODO:  Verify format of string -> JSON ...
    try {
        this->val = std::string(v.PeekBuffer());
    } catch (...) {
    }

    return false;
}


void TransferFunctionParam::SetValue(const std::string& v, bool setDirty) {

    ///TODO:  Verify format of string -> JSON ...
    if (v != this->val) {
        this->val = v;
        if (setDirty) this->setDirty();
    }
}


vislib::TString TransferFunctionParam::ValueString(void) const {
    return vislib::TString(this->val.c_str());
}


