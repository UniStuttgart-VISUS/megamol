/*
 * ButtonParam.cpp
 *
 * Copyright (C) 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "mmcore/param/ButtonParam.h"

using namespace megamol::core::param;


/*
 * ButtonParam::ButtonParam
 */
ButtonParam::ButtonParam(WORD key) : AbstractParam(), key(key) {
    // intentionally empty
}


/*
 * ButtonParam::ButtonParam
 */
ButtonParam::ButtonParam(const vislib::sys::KeyCode &key) : AbstractParam(),
        key(key) {
    // intentionally empty
}


/*
 * ButtonParam::~ButtonParam
 */
ButtonParam::~ButtonParam(void) {
    // intentionally empty
}


/*
 * ButtonParam::Definition
 */
void ButtonParam::Definition(vislib::RawStorage& outDef) const {
    outDef.AssertSize(8);
    memcpy(outDef.AsAt<char>(0), "MMBUTN", 6);
    *outDef.AsAt<WORD>(6) = (WORD)this->key;
}


/*
 * ButtonParam::ParseValue
 */
bool ButtonParam::ParseValue(const vislib::TString& v) {
    this->setDirty();
    return true;
}


/*
 * ButtonParam::ValueString
 */
vislib::TString ButtonParam::ValueString(void) const {
    // intentionally empty
    return _T("");
}
