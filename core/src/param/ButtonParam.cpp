/*
 * ButtonParam.cpp
 *
 * Copyright (C) 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "mmcore/param/ButtonParam.h"

using namespace megamol;
using namespace megamol::core::param;


/*
 * ButtonParam::ButtonParam
 */
ButtonParam::ButtonParam(void) : AbstractParam(), keycode() {

    // intentionally empty
}


/*
 * ButtonParam::ButtonParam
 */
ButtonParam::ButtonParam(const core::view::KeyCode &keycode) : AbstractParam(),
    keycode(keycode) {

    // intentionally empty
}


/*
 * ButtonParam::ButtonParam
 */
ButtonParam::ButtonParam(const core::view::Key &key, const core::view::Modifiers &mods) : AbstractParam(), 
    keycode(key, mods) {

    // intentionally empty
}


/*
 * ButtonParam::ButtonParam
 */
ButtonParam::ButtonParam(const core::view::Key &key, const core::view::Modifier &mod) : AbstractParam(),
keycode(key, core::view::Modifiers(mod)) {

    // intentionally empty
}


/*
 * ButtonParam::ButtonParam
 */
ButtonParam::ButtonParam(const core::view::Key &key) : AbstractParam(), keycode() {

    this->keycode.key = key;
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

    outDef.AssertSize(6 + (2 * sizeof(WORD)));
    memcpy(outDef.AsAt<char>(0), "MMBUTN", 6);
    *outDef.AsAt<WORD>(6) = (WORD)this->keycode.key;
    core::view::Modifiers mods = this->keycode.mods;
    *outDef.AsAt<WORD>(6 + sizeof(WORD)) = (WORD)mods.toInt();
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
