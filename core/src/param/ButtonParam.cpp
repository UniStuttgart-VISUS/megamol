/**
 * MegaMol
 * Copyright (c) 2008, MegaMol Dev Team
 * All rights reserved.
 */

#include "mmcore/param/ButtonParam.h"

using namespace megamol;
using namespace megamol::core::param;


/*
 * ButtonParam::ButtonParam
 */
ButtonParam::ButtonParam() : AbstractParam(), keycode() {
    initialize();
}


/*
 * ButtonParam::ButtonParam
 */
ButtonParam::ButtonParam(const core::view::KeyCode& keycode) : AbstractParam(), keycode(keycode) {
    initialize();
}


/*
 * ButtonParam::ButtonParam
 */
ButtonParam::ButtonParam(const core::view::Key& key, const core::view::Modifiers& mods)
        : AbstractParam()
        , keycode(key, mods) {
    initialize();
}


/*
 * ButtonParam::ButtonParam
 */
ButtonParam::ButtonParam(const core::view::Key& key, const core::view::Modifier& mod)
        : AbstractParam()
        , keycode(key, core::view::Modifiers(mod)) {
    initialize();
}


/*
 * ButtonParam::ButtonParam
 */
ButtonParam::ButtonParam(const core::view::Key& key) : AbstractParam(), keycode(key) {
    initialize();
}


/*
 * ButtonParam::~ButtonParam
 */
ButtonParam::~ButtonParam(void) {

    // intentionally empty
}


/*
 * ButtonParam::ParseValue
 */
bool ButtonParam::ParseValue(std::string const& v) {

    this->setDirty();
    return true;
}


/*
 * ButtonParam::ValueString
 */
std::string ButtonParam::ValueString() const {

    // intentionally empty
    return std::string();
}

void ButtonParam::initialize() {
    this->InitPresentation(AbstractParamPresentation::ParamType::BUTTON);
}
