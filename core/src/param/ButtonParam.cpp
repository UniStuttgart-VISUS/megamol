/*
 * ButtonParam.cpp
 *
 * Copyright (C) 2008 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
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
 * ButtonParam::Definition
 */
std::string ButtonParam::Definition() const {
    std::string name = "MMBUTN";
    std::string return_str;
    return_str.resize(name.size() + (2 * sizeof(WORD)));
    std::copy(name.begin(), name.end(), return_str.begin());
    std::copy(reinterpret_cast<char const*>(&this->keycode.key),
        reinterpret_cast<char const*>(&this->keycode.key) + sizeof(WORD), return_str.begin() + name.size());
    auto const mods = this->keycode.mods.toInt();
    std::copy(reinterpret_cast<char const*>(&this->keycode.key), reinterpret_cast<char const*>(&mods) + sizeof(WORD),
        return_str.begin() + name.size() + sizeof(WORD));
    return return_str;
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
std::string ButtonParam::ValueString(void) const {

    // intentionally empty
    return std::string();
}

void ButtonParam::initialize() {
    this->InitPresentation(AbstractParamPresentation::ParamType::BUTTON);
}
