/*
 * KeyCode.cpp
 *
 * Copyright (C) 2006 - 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#include "vislib/KeyCode.h"
#include "vislib/CharTraits.h"


/*
 * vislib::sys::KeyCode::KeyCode
 */
vislib::sys::KeyCode::KeyCode(void) : key() {
    this->normalise();
}


/*
 * vislib::sys::KeyCode::KeyCode
 */
vislib::sys::KeyCode::KeyCode(WORD key) : key(key) {
    this->normalise();
}


/*
 * vislib::sys::KeyCode::KeyCode
 */
vislib::sys::KeyCode::KeyCode(const vislib::sys::KeyCode& src) : key(src.key) {
    // intentionally empty
}


/*
 * vislib::sys::KeyCode::~KeyCode
 */
vislib::sys::KeyCode::~KeyCode(void) {
    // intentionally empty
}


/*
 * vislib::sys::KeyCode::ToStringA
 */
vislib::StringA vislib::sys::KeyCode::ToStringA(void) const {
    vislib::StringA msg;
    if (this->IsShiftMod()) msg += "Shift + ";
    if (this->IsCtrlMod()) msg += "Ctrl + ";
    if (this->IsAltMod()) msg += "Alt + ";
    if (this->IsSpecial()) {
        switch (this->NoModKeys()) {
            case vislib::sys::KeyCode::KEY_ENTER: msg += "Enter"; break;
            case vislib::sys::KeyCode::KEY_ESC: msg += "ESC"; break;
            case vislib::sys::KeyCode::KEY_TAB: msg += "Tab"; break;
            case vislib::sys::KeyCode::KEY_LEFT: msg += "Left"; break;
            case vislib::sys::KeyCode::KEY_UP: msg += "Up"; break;
            case vislib::sys::KeyCode::KEY_DOWN: msg += "Down"; break;
            case vislib::sys::KeyCode::KEY_PAGE_UP: msg += "Page Up"; break;
            case vislib::sys::KeyCode::KEY_PAGE_DOWN: msg += "Page Down"; break;
            case vislib::sys::KeyCode::KEY_HOME: msg += "Home"; break;
            case vislib::sys::KeyCode::KEY_END: msg += "End"; break;
            case vislib::sys::KeyCode::KEY_INSERT: msg += "Insert"; break;
            case vislib::sys::KeyCode::KEY_DELETE: msg += "Delete"; break;
            case vislib::sys::KeyCode::KEY_BACKSPACE: msg += "Backspace"; break;
            case vislib::sys::KeyCode::KEY_F1: msg += "F1"; break;
            case vislib::sys::KeyCode::KEY_F2: msg += "F2"; break;
            case vislib::sys::KeyCode::KEY_F3: msg += "F3"; break;
            case vislib::sys::KeyCode::KEY_F4: msg += "F4"; break;
            case vislib::sys::KeyCode::KEY_F5: msg += "F5"; break;
            case vislib::sys::KeyCode::KEY_F6: msg += "F6"; break;
            case vislib::sys::KeyCode::KEY_F7: msg += "F7"; break;
            case vislib::sys::KeyCode::KEY_F8: msg += "F8"; break;
            case vislib::sys::KeyCode::KEY_F9: msg += "F9"; break;
            case vislib::sys::KeyCode::KEY_F10: msg += "F10"; break;
            case vislib::sys::KeyCode::KEY_F11: msg += "F11"; break;
            case vislib::sys::KeyCode::KEY_F12: msg += "F12"; break;
            default: {
                vislib::StringA ks;
                ks.Format("[%d]", static_cast<int>(this->NoModKeys()));
                msg += ks;
            } break;
        }
    } else {
        char c = key & 0x00FF;
        if (c == 20) {
            msg += "Space";
        } else {
            vislib::StringA ks;
            if (c < 20) {
                ks.Format("[%d]", static_cast<int>(c));
            } else {
                ks.Format("'%c'", c);
            }
            msg += ks;
        }
    }
    return msg;
}


/*
 * vislib::sys::KeyCode::operator=
 */
vislib::sys::KeyCode& vislib::sys::KeyCode::operator=(WORD rhs) {
    this->key = rhs;
    this->normalise();
    return *this;
}


/*
 * vislib::sys::KeyCode::operator=
 */
vislib::sys::KeyCode& vislib::sys::KeyCode::operator=(
        const vislib::sys::KeyCode& rhs) {
    this->key = rhs.key;
    return *this;
}


/*
 * vislib::sys::KeyCode::operator==
 */
bool vislib::sys::KeyCode::operator==(const vislib::sys::KeyCode& rhs) const {
    return this->key == rhs.key;
}


/*
 * vislib::sys::KeyCode::operator WORD
 */
vislib::sys::KeyCode::operator WORD(void) const {
    return this->key;
}


/*
 * vislib::sys::KeyCode::normalise
 */
void vislib::sys::KeyCode::normalise(void) {
    // the idea of this method is to work around the KEY_MOD_SHIFT problem
    // If you do not understand that problem think of '$' + Shift vs. '$'

    if (this->IsSpecial()) return; // special keys are as they are

    char c = static_cast<char>(this->key & 0x00FF);
    if (vislib::CharTraitsA::IsUpperCase(c)) {
        this->key = (this->key & 0xFF00) | KEY_MOD_SHIFT | vislib::CharTraitsA::ToLower(c);
    }

    // TODO: Implement something intelligent here

}
