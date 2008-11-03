/*
 * KeyCode.cpp
 *
 * Copyright (C) 2006 - 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#include "vislib/KeyCode.h"


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
 * vislib::sys::KeyCode::normalise
 */
void vislib::sys::KeyCode::normalise(void) {
    // the idea of this method is to work around the KEY_MOD_SHIFT problem
    // If you do not understand that problem think of '$' + Shift vs. '$'

    // TODO: Implement something intelligent here

}
