/*
 * Ternary.cpp
 *
 * Copyright (C) 2006 - 2009 by Visualisierungsinstitut Universitaet Stuttgart.
 * Alle Rechte vorbehalten.
 */

#include "vislib/Ternary.h"
#include "vislib/assert.h"


/*
 * vislib::math::Ternary::TRUE
 */
const vislib::math::Ternary vislib::math::Ternary::TRI_TRUE(1);


/*
 * vislib::math::Ternary::UNKNOWN
 */
const vislib::math::Ternary vislib::math::Ternary::TRI_UNKNOWN(0);


/*
 * vislib::math::Ternary::FALSE
 */
const vislib::math::Ternary vislib::math::Ternary::TRI_FALSE(-1);


/*
 * vislib::math::Ternary::Ternary
 */
vislib::math::Ternary::Ternary(const Ternary& src) : value(src.value) {
    // Intentionally empty
}


/*
 * vislib::math::Ternary::Ternary
 */
vislib::math::Ternary::Ternary(int value) : value(0) {
    *this = value;
}


/*
 * vislib::math::Ternary::Ternary
 */
vislib::math::Ternary::Ternary(bool value) : value(0) {
    *this = value;
}


/*
 * vislib::math::Ternary::~Ternary
 */
vislib::math::Ternary::~Ternary(void) {
    // Intentionally empty
}


/*
 * vislib::math::Ternary::Parse
 */
bool vislib::math::Ternary::Parse(const vislib::StringA& str) {
    if (str.Equals("yes", false) || str.Equals("y", false)
            || str.Equals("true", false) || str.Equals("t", false)
            || str.Equals("on", false)) {
        this->value = 1;
        return true;
    }
    if (str.Equals("no", false) || str.Equals("n", false)
            || str.Equals("false", false) || str.Equals("f", false)
            || str.Equals("off", false)) {
        this->value = -1;
        return true;
    }
    if (str.Equals("undefined", false) || str.Equals("undef", false)
            || str.Equals("unknown", false) || str.Equals("u", false)
            || str.Equals("x", false)) { // x comes from 'digital design' VHDL
        this->value = 0;
        return true;
    }

    try {
        *this = CharTraitsA::ParseInt(str);
    } catch(...) {
    }

    return false;
}


/*
 * vislib::math::Ternary::Parse
 */
bool vislib::math::Ternary::Parse(const vislib::StringW& str) {
    // I know, I am lazy ... I don't care
    return this->Parse(vislib::StringA(str));
}


/*
 * vislib::math::Ternary::ToStringA
 */
vislib::StringA vislib::math::Ternary::ToStringA(void) const {
    switch (this->value) {
        case 1: return "true";
        case 0: return "unknown";
        case -1: return "false";
        default: ASSERT(false);
    }
    return "";
}


/*
 * vislib::math::Ternary::ToStringW
 */
vislib::StringW vislib::math::Ternary::ToStringW(void) const {
    switch (this->value) {
        case 1: return L"true";
        case 0: return L"unknown";
        case -1: return L"false";
        default: ASSERT(false);
    }
    return L"";
}


/*
 * vislib::math::Ternary::operator=
 */
vislib::math::Ternary& vislib::math::Ternary::operator=(
        const vislib::math::Ternary &rhs) {
    this->value = rhs.value;
    return *this;
}


/*
 * vislib::math::Ternary::operator=
 */
vislib::math::Ternary& vislib::math::Ternary::operator=(int rhs) {
    this->value = this->getValue(rhs);
    return *this;
}


/*
 * vislib::math::Ternary::operator=
 */
vislib::math::Ternary& vislib::math::Ternary::operator=(bool rhs) {
    this->value = this->getValue(rhs);
    return *this;
}


/*
 * vislib::math::Ternary::operator==
 */
bool vislib::math::Ternary::operator==(const Ternary& rhs) const {
    return this->value == rhs.value;
}


/*
 * vislib::math::Ternary::operator==
 */
bool vislib::math::Ternary::operator==(int rhs) const {
    return this->value == this->getValue(rhs);
}


/*
 * vislib::math::Ternary::operator==
 */
bool vislib::math::Ternary::operator==(bool rhs) const {
    return this->value == this->getValue(rhs);
}


/*
 * vislib::math::Ternary::operator!
 */
vislib::math::Ternary vislib::math::Ternary::operator!(void) const {
    return Ternary(-this->value);
}


/*
 * vislib::math::Ternary::operator!
 */
vislib::math::Ternary vislib::math::Ternary::operator-(void) const {
    return Ternary(-this->value);
}


/*
 * vislib::math::Ternary::operator~
 */
vislib::math::Ternary vislib::math::Ternary::operator~(void) const {
    return Ternary(this->value != 1);
}


/*
 * vislib::math::Ternary::operator&
 */
vislib::math::Ternary vislib::math::Ternary::operator&(const vislib::math::Ternary& rhs) const {
    return Ternary(*this) &= rhs;
}


/*
 * vislib::math::Ternary::operator&=
 */
vislib::math::Ternary& vislib::math::Ternary::operator&=(const vislib::math::Ternary& rhs) {
    if ((this->value < 0) || (rhs.value < 0)) {
        this->value = -1;
    } else if ((this->value == 0) || (rhs.value == 0)) {
        this->value = 0;
    } else {
        this->value = 1;
    }
    return *this;
}


/*
 * vislib::math::Ternary::operator|
 */
vislib::math::Ternary vislib::math::Ternary::operator|(const vislib::math::Ternary& rhs) const {
    return Ternary(*this) |= rhs;
}


/*
 * vislib::math::Ternary::operator|=
 */
vislib::math::Ternary& vislib::math::Ternary::operator|=(const vislib::math::Ternary& rhs) {
    if ((this->value > 0) || (rhs.value > 0)) {
        this->value = 1;
    } else if ((this->value == 0) || (rhs.value == 0)) {
        this->value = 0;
    } else {
        this->value = -1;
    }
    return *this;
}


/*
 * vislib::math::Ternary::operator int
 */
vislib::math::Ternary::operator int(void) const {
    return this->value;
}


/*
 * vislib::math::Ternary::getValue
 */
int vislib::math::Ternary::getValue(int v) const {
    return (v > 0) ? 1 : ((v < 0) ? -1 : 0);
}


/*
 * vislib::math::Ternary::getValue
 */
int vislib::math::Ternary::getValue(bool v) const {
    return v ? 1 : -1;
}
