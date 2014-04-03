/*
 * Ternary.cpp
 *
 * Copyright (C) 2006 - 2009 by Visualisierungsinstitut Universitaet Stuttgart.
 * Alle Rechte vorbehalten.
 */

#include "vislib/Ternary.h"
#include "the/assert.h"


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
bool vislib::math::Ternary::Parse(const the::astring& str) {
    if (the::text::string_utility::equals(str, "yes", false) || the::text::string_utility::equals(str, "y", false)
            || the::text::string_utility::equals(str, "true", false) || the::text::string_utility::equals(str, "t", false)
            || the::text::string_utility::equals(str, "on", false)) {
        this->value = 1;
        return true;
    }
    if (the::text::string_utility::equals(str, "no", false) || the::text::string_utility::equals(str, "n", false)
            || the::text::string_utility::equals(str, "false", false) || the::text::string_utility::equals(str, "f", false)
            || the::text::string_utility::equals(str, "off", false)) {
        this->value = -1;
        return true;
    }
    if (the::text::string_utility::equals(str, "undefined", false) || the::text::string_utility::equals(str, "undef", false)
            || the::text::string_utility::equals(str, "unknown", false) || the::text::string_utility::equals(str, "u", false)
            || the::text::string_utility::equals(str, "x", false)) { // x comes from 'digital design' VHDL
        this->value = 0;
        return true;
    }

    try {
        *this = the::text::string_utility::parse_int(str);
    } catch(...) {
    }

    return false;
}


/*
 * vislib::math::Ternary::Parse
 */
bool vislib::math::Ternary::Parse(const the::wstring& str) {
    // I know, I am lazy ... I don't care
    return this->Parse(the::text::string_converter::to_a(str));
}


/*
 * vislib::math::Ternary::ToStringA
 */
the::astring vislib::math::Ternary::ToStringA(void) const {
    switch (this->value) {
        case 1: return "true";
        case 0: return "unknown";
        case -1: return "false";
        default: THE_ASSERT(false);
    }
    return "";
}


/*
 * vislib::math::Ternary::ToStringW
 */
the::wstring vislib::math::Ternary::ToStringW(void) const {
    switch (this->value) {
        case 1: return L"true";
        case 0: return L"unknown";
        case -1: return L"false";
        default: THE_ASSERT(false);
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
