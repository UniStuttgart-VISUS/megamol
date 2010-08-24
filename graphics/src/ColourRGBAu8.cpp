/*
 * ColourRGBAu8.cpp
 *
 * Copyright (C) 2006 - 2010 by Visualisierungsinstitut Universitaet Stuttgart.
 * Alle Rechte vorbehalten.
 */

#include "vislib/ColourRGBAu8.h"


/*
 * vislib::graphics::ColourRGBAu8::ColourRGBAu8
 */
vislib::graphics::ColourRGBAu8::ColourRGBAu8(void) {
    this->comp[0] = 0;
    this->comp[1] = 0;
    this->comp[2] = 0;
    this->comp[3] = 255;
}


/*
 * vislib::graphics::ColourRGBAu8::ColourRGBAu8
 */
vislib::graphics::ColourRGBAu8::ColourRGBAu8(unsigned char r, unsigned char g,
        unsigned char b, unsigned char a) {
    this->comp[0] = r;
    this->comp[1] = g;
    this->comp[2] = b;
    this->comp[3] = a;
}


/*
 * vislib::graphics::ColourRGBAu8::ColourRGBAu8
 */
vislib::graphics::ColourRGBAu8::ColourRGBAu8(
        const vislib::graphics::ColourRGBAu8& src) {
    *this = src;
}


/*
 * vislib::graphics::ColourRGBAu8::~ColourRGBAu8
 */
vislib::graphics::ColourRGBAu8::~ColourRGBAu8(void) {
    // Intentionally empty
}


/*
 * vislib::graphics::ColourRGBAu8::Interpolate
 */
vislib::graphics::ColourRGBAu8 vislib::graphics::ColourRGBAu8::Interpolate(
        const vislib::graphics::ColourRGBAu8& rhs, float t) const {
    float a = 1.0f - t;
    return vislib::graphics::ColourRGBAu8(
        static_cast<unsigned char>(static_cast<float>(this->comp[0]) * a
            + static_cast<float>(rhs.comp[0]) * t),
        static_cast<unsigned char>(static_cast<float>(this->comp[1]) * a
            + static_cast<float>(rhs.comp[1]) * t),
        static_cast<unsigned char>(static_cast<float>(this->comp[2]) * a
            + static_cast<float>(rhs.comp[2]) * t),
        static_cast<unsigned char>(static_cast<float>(this->comp[3]) * a
            + static_cast<float>(rhs.comp[3]) * t));
}


/*
 * vislib::graphics::ColourRGBAu8::Set
 */
void vislib::graphics::ColourRGBAu8::Set(unsigned char r, unsigned char g,
        unsigned char b, unsigned char a) {
    this->comp[0] = r;
    this->comp[1] = g;
    this->comp[2] = b;
    this->comp[3] = a;
}


/*
 * vislib::graphics::ColourRGBAu8::operator==
 */
bool vislib::graphics::ColourRGBAu8::operator==(
        const vislib::graphics::ColourRGBAu8& rhs) const {
    return (this->comp[0] == rhs.comp[0])
        && (this->comp[1] == rhs.comp[1])
        && (this->comp[2] == rhs.comp[2])
        && (this->comp[3] == rhs.comp[3]);
}


/*
 * vislib::graphics::ColourRGBAu8::operator=
 */
vislib::graphics::ColourRGBAu8& vislib::graphics::ColourRGBAu8::operator=(
        const vislib::graphics::ColourRGBAu8& rhs) {
    this->comp[0] = rhs.comp[0];
    this->comp[1] = rhs.comp[1];
    this->comp[2] = rhs.comp[2];
    this->comp[3] = rhs.comp[3];
    return *this;
}
