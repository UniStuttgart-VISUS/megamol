/*
 * ColourHSVf.cpp
 *
 * Copyright (C) 2015 by Sebastian Grottel, TU Dresden
 * Alle Rechte vorbehalten.
 */

#include "vislib/graphics/ColourHSVf.h"
#include <algorithm>


/*
 * vislib::graphics::ColourHSVf::ColourHSVf
 */
vislib::graphics::ColourHSVf::ColourHSVf() {
    this->comp[0] = 0;
    this->comp[1] = 0;
    this->comp[2] = 0;
}


/*
 * vislib::graphics::ColourHSVf::ColourHSVf
 */
vislib::graphics::ColourHSVf::ColourHSVf(float h, float s, float v) {
    SetH(h);
    SetS(s);
    SetV(v);
}


/*
 * vislib::graphics::ColourHSVf::ColourHSVf
 */
vislib::graphics::ColourHSVf::ColourHSVf(const vislib::graphics::ColourHSVf& src) {
    *this = src;
}


/*
 * vislib::graphics::ColourHSVf::ColourHSVf
 */
vislib::graphics::ColourHSVf::ColourHSVf(const vislib::graphics::ColourRGBAu8& src) {
    const float epsilon = 0.0001f;
    float R = static_cast<float>(src.R() * 255.0f);
    float G = static_cast<float>(src.G() * 255.0f);
    float B = static_cast<float>(src.B() * 255.0f);
    float MAX = std::max<float>(R, std::max<float>(G, B));
    float MIN = std::min<float>(R, std::min<float>(G, B));

    if (math::IsEqual(MAX, MIN, epsilon)) {
        comp[0] = 0.0f;
    } else if (math::IsEqual(MAX, R, epsilon)) {
        comp[0] = 60.0f * (0.0f + (G - B) / (MAX - MIN));
    } else if (math::IsEqual(MAX, G, epsilon)) {
        comp[0] = 60.0f * (2.0f + (B - R) / (MAX - MIN));
    } else if (math::IsEqual(MAX, B, epsilon)) {
        comp[0] = 60.0f * (4.0f + (R - G) / (MAX - MIN));
    } else {
        comp[0] = 0.0f; // should never happen
    }
    SetH(comp[0]); // to clamp to the right range

    comp[1] = math::IsEqual(MAX, 0.0f, epsilon) ? 0.0f : ((MAX - MIN) / MAX);

    comp[2] = MAX;
}


/*
 * vislib::graphics::ColourHSVf::~ColourHSVf
 */
vislib::graphics::ColourHSVf::~ColourHSVf() {
    // Intentionally empty
}


/*
 * vislib::graphics::ColourHSVf::operator==
 */
bool vislib::graphics::ColourHSVf::operator==(const vislib::graphics::ColourHSVf& rhs) const {
    return vislib::math::IsEqual(this->comp[0], rhs.comp[0], 0.0001f) &&
           vislib::math::IsEqual(this->comp[1], rhs.comp[1], 0.0001f) &&
           vislib::math::IsEqual(this->comp[2], rhs.comp[2], 0.0001f);
}


/*
 * vislib::graphics::ColourHSVf::operator ColourRGBAu8
 */
vislib::graphics::ColourHSVf::operator vislib::graphics::ColourRGBAu8() const {
    unsigned int h = static_cast<unsigned int>(std::floor(comp[0] / 60.0f)) % 6u;
    float f = comp[0] / 60.0f - static_cast<float>(h);

    float p = comp[2] * (1.0f - comp[1]);
    float q = comp[2] * (1.0f - comp[1] * f);
    float t = comp[2] * (1.0f - comp[1] * (1.0f - f));

    float r, g, b;
    switch (h) {
    case 0:
        r = comp[2];
        g = t;
        b = p;
        break;
    case 1:
        r = q;
        g = comp[2];
        b = p;
        break;
    case 2:
        r = p;
        g = comp[2];
        b = t;
        break;
    case 3:
        r = p;
        g = q;
        b = comp[2];
        break;
    case 4:
        r = t;
        g = p;
        b = comp[2];
        break;
    case 5:
        r = comp[2];
        g = p;
        b = q;
        break;
    default:
        r = g = b = 0.0f;
    }

    return ColourRGBAu8(static_cast<unsigned char>(r * 255.0f), static_cast<unsigned char>(g * 255.0f),
        static_cast<unsigned char>(b * 255.0f), 255);
}
