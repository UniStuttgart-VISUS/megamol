/*
 * StringParam.cpp
 *
 * Copyright (C) 2008 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "mmcore/param/ColorParam.h"

#include "mmcore/utility/ColourParser.h"

#include "vislib/StringConverter.h"


using namespace megamol::core::param;

ColorParam::ColorParam(const Type& initVal) { std::memcpy(val, initVal, sizeof(Type)); }

ColorParam::ColorParam(const vislib::TString& initVal) { ParseValue(initVal); }

ColorParam::~ColorParam(void) {}

void ColorParam::Definition(vislib::RawStorage& outDef) const {
    outDef.AssertSize(6);
#if defined(UNICODE) || defined(_UNICODE)
    memcpy(outDef.AsAt<char>(0), "MMCOLW", 6);
#else  /* defined(UNICODE) || defined(_UNICODE) */
    memcpy(outDef.AsAt<char>(0), "MMCOLA", 6);
#endif /* defined(UNICODE) || defined(_UNICODE) */
}

bool ColorParam::ParseValue(vislib::TString const& v) {
    try {
        float vParsed[4];
        if (core::utility::ColourParser::FromString(v, 4, vParsed)) {
            this->SetValue(vParsed);
            return true;
        }
    } catch (...) {
    }
    return false;
}

vislib::TString ColorParam::ValueString(void) const {
    return core::utility::ColourParser::ToString(this->val[0], this->val[1], this->val[2], this->val[3]);
}

void ColorParam::SetValue(const Type& v, bool setDirty) {
    if (std::memcmp(this->val, v, sizeof(Type))) {
        std::memcpy(this->val, v, sizeof(Type));
        if (setDirty) this->setDirty();
    }
}
