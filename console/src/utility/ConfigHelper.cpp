/*
 * ConfigHelper.cpp
 *
 * Copyright (C) 2016 by MegaMol Team, TU Dresden
 * Alle Rechte vorbehalten.
 */
#include "stdafx.h"
#include "utility/ConfigHelper.h"
#include "mmcore/api/MegaMolCore.h"
#include <cstdint>
#include <climits>

using namespace megamol;
using namespace megamol::console;


bool utility::WindowPlacement::Parse(const vislib::TString& str) {
    int x, y, w, h, f;
    bool nd; // no decorations (HAZARD: what about full screen?)

    x = y = w = h = f = INT_MIN;
    nd = false;

    vislib::StringW v(str);
    int vi = -1;
    v.TrimSpaces();

    while (!v.IsEmpty()) {
        if ((v[0] == L'X') || (v[0] == L'x')) vi = 0;
        else if ((v[0] == L'Y') || (v[0] == L'y')) vi = 1;
        else if ((v[0] == L'W') || (v[0] == L'w')) vi = 2;
        else if ((v[0] == L'H') || (v[0] == L'h')) vi = 3;
        else if ((v[0] == L'F') || (v[0] == L'f')) vi = 5;
        else if ((v[0] == L'N') || (v[0] == L'n')) vi = 4;
        else if ((v[0] == L'D') || (v[0] == L'd')) {
            nd = (vi == 4);
            vi = 4;
        } else {
            /*Log::DefaultLog.WriteMsg(
            vislib::sys::Log::LEVEL_WARN,
            "Unexpected character %s in window position definition.\n",
            vislib::StringA(vislib::StringA(v)[0], 1).PeekBuffer());*/
            break;
        }
        v = v.Substring(1);
        v.TrimSpaces();

        if (vi == 4) continue; // [n]d are not followed by a number

        if (vi >= 0) {
            // now we want to parse an int
            int cp = 0;
            int len = v.Length();
            while ((cp < len) && (((v[cp] >= L'0') && (v[cp] <= L'9')) || (v[cp] == L'+') || (v[cp] == L'-'))) {
                cp++;
            }

            try {
                int i = (cp > 0) ? vislib::CharTraitsW::ParseInt(v.Substring(0, cp)) : 0;
                switch (vi) {
                case 0: x = i; break;
                case 1: y = i; break;
                case 2: w = i; break;
                case 3: h = i; break;
                case 5: f = i; break;
                }
            } catch (...) {
            }
            v = v.Substring(cp);
        }

    }

    if (x != INT_MIN && y != INT_MIN) {
        this->x = x;
        this->y = y;
        this->pos = true;
    } else {
        this->x = 0;
        this->y = 0;
        this->pos = false;
    }
    if (w != INT_MIN && h != INT_MIN) {
        this->w = w;
        this->h = h;
        this->size = true;
    } else {
        this->w = 0;
        this->h = 0;
        this->size = false;
    }
    this->noDec = nd;
    if (f != INT_MIN) {
        this->mon = f;
        this->fullScreen = true;
    } else {
        this->mon = 0;
        this->fullScreen = false;
    }

    return true;
}

vislib::TString utility::WindowPlacement::ToString() const {
    vislib::TString v;
    if (pos) {
        vislib::TString t;
        t.Format(_T("x%dy%d"), x, y);
        v += t;
    }
    if (size) {
        vislib::TString t;
        t.Format(_T("w%dh%d"), w, h);
        v += t;
    }
    if (fullScreen) {
        vislib::TString t;
        t.Format(_T("f%d"), mon);
        v += t;
    }
    if (noDec) {
        v += _T("nd");
    }
    return v;
}

namespace {

    vislib::math::Ternary boolCfgVal(void *hCore, const char *name) {
        mmcValueType type;
        const void *cfgv = ::mmcGetConfigurationValueA(hCore, MMC_CFGID_VARIABLE, name, &type);
        switch (type) {
        case MMC_TYPE_BOOL:
            return (*static_cast<const bool*>(cfgv)) ? vislib::math::Ternary::TRI_TRUE : vislib::math::Ternary::TRI_FALSE;
            break;
        case MMC_TYPE_CSTR:
            try {
                bool b = !vislib::CharTraitsA::ParseBool(static_cast<const char *>(cfgv));
                return b ? vislib::math::Ternary::TRI_TRUE : vislib::math::Ternary::TRI_FALSE;
            } catch (...) {
            }
            break;
        case MMC_TYPE_WSTR:
            try {
                bool b = !vislib::CharTraitsW::ParseBool(static_cast<const wchar_t *>(cfgv));
                return b ? vislib::math::Ternary::TRI_TRUE : vislib::math::Ternary::TRI_FALSE;
            } catch (...) {
            }
            break;
    #ifndef _WIN32
        default:
            // intentionally empty
            break;
    #endif /* !_WIN32 */
        }
        return vislib::math::Ternary::TRI_UNKNOWN; // not set
    }

}

vislib::math::Ternary utility::VSync(void *hCore) {
    return boolCfgVal(hCore, "vsync");
}

vislib::math::Ternary utility::ShowGUI(void *hCore) {
    return boolCfgVal(hCore, "consolegui");
}

vislib::math::Ternary utility::UseKHRDebug(void *hCore) {
    return boolCfgVal(hCore, "useKHRdebug");
}