/*
 * utility/HotFixFileName.cpp
 *
 * Copyright (C) 2016 by MegaMol Team
 * Alle Rechte vorbehalten.
 */
#include "stdafx.h"
#include "utility/HotFixFileName.h"
#include "mmcore/api/MegaMolCore.h"
#include "vislib/String.h"
#include "vislib/sys/File.h"
#include "CoreHandle.h"
#include <KeyboardMouseInput.h>

using namespace megamol;
using namespace megamol::console;
using namespace megamol::input_events;

extern "C" {

/**
* Searching for file name parameters
*
* @param str The parameter name
* @param data not used
*/
void MEGAMOLCORE_CALLBACK fixFileNameEnum(const char *str, void *hCore) {
    vislib::StringA n(str);
    n.ToLowerCase();
    if (n.EndsWith("filename")) {
        megamol::console::CoreHandle hParam;

        if (!::mmcGetParameterA(hCore, str, hParam)) {
            fprintf(stderr, "Failed to get handle for parameter %s\n", str);
            return;
        }

        vislib::StringA fn(::mmcGetParameterValueA(hParam));

        if (!vislib::sys::File::Exists(fn)) {
            // we need to search for a better file

            // try 1: remove last char:
            vislib::StringA tfn = fn.Substring(0, fn.Length() - 1);
            if (vislib::sys::File::Exists(tfn)) {
                ::mmcSetParameterValueA(hParam, tfn);
                return;
            }

        }

    }
}

}

utility::HotFixFileName::HotFixFileName(void* hCore)
        : hCore(hCore) {
    // intentionally empty
}

utility::HotFixFileName::~HotFixFileName() {
    hCore = nullptr; // we don't own, so we don't delete
}

bool utility::HotFixFileName::OnKey(Key key,  KeyAction action, Modifiers mods) {
    if (((key == Key::KEY_F)) && (action == KeyAction::PRESS) && mods.none()) {
        ::mmcEnumParametersA(hCore, fixFileNameEnum, hCore);
        return true;
    }
    return false;
}
