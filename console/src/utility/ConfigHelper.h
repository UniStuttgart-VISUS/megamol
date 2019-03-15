/*
 * ConfigHelper.h
 *
 * Copyright (C) 2016 by MegaMol Team, TU Dresden
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCON_UTILITY_CONFIGHELPER_H_INCLUDED
#define MEGAMOLCON_UTILITY_CONFIGHELPER_H_INCLUDED
#pragma once

#include "vislib/String.h"
#include "vislib/math/Ternary.h"

namespace megamol {
namespace console {
namespace utility {

    /** Configuration for window placement */
    struct WindowPlacement {
        int x = 100, y = 100, w = 800, h = 600, mon = 0;
        bool pos = false;
        bool size = false;
        bool noDec = false;
        bool fullScreen = false;
        bool topMost = false;

        bool Parse(const vislib::TString& str);
        vislib::TString ToString() const;
    };

    /**
     * Checks the configuration if vsync should be set or unset
     *
     * @return 0 if VSync is not to be set
     *         true if VSync should be enabled
     *         false if VSync should be disabled
     */
    vislib::math::Ternary VSync(void *hCore);

    /**
     * Checks the configuration if the console gui should be enabled or disabled
     *
     * @return 0 if GUI state is default
     *         true if GUI should be shown
     *         false if GUI should be hidden
     */
    vislib::math::Ternary ShowGUI(void *hCore);

    /**
    * Checks the configuration if the KHR debugger should be enabled or disabled
    *
    * @return 0 if KHR state is default
    *         true if KHR should be active
    *         false if KHR should be not active
    */
    vislib::math::Ternary UseKHRDebug(void *hCore);

} /* end namespace utility */
} /* end namespace console */
} /* end namespace megamol */

#endif /* MEGAMOLCON_UTILITY_CONFIGHELPER_H_INCLUDED */
