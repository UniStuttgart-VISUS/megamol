/*
 * HotKeyButtonParam.cpp
 *
 * Copyright (C) 2009 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "HotKeyButtonParam.h"
#include "mmcore/api/MegaMolCore.h"


/*
 * megamol::console::HotKeyButtonParam::HotKeyButtonParam
 */
megamol::console::HotKeyButtonParam::HotKeyButtonParam(void) : HotKeyAction(),
        hParam(NULL) {
    // intentionally empty
}


/*
 * megamol::console::HotKeyButtonParam::HotKeyButtonParam
 */
megamol::console::HotKeyButtonParam::HotKeyButtonParam(
        const vislib::SmartPtr<CoreHandle>& hParam) : HotKeyAction(),
        hParam(hParam) {
    // intentionally empty
}

/*
 * megamol::console::HotKeyButtonParam::~HotKeyButtonParam
 */
megamol::console::HotKeyButtonParam::~HotKeyButtonParam(void) {
    // intentionally empty
}


/*
 * megamol::console::HotKeyButtonParam::Trigger
 */
void megamol::console::HotKeyButtonParam::Trigger(void) {
    if (!this->hParam.IsNull()) {
        ::mmcSetParameterValueA(*hParam, "");
    }
}
