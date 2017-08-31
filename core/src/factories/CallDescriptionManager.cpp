/*
 * CallDescriptionManager.cpp
 * Copyright (C) 2008 - 2015 by MegaMol Consortium
 * All rights reserved. Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "mmcore/factories/CallDescriptionManager.h"


/*
 * megamol::core::factories::CallDescriptionManager::CallDescriptionManager
 */
megamol::core::factories::CallDescriptionManager::CallDescriptionManager()
        : ObjectDescriptionManager<megamol::core::factories::CallDescription>() {
    // intentionally empty
}


/*
 * megamol::core::factories::CallDescriptionManager::~CallDescriptionManager
 */
megamol::core::factories::CallDescriptionManager::~CallDescriptionManager() {
    // intentionally empty
}


/*
 * megamol::core::factories::CallDescriptionManager::AssignmentCrowbar
 */
bool megamol::core::factories::CallDescriptionManager::AssignmentCrowbar(
        megamol::core::Call *tar, megamol::core::Call *src) const {
    for (auto desc : *this) {
        if (desc->IsDescribing(tar)) {
            if (desc->IsDescribing(src)) {
                desc->AssignmentCrowbar(tar, src);
                return true;
            } else {
                // TODO: ARGLHARGLGARGLGARG
            }
        }
    }
    return false;
}
