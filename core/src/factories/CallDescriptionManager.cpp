/**
 * MegaMol
 * Copyright (c) 2008, MegaMol Dev Team
 * All rights reserved.
 */

#include "mmcore/factories/CallDescriptionManager.h"

#include "mmcore/Call.h"

/*
 * megamol::core::factories::CallDescriptionManager::AssignmentCrowbar
 */
bool megamol::core::factories::CallDescriptionManager::AssignmentCrowbar(
    megamol::core::Call* tar, megamol::core::Call* src) const {
    for (const auto& desc : *this) {
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
