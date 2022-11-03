/**
 * MegaMol
 * Copyright (c) 2020, MegaMol Dev Team
 * All rights reserved.
 */

#pragma once

#include "mmcore/utility/log/Log.h"

namespace megamol::core {

/**
 * Provide flag to present caller/callee slot in GUI depending on necessity
 */
class AbstractCallSlotPresentation {
public:
    enum Necessity { SLOT_OPTIONAL, SLOT_REQUIRED };

    void SetNecessity(Necessity n) {
        this->necessity = n;
    }

    Necessity GetNecessity() {
        return this->necessity;
    }

protected:
    AbstractCallSlotPresentation() {
        this->necessity = Necessity::SLOT_OPTIONAL;
    }

    virtual ~AbstractCallSlotPresentation() = default;

private:
    Necessity necessity;
};

} // namespace megamol::core
