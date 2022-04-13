/*
 * AbstractCallSlotPresentation.h
 *
 * Copyright (C) 2020 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_ABSTRACTCALLSLOTPRESENTATION_H_INCLUDED
#define MEGAMOLCORE_ABSTRACTCALLSLOTPRESENTATION_H_INCLUDED

#include "mmcore/utility/log/Log.h"


namespace megamol {
namespace core {

/**
 * Provide flag to present caller/callee slot in GUI depending on necessity
 */
class AbstractCallSlotPresentation {
public:
    enum Necessity { SLOT_OPTIONAL, SLOT_REQUIRED };

    void SetNecessity(Necessity n) {
        this->necessity = n;
    }

    Necessity GetNecessity(void) {
        return this->necessity;
    }

protected:
    AbstractCallSlotPresentation(void) {
        this->necessity = Necessity::SLOT_OPTIONAL;
    }

    virtual ~AbstractCallSlotPresentation(void) = default;

private:
    Necessity necessity;
};

} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_ABSTRACTCALLSLOTPRESENTATION_H_INCLUDED */
