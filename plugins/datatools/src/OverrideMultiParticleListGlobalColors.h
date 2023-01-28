/*
 * OverrideMultiParticleListGlobalColors.h
 *
 * Copyright (C) 2015 by MegaMol Team (TU Dresden)
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_OVERRIDEMULTIPARTICLELISTGLOBALCOLORS_H_INCLUDED
#define MEGAMOLCORE_OVERRIDEMULTIPARTICLELISTGLOBALCOLORS_H_INCLUDED
#pragma once

#include "datatools/AbstractParticleManipulator.h"
#include "mmcore/param/ParamSlot.h"


namespace megamol::datatools {

/**
 * Module overriding global colors of multi particle lists
 */
class OverrideMultiParticleListGlobalColors : public AbstractParticleManipulator {
public:
    static const char* ClassName() {
        return "OverrideMultiParticleListGlobalColors";
    }
    static const char* Description() {
        return "Module overriding global colors of multi particle lists";
    }
    static bool IsAvailable() {
        return true;
    }

    OverrideMultiParticleListGlobalColors();
    ~OverrideMultiParticleListGlobalColors() override;

protected:
    /**
     * Manipulates the particle data
     *
     * @remarks the default implementation does not changed the data
     *
     * @param outData The call receiving the manipulated data
     * @param inData The call holding the original data
     *
     * @return True on success
     */
    bool manipulateData(geocalls::MultiParticleDataCall& outData, geocalls::MultiParticleDataCall& inData) override;

private:
};

} // namespace megamol::datatools

#endif /* MEGAMOLCORE_OVERRIDEMULTIPARTICLELISTGLOBALCOLORS_H_INCLUDED */
