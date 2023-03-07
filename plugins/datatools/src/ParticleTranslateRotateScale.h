/*
 * ParticleTranslateRotateScale.h
 *
 * Copyright (C) 2018 MegaMol Team
 * Alle Rechte vorbehalten.
 */

#pragma once

#include "datatools/AbstractParticleManipulator.h"
#include "mmcore/param/ParamSlot.h"


namespace megamol::datatools {

/**
 * Module thinning the number of particles
 *
 * Migrated from SGrottel particle's tool box
 */
class ParticleTranslateRotateScale : public AbstractParticleManipulator {
public:
    /** Return module class name */
    static const char* ClassName() {
        return "ParticleTranslateRotateScale";
    }

    /** Return module class description */
    static const char* Description() {
        return "Rotates, translates and scales the data";
    }

    /** Module is always available */
    static bool IsAvailable() {
        return true;
    }

    /** Ctor */
    ParticleTranslateRotateScale();

    /** Dtor */
    ~ParticleTranslateRotateScale() override;
    bool InterfaceIsDirty() const;
    void InterfaceResetDirty();

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
    bool manipulateExtent(geocalls::MultiParticleDataCall& outData, geocalls::MultiParticleDataCall& inData) override;

private:
    core::param::ParamSlot translateSlot;
    core::param::ParamSlot quaternionSlot;
    core::param::ParamSlot scaleSlot;

    size_t hash = -1;
    unsigned int frameID = -1;

    std::vector<std::vector<float>> finalData;
    std::vector<vislib::math::Cuboid<float>> finalLocalBBox;
    vislib::math::Cuboid<float> _global_box;
};

} // namespace megamol::datatools
