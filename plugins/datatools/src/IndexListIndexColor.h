/*
 * IndexListIndexColor.h
 *
 * Copyright (C) 2016 by MegaMol Team
 * Alle Rechte vorbehalten.
 */
#pragma once

#include "datatools/AbstractParticleManipulator.h"
#include "mmcore/CallerSlot.h"
#include <vector>


namespace megamol {
namespace datatools {

/**
 * Creates an ICol data stream holding the index of the first index list
 * in the connected MultiIndexListDataCall referencing the corresponding
 * particle.
 */
class IndexListIndexColor : public AbstractParticleManipulator {
public:
    /** Factory metadata */
    static const char* ClassName() {
        return "IndexListIndexColor";
    }
    static const char* Description() {
        return "Creates an ICol data stream holding the index of the "
               "first index list in the connected MultiIndexListDataCall "
               "referencing the corresponding particle.";
    }
    static bool IsAvailable() {
        return true;
    }

    /** ctor */
    IndexListIndexColor();
    /** dtor */
    ~IndexListIndexColor() override;

protected:
    /** Create updated particle data */
    bool manipulateData(geocalls::MultiParticleDataCall& outData, geocalls::MultiParticleDataCall& inData) override;

private:
    /** In slot for index lists */
    core::CallerSlot inIndexListDataSlot;

    /** Data hashes */
    size_t inPartsHash;
    size_t inIndexHash;
    size_t outHash;
    unsigned int frameID;

    /** new color data */
    std::vector<float> colors;
    float minCol, maxCol;
};

} // namespace datatools
} // namespace megamol
