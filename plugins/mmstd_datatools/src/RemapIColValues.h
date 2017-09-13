/*
 * RemapIColValues.h
 *
 * Copyright (C) 2016 by MegaMol Team
 * Alle Rechte vorbehalten.
 */
#ifndef MEGAMOL_DATATOOLS_REMAPICOLVALUES_H_INCLUDED
#define MEGAMOL_DATATOOLS_REMAPICOLVALUES_H_INCLUDED
#pragma once

#include "mmstd_datatools/AbstractParticleManipulator.h"
#include "mmcore/CallerSlot.h"
#include <vector>

namespace megamol {
namespace stdplugin {
namespace datatools {


    /**
     * Replaces ICol Values in filtered particles from unfiltered particles
     */
    class RemapIColValues : public AbstractParticleManipulator {
    public:
        static const char *ClassName(void) { return "RemapIColValues"; }
        static const char *Description(void) { return "Replaces ICol Values in filtered particles from unfiltered particles"; }
        static bool IsAvailable(void) { return true; }
        static bool SupportQuickstart(void) { return false; }

        RemapIColValues(void);
        virtual ~RemapIColValues(void);

    protected:

        virtual bool manipulateData(
            megamol::core::moldyn::MultiParticleDataCall& outData,
            megamol::core::moldyn::MultiParticleDataCall& inData);

    private:

        core::CallerSlot inIColValuesSlot;
        core::CallerSlot inParticleMapSlot;

        size_t dataHash;
        size_t inIColHash;
        size_t inMapHash;
        size_t outDataHash;
        unsigned int frameId;
        std::vector<float> col;
        float minCol, maxCol;

    };

} /* end namespace datatools */
} /* end namespace stdplugin */
} /* end namespace megamol */

#endif /* MEGAMOL_DATATOOLS_REMAPICOLVALUES_H_INCLUDED */
