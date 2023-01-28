/*
 * SolventCounter.h
 *
 * Copyright (C) 2015 by Michael Krone
 * Copyright (C) 2015 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#ifndef MMMOLMAPPLG_SOLVENTCOUNTER_H_INCLUDED
#define MMMOLMAPPLG_SOLVENTCOUNTER_H_INCLUDED
#pragma once

#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/Module.h"
#include "mmcore/param/ParamSlot.h"
#include "protein_calls/MolecularDataCall.h"
#include "vislib/Array.h"


namespace megamol::protein {

/**
 * Class for loading MSMS mesh data
 */
class SolventCounter : public core::Module {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName() {
        return "SolventCounter";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description() {
        return "Finds solvent molecules within a given distance for each atom";
    }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable() {
        return true;
    }

    /** Ctor */
    SolventCounter();

    /** Dtor */
    ~SolventCounter() override;

    float GetMinValue() const {
        return this->minValue;
    }
    float GetMidValue() const {
        return this->midValue;
    }
    float GetMaxValue() const {
        return this->maxValue;
    }

protected:
    /**
     * Implementation of 'Create'.
     *
     * @return 'true' on success, 'false' otherwise.
     */
    bool create() override;

    /**
     * Implementation of 'Release'.
     */
    void release() override;

private:
    /**
     * Gets the data from the source.
     *
     * @param caller The calling call.
     *
     * @return 'true' on success, 'false' on failure.
     */
    bool getDataCallback(core::Call& caller);

    /** The slot for requesting data */
    core::CalleeSlot getDataSlot;

    /** The slot for getting protein data */
    core::CallerSlot molDataSlot;

    /** The slot for getting solvent data */
    core::CallerSlot solDataSlot;

    /** MSMS detail parameter */
    megamol::core::param::ParamSlot radiusParam;

    /** The array that stores the solvent around each atom */
    vislib::Array<float> solvent;

    float minValue;
    float midValue;
    float maxValue;

    /**
     * A unique hash number of the returned data, or zero if such a number
     * can not be provided
     */
    SIZE_T datahash;
};

} // namespace megamol::protein

#endif /* MMMOLMAPPLG_SolventCounter_H_INCLUDED */
