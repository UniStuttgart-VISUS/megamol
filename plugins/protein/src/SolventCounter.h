/*
 * SolventCounter.h
 *
 * Copyright (C) 2015 by Michael Krone
 * Copyright (C) 2015 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#ifndef MMMOLMAPPLG_SOLVENTCOUNTER_H_INCLUDED
#define MMMOLMAPPLG_SOLVENTCOUNTER_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "mmcore/Module.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/param/ParamSlot.h"
#include "protein_calls/MolecularDataCall.h"
#include "vislib/Array.h"


namespace megamol {
namespace protein {

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
        static const char *ClassName(void) {
            return "SolventCounter";
        }

        /**
        * Answer a human readable description of this module.
        *
        * @return A human readable description of this module.
        */
        static const char *Description(void) {
            return "Finds solvent molecules within a given distance for each atom";
        }

        /**
        * Answers whether this module is available on the current system.
        *
        * @return 'true' if the module is available, 'false' otherwise.
        */
        static bool IsAvailable(void) {
            return true;
        }

        /** Ctor */
        SolventCounter(void);

        /** Dtor */
        virtual ~SolventCounter(void);

        float GetMinValue(void) const { return this->minValue; }
        float GetMidValue(void) const { return this->midValue; }
        float GetMaxValue(void) const { return this->maxValue; }

    protected:

        /**
        * Implementation of 'Create'.
        *
        * @return 'true' on success, 'false' otherwise.
        */
        virtual bool create(void);

        /**
        * Implementation of 'Release'.
        */
        virtual void release(void);

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

} /* end namespace protein */
} /* end namespace megamol */

#endif /* MMMOLMAPPLG_SolventCounter_H_INCLUDED */
