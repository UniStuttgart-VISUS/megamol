/*
 * DataFileSequence.h
 *
 * Copyright (C) 2010 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_DATAFILESEQUENCE_H_INCLUDED
#define MEGAMOLCORE_DATAFILESEQUENCE_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "Module.h"
#include "param/ParamSlot.h"
#include "CalleeSlot.h"
#include "CallerSlot.h"
#include "vislib/Cuboid.h"


namespace megamol {
namespace core {
namespace moldyn {



    /**
     * In-Between management module for seralize multiple data files with one
     * data frame each (the first if more are available) to write a continous
     * data series into a data writer
     */
    class DataFileSequence : public Module {
    public:

        /**
         * Answer the name of this module.
         *
         * @return The name of this module.
         */
        static const char *ClassName(void) {
            return "DataFileSequence";
        }

        /**
         * Answer a human readable description of this module.
         *
         * @return A human readable description of this module.
         */
        static const char *Description(void) {
            return "Data file sequence module for writing series";
        }

        /**
         * Answers whether this module is available on the current system.
         *
         * @return 'true' if the module is available, 'false' otherwise.
         */
        static bool IsAvailable(void) {
            return true;
        }

        /** Ctor. */
        DataFileSequence(void);

        /** Dtor. */
        virtual ~DataFileSequence(void);

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
        bool getDataCallback(Call& caller);

        /**
         * Gets the data from the source.
         *
         * @param caller The calling call.
         *
         * @return 'true' on success, 'false' on failure.
         */
        bool getExtentCallback(Call& caller);

        /** The file name template */
        param::ParamSlot fileNameTemplateSlot;

        /** Slot for the minimum file number */
        param::ParamSlot fileNumberMinSlot;

        /** Slot for the maximum file number */
        param::ParamSlot fileNumberMaxSlot;

        /** Slot for the file number increase step */
        param::ParamSlot fileNumberStepSlot;

        /** The name of the data source file name parameter slot */
        param::ParamSlot fileNameSlotNameSlot;

        /** The slot for publishing data to the writer */
        CalleeSlot outDataSlot;

        /** The slot for requesting data from the source */
        CallerSlot inDataSlot;

        /** The clip box fit from multiple data frames */
        vislib::math::Cuboid<float> clipbox;

        /** The data hash */
        SIZE_T datahash;

    };

} /* end namespace moldyn */
} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_DATAFILESEQUENCE_H_INCLUDED */
