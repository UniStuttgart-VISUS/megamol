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
#include "CallDescriptionManager.h"
#include "CalleeSlot.h"
#include "CallerSlot.h"
#include "mmcore/param/ParamSlot.h"
#include "vislib/math/Cuboid.h"


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

        /**
         * Disallow usage in quickstarts
         *
         * @return false
         */
        static bool SupportQuickstart(void) {
            return false;
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

        /**
         * Moves the iterator directly behind the next description of the
         * next call compatible with this module
         *
         * @param iterator The iterator to iterate
         *
         * @return The call description iterated to, or NULL if there are no
         *         more compatible calls
         */
        const CallDescription* moveToNextCompatibleCall(
            CallDescriptionManager::DescriptionIterator &iterator) const;

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

        /**
         * Checks if the callee and the caller slot are connected with the
         * same call classes
         *
         * @param outCall The incoming call requesting data
         *
         * @return True if everything is fine.
         */
        bool checkConnections(Call *outCall);

        /**
         * Checks the module parameters for updates
         */
        void checkParameters(void);

        /**
         * Update when the file name template changes
         *
         * @param slot Must be 'fileNameTemplateSlot'
         *
         * @return true
         */
        bool onFileNameTemplateChanged(param::ParamSlot& slot);

        /**
         * Update when the file name slot name changes
         *
         * @param slot Must be 'fileNameSlotNameSlot'
         *
         * @return true
         */
        bool onFileNameSlotNameChanged(param::ParamSlot& slot);

        /**
         * Finds the parameter slot for the file name
         *
         * @return The parameter slot for the file name or NULL if not found
         */
        param::ParamSlot *findFileNameSlot(void);

        /**
         * Asserts the data is available blablabla
         */
        void assertData(void);

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

        /** Flag controlling the bounding box */
        param::ParamSlot useClipBoxAsBBox;

        /** The slot for publishing data to the writer */
        CalleeSlot outDataSlot;

        /** The slot for requesting data from the source */
        CallerSlot inDataSlot;

        /** The clip box fit from multiple data frames */
        vislib::math::Cuboid<float> clipbox;

        /** The data hash */
        SIZE_T datahash;

        /** The file name template */
        vislib::TString fileNameTemplate;

        /** The minimum file number */
        unsigned int fileNumMin;

        /** The maximum file number */
        unsigned int fileNumMax;

        /** The file number increase step */
        unsigned int fileNumStep;

        /** Needs to update the data */
        bool needDataUpdate;

        /** The actual number of frames available */
        unsigned int frameCnt;

        /** The last frame index requested */
        unsigned int lastIdxRequested;

    };

} /* end namespace moldyn */
} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_DATAFILESEQUENCE_H_INCLUDED */
