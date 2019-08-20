/*
 * CSVFileSequence.h
 *
 * Copyright (C) 2019 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#pragma once

#include "mmcore/Module.h"
#include "mmcore/factories/CallDescription.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/param/ParamSlot.h"
#include "vislib/math/Cuboid.h"


namespace megamol {
namespace stdplugin {
namespace datatools {


    /**
     * In-Between management module for seralize multiple data files with one
     * data frame each (the first if more are available) to write a continous
     * data series into a data writer
     */
    class CSVFileSequence : public core::Module {
    public:

        /**
         * Answer the name of this module.
         *
         * @return The name of this module.
         */
        static const char *ClassName(void) {
            return "CSVFileSequence";
        }

        /**
         * Answer a human readable description of this module.
         *
         * @return A human readable description of this module.
         */
        static const char *Description(void) {
            return "This modules is pluged between a data loader and consumer modules. It will change the name of the data file loaded depending on the requested time. This module only supports time independent data sets.";
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
        CSVFileSequence(void);

        /** Dtor. */
        virtual ~CSVFileSequence(void);

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
         * Tests if th description of a call seems compatible
         *
         * @param desc The description to test
         *
         * @return True if description seems compatible
         */
        static bool IsCallDescriptionCompatible(core::factories::CallDescription::ptr desc);

        /**
         * Gets the data from the source.
         *
         * @param caller The calling call.
         *
         * @return 'true' on success, 'false' on failure.
         */
        bool getDataCallback(core::Call& caller);

        /**
         * Gets the data from the source.
         *
         * @param caller The calling call.
         *
         * @return 'true' on success, 'false' on failure.
         */
        bool getExtentCallback(core::Call& caller);

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
        bool onFileNameTemplateChanged(core::param::ParamSlot& slot);

        /**
         * Update when the file name slot name changes
         *
         * @param slot Must be 'fileNameSlotNameSlot'
         *
         * @return true
         */
        bool onFileNameSlotNameChanged(core::param::ParamSlot& slot);

        /**
         * Finds the parameter slot for the file name
         *
         * @return The parameter slot for the file name or NULL if not found
         */
        core::param::ParamSlot *findFileNameSlot(void);

        /**
         * Asserts the data is available blablabla
         */
        void assertData(void);

        /** The file name template */
        core::param::ParamSlot fileNameTemplateSlot;

        /** Slot for the minimum file number */
        core::param::ParamSlot fileNumberMinSlot;

        /** Slot for the maximum file number */
        core::param::ParamSlot fileNumberMaxSlot;

        /** Slot for the file number increase step */
        core::param::ParamSlot fileNumberStepSlot;

        /** The name of the data source file name parameter slot */
        core::param::ParamSlot fileNameSlotNameSlot;

        /** Flag controlling the bounding box */
        core::param::ParamSlot useClipBoxAsBBox;

        /** The slot for publishing data to the writer */
        core::CalleeSlot outDataSlot;

        /** The slot for requesting data from the source */
        core::CallerSlot inDataSlot;

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

} /* end namespace datatools */
} /* end namespace stdplugin */
} /* end namespace megamol */
