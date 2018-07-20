/*
 * QuartzCrystalDataSource.h
 *
 * Copyright (C) 2018 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#pragma once

#include "mmcore/Module.h"
#include "mmcore/param/ParamSlot.h"
#include "mmcore/CalleeSlot.h"
#include "vislib/Array.h"
#include "QuartzCrystalDataCall.h"


namespace megamol {
namespace demos {

    /**
     * Module loading a quartz crystal definition file
     */
    class CrystalDataSource : public megamol::core::Module {
    public:

        /**
         * Answer the name of this module.
         *
         * @return The name of this module.
         */
        static const char *ClassName(void) {
            return "QuartzCrystalDataSource";
        }

        /**
         * Answer a human readable description of this module.
         *
         * @return A human readable description of this module.
         */
        static const char *Description(void) {
            return "Module loading a quartz crystal definition file";
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
        CrystalDataSource(void);

        /** Dtor */
        virtual ~CrystalDataSource(void);

    protected:

        /**
         * Implementation of 'Create'.
         *
         * @return 'true' on success, 'false' otherwise.
         */
        virtual bool create(void);

        /**
         * Call callback to get the data
         *
         * @param c The calling call
         *
         * @return True on success
         */
        bool getData(core::Call& c);

        /**
         * Implementation of 'Release'.
         */
        virtual void release(void);

    private:

        /** The file name slot */
        core::param::ParamSlot filenameSlot;

        /** The data callee slot */
        core::CalleeSlot dataOutSlot;

        /**
         * Loads a BDMD file
         *
         * @param filename The path to the file to load
         */
        void loadFile(const vislib::TString& filename);

        /** The data hash */
        SIZE_T datahash;

        /** The crystals */
        vislib::Array<CrystalDataCall::Crystal> crystals;

    };

} /* end namespace demos */
} /* end namespace megamol */

