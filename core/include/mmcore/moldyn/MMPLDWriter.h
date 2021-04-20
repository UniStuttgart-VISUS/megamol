/*
 * MMPLDWriter.h
 *
 * Copyright (C) 2010-2021 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_MMPLDWRITER_H_INCLUDED
#define MEGAMOLCORE_MMPLDWRITER_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "mmcore/AbstractDataWriter.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/moldyn/MultiParticleDataCall.h"
#include "mmcore/param/ParamSlot.h"
#include "vislib/sys/File.h"


namespace megamol {
namespace core {
namespace moldyn {


    /**
     * MMPLD (MegaMol Particle List Dump) file writer
     */
    class MMPLDWriter : public AbstractDataWriter {
    public:

        /**
         * Answer the name of this module.
         *
         * @return The name of this module.
         */
        static const char *ClassName(void) {
            return "MMPLDWriter";
        }

        /**
         * Answer a human readable description of this module.
         *
         * @return A human readable description of this module.
         */
        static const char *Description(void) {
            return "MMPLD file writer";
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
        MMPLDWriter(void);

        /** Dtor. */
        virtual ~MMPLDWriter(void);

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
         * The main function
         *
         * @return True on success
         */
        virtual bool run(void);

        /**
         * Function querying the writers capabilities
         *
         * @param call The call to receive the capabilities
         *
         * @return True on success
         */
        virtual bool getCapabilities(DataWriterCtrlCall& call);

    private:

        /**
         * Writes the data of one frame to the file
         *
         * @param file The output data file
         * @param data The data of the current frame
         *
         * @return True on success
         */
        bool writeFrame(vislib::sys::File& file, MultiParticleDataCall& data);

        /** The file name of the file to be written */
        param::ParamSlot filenameSlot;

        /** The file format version to be written */
        param::ParamSlot versionSlot;

        param::ParamSlot startFrameSlot;
        param::ParamSlot endFrameSlot;
        param::ParamSlot subsetSlot;

        /** The slot asking for data */
        CallerSlot dataSlot;

    };

} /* end namespace moldyn */
} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_MMPLDWRITER_H_INCLUDED */
