/*
 * SIFFWriter.h
 *
 * Copyright (C) 2013 by TU Dresden
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_SIFFWRITER_H_INCLUDED
#define MEGAMOLCORE_SIFFWRITER_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "AbstractDataWriter.h"
#include "CallerSlot.h"
#include "moldyn/MultiParticleDataCall.h"
#include "param/ParamSlot.h"
#include "vislib/File.h"


namespace megamol {
namespace core {
namespace moldyn {

    /**
     * SIFF writer module
     */
    class SIFFWriter : public AbstractDataWriter {
    public:

        /**
         * Answer the name of this module.
         *
         * @return The name of this module.
         */
        static const char *ClassName(void) {
            return "SIFFWriter";
        }

        /**
         * Answer a human readable description of this module.
         *
         * @return A human readable description of this module.
         */
        static const char *Description(void) {
            return "Writing SIFF";
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
        SIFFWriter(void);

        /** Dtor. */
        virtual ~SIFFWriter(void);

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

        /** The file name of the file to be written */
        param::ParamSlot filenameSlot;

        /** The slot asking for data */
        param::ParamSlot asciiSlot;

        /** The slot asking for data */
        param::ParamSlot versionSlot;

        /** The slot asking for data */
        CallerSlot dataSlot;

    };


} /* end namespace moldyn */
} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_SIFFWRITER_H_INCLUDED */
