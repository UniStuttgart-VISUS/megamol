/*
 * MMFTDataWriter.h
 *
 * Copyright (C) 2016 by CGV (TU Dresden)
 * Alle Rechte vorbehalten.
 */
#ifndef MEGAMOL_DATATOOLS_MMFTDATAWRITER_H_INCLUDED
#define MEGAMOL_DATATOOLS_MMFTDATAWRITER_H_INCLUDED
#pragma once

#include "mmcore/AbstractDataWriter.h"
#include "mmcore/CallerSlot.h"
#include "mmstd_datatools/floattable/CallFloatTableData.h"
#include "mmcore/param/ParamSlot.h"


namespace megamol {
namespace stdplugin {
namespace datatools {
namespace floattable {


    /**
     * MMFTDataWriter (MegaMol Particle List Dump) file writer
     */
    class MMFTDataWriter : public core::AbstractDataWriter {
    public:

        /**
         * Answer the name of this module.
         *
         * @return The name of this module.
         */
        static const char *ClassName(void) {
            return "MMFTDataWriter";
        }

        /**
         * Answer a human readable description of this module.
         *
         * @return A human readable description of this module.
         */
        static const char *Description(void) {
            return "Binary float table data file writer";
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
        MMFTDataWriter(void);

        /** Dtor. */
        virtual ~MMFTDataWriter(void);

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
        virtual bool getCapabilities(core::DataWriterCtrlCall& call);

    private:

        /** The file name of the file to be written */
        core::param::ParamSlot filenameSlot;

        /** The slot asking for data */
        core::CallerSlot dataSlot;

    };

} /* end namespace floattable */
} /* end namespace datatools */
} /* end namespace stdplugin */
} /* end namespace megamol */

#endif /* MEGAMOL_DATATOOLS_MMFTDATAWRITER_H_INCLUDED */
