/*
 * BezDatWriter.h
 *
 * Copyright (C) 2013 by TU Dresden
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOL_BEZTUBE_BEZDATWRITER_H_INCLUDED
#define MEGAMOL_BEZTUBE_BEZDATWRITER_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "mmcore/AbstractDataWriterBase.h"
#include "mmcore/misc/BezierCurvesListDataCall.h"
#include "mmcore/param/ParamSlot.h"
#include "mmcore/param/EnumParam.h"


namespace megamol {
namespace beztube {


    /**
     * Write bezdat data to a file
     */
    class BezDatWriter : public core::AbstractDataWriterBase<core::misc::BezierCurvesListDataCall> {
    public:

        /**
         * Answer the name of this module.
         *
         * @return The name of this module.
         */
        static const char *ClassName(void) {
            return "BezierDataWriter";
        }

        /**
         * Answer a human readable description of this module.
         *
         * @return A human readable description of this module.
         */
        static const char *Description(void) {
            return "Writes 'BezierCurvesListDataCall' to a file";
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

        /** Ctor */
        BezDatWriter(void);

        /** Dtor */
        virtual ~BezDatWriter(void);

    protected:

        /**
         * Starts writing the file
         *
         * @param file The file
         * @param data The data call after calling "GetExtent"
         */
        virtual void writeFileStart(vislib::sys::File& file, core::misc::BezierCurvesListDataCall& data);

        /**
         * Writes one data frame to the file. Frames are written from zero on
         * with continuously increasing frame numbers.
         *
         * @param file The file
         * @param idx The zero-based index of the current frame
         * @param data The data call after calling "GetData"
         */
        virtual void writeFileFrameData(vislib::sys::File& file, unsigned int idx, core::misc::BezierCurvesListDataCall& data);

        /**
         * Finishes writing the file
         *
         * @param file The file
         */
        virtual void writeFileEnd(vislib::sys::File& file);

    private:

        /** base type */
        typedef core::AbstractDataWriterBase<core::misc::BezierCurvesListDataCall> base;

        /** parameter of subfiletypes */
        core::param::ParamSlot fileTypeSlot;

        /** The subfiletype */
        unsigned int fileType;

    };


} /* end namespace beztube */
} /* end namespace megamol */

#endif /* MEGAMOL_BEZTUBE_BEZDATWRITER_H_INCLUDED */
