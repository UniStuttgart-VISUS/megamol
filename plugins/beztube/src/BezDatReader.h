/*
 * BezDatReader.h
 *
 * Copyright (C) 2013 by TU Dresden
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOL_BEZTUBE_BEZDATREADER_H_INCLUDED
#define MEGAMOL_BEZTUBE_BEZDATREADER_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "mmcore/Module.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/misc/BezierCurvesListDataCall.h"
#include "mmcore/param/ParamSlot.h"
#include "vislib/sys/File.h"
#include "vislib/sys/ASCIIFileBuffer.h"
#include "vislib/Array.h"
#include "vislib/Pair.h"


namespace megamol {
namespace beztube {


    /**
     * Loader for BezDat files
     *
     * @remarks The whole file needs to fit into memory and is loaded at the
     *          time it is requested for the first time after the filename
     *          parameter has been changed
     */
    class BezDatReader : public core::Module {
    public:

        /**
         * Answer the name of this module.
         *
         * @return The name of this module.
         */
        static const char *ClassName(void) {
            return "BezierDataReader";
        }

        /**
         * Answer a human readable description of this module.
         *
         * @return A human readable description of this module.
         */
        static const char *Description(void) {
            return "Reads 'BezierCurvesListDataCall' from a file (*.bezdat)";
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
            return true;
        }

        /**
         * Answer the file format file name extensions usually used by files
         * for this loader module, or NULL if there are none. The file name
         * extensions include the periode but no asterix (e. g. '.dat').
         * Multiple extensions are separated by semicolons.
         *
         * @return The file name extenions for data files for this loader
         *         module.
         */
        static const char *FilenameExtensions(void) {
            return ".bezdat;.bezdat2";
        }

        /**
         * Answer the file type name (e. g. "Particle Data")
         *
         * @return The file type name
         */
        static const char *FileTypeName(void) {
            return "Bezier Curve Tubes Data";
        }

        /**
         * Answer the relative name of the parameter slot for the data file
         * name of the data file to load.
         *
         * @return The name of the file name slot
         */
        static const char *FilenameSlotName(void) {
            return "filename";
        }
        /**
         * Performs an file format auto detection check based on the first
         * 'dataSize' bytes of the file.
         *
         * @param data Pointer to the first 'dataSize' bytes of the file to
         *             test
         * @param dataSize The number of valid bytes stored at 'data'
         *
         * @return The confidence if this data file can be loaded with this
         *         module. A value of '1' tells the caller that the data file
         *         can be loaded (as long as the file is not corrupted). A
         *         value of '0' tells that this file cannot be loaded. Any
         *         value inbetween tells that loading this file might fail or
         *         might result in undefined behaviour.
         */
        static float FileFormatAutoDetect(const unsigned char* data, SIZE_T dataSize);

        /** Ctor */
        BezDatReader(void);

        /** Dtor */
        virtual ~BezDatReader(void);

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
         * Asks for the data
         *
         * @param call The calling call
         *
         * @return True on success
         */
        bool getData(megamol::core::Call& call);

        /**
         * Asks for the extent of the data
         *
         * @param call The calling call
         *
         * @return True on success
         */
        bool getExtent(megamol::core::Call& call);

        /** Ensures that the data is loaded, if possible */
        inline void assertData(void);

        /**
         * Loads the data from a file
         *
         * @param filename The file to load from
         */
        void loadData(const vislib::TString& filename);

        /** Clears all data */
        void clear(void);

        /**
         * Loads the data from a binary file in format version 2.0
         *
         * @param file The file to load from; the file pointer points right
         *             behind the file header, before the number of frames
         */
        void loadBinary_2_0(vislib::sys::File& file);

        /**
         * Loads the data from an ASCII file in format version 1.0
         *
         * @param file The file to load from
         */
        void loadASCII_1_0(vislib::sys::ASCIIFileBuffer& file);

        /**
         * Loads the data from an ASCII file in format version 2.0
         *
         * @param file The file to load from
         */
        void loadASCII_2_0(vislib::sys::ASCIIFileBuffer& file);

        /** Slot providing data */
        core::CalleeSlot outDataSlot;

        /** Slot for the file name to load */
        core::param::ParamSlot filenameSlot;

        /** The data hash */
        SIZE_T dataHash;

        /** per frame: Curves, BBoxes */
        vislib::Array<vislib::Pair<
            vislib::Array<core::misc::BezierCurvesListDataCall::Curves>,
            core::BoundingBoxes> > data;

        /** Flag whether or not the data uses static indices */
        bool hasStaticIndices;

    };


} /* end namespace beztube */
} /* end namespace megamol */

#endif /* MEGAMOL_BEZTUBE_BEZDATREADER_H_INCLUDED */
