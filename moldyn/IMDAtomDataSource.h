/*
 * IMDAtomDataSource.h
 *
 * Copyright (C) 2009 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_IMDATOMDATASOURCE_H_INCLUDED
#define MEGAMOLCORE_IMDATOMDATASOURCE_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "Module.h"
#include "param/ParamSlot.h"
#include "CalleeSlot.h"
#include "vislib/Array.h"
#include "vislib/File.h"
#include "vislib/RawStorage.h"
#include "vislib/RawStorageWriter.h"
#include "vislib/String.h"


namespace megamol {
namespace core {
namespace moldyn {


    /**
     * Data loader module for the IMD atom data file format (Itap Molecular
     * Dynamic; Stuttgart, Germany).
     *
     * http://www.itap.physik.uni-stuttgart.de/~imd/userguide/output.html
     * header definition:
     * http://www.itap.physik.uni-stuttgart.de/~imd/userguide/header.html
     */
    class IMDAtomDataSource : public Module {
    public:

        /**
         * Answer the name of this module.
         *
         * @return The name of this module.
         */
        static const char *ClassName(void) {
            return "IMDAtomData";
        }

        /**
         * Answer a human readable description of this module.
         *
         * @return A human readable description of this module.
         */
        static const char *Description(void) {
            return "Data source module for IMD atom files.";
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
         * Tests if a file can be loaded with this module
         *
         * @param data The data to test
         * @param dataSize The size of the data to test
         *
         * @return The loading confidence value
         */
        static float FileFormatAutoDetect(const unsigned char* data, SIZE_T dataSize);

        /**
         * Answer the file name extensions often used
         *
         * @return The file name extensions
         */
        static const char *FilenameExtensions() {
            return ".imd;.crist;.chkpt";
        }

        /**
         * Answer the file name slot name
         *
         * @return The file name slot name
         */
        static const char *FilenameSlotName(void) {
            return "filename";
        }

        /**
         * Answer the file type name (e. g. "Particle Data")
         *
         * @return The file type name
         */
        static const char *FileTypeName(void) {
            return "IMD Atom";
        }

        /** Ctor. */
        IMDAtomDataSource(void);

        /** Dtor. */
        virtual ~IMDAtomDataSource(void);

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
         * Utility struct storing all information from the header
         */
        typedef struct _headerdata_t {
            char format;    //< 'A', 'B', 'b', 'L', or 'l'
            bool id;        //< atom ids present
            bool type;      //< atom types present
            bool mass;      //< atom masses present
            int pos;        //< dimension of position vectors
            int vel;        //< dimension of velocity vectors
            int dat;        //< data dimensions
            vislib::Array<vislib::StringA> captions; //< The column captions
        } HeaderData;

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
         * Removes all data
         */
        void clear(void);

        /**
         * Ensures that the data file is loaded into memory, if possible
         */
        void assertData(void);

        /**
         * Reads the header of the imd file.
         *
         * @param file The file object to read from
         * @param header The struct receiving the data
         *
         * @return 'true' on success
         */
        bool readHeader(vislib::sys::File& file, HeaderData& header);

        /**
         * Reads the data of the imd file. This method also calculated the
         * data bounding box and sets the corresponding members.
         *
         * Use a imdinternal::AtomReader* class as template type.
         *
         * @param file The file object to read from
         * @param header The struct holding the header data
         * @param pos The writer receiving the read position data
         * @param col The writer receiving the read colour data
         *
         * @return 'true' on success
         */
        template<typename T> bool readData(vislib::sys::File& file,
            const HeaderData& header, vislib::RawStorageWriter& pos,
            vislib::RawStorageWriter& col);

        /** The file name */
        param::ParamSlot filenameSlot;

        /** The slot for requesting data */
        CalleeSlot getDataSlot;

        /** The global radius */
        param::ParamSlot radiusSlot;

        /** The colouring option */
        param::ParamSlot colourModeSlot;

        /** The default colour to be used */
        param::ParamSlot colourSlot;

        /** The colour column */
        param::ParamSlot colourColumnSlot;

        /** Whether or not to automatically calculate the column value range */
        param::ParamSlot autoColumnRangeSlot;

        /** The minimum value for the colour mapping of the column */
        param::ParamSlot minColumnValSlot;

        /** The maximum value for the colour mapping of the column */
        param::ParamSlot maxColumnValSlot;

        /** The xyz position data */
        vislib::RawStorage posData;

        /** The colour data */
        vislib::RawStorage colData;

        /** The position bounding box as read from the data file header */
        float headerMinX, headerMinY, headerMinZ, headerMaxX, headerMaxY, headerMaxZ;

        /** The bounding box of positions*/
        float minX, minY, minZ, maxX, maxY, maxZ;

        /** The default colour to be used */
        unsigned char defCol[3];

        /** The bounding values of the colour column */
        float minC, maxC;

        /** The hash value of the loaded data */
        SIZE_T datahash;

    };

} /* end namespace moldyn */
} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_IMDATOMDATASOURCE_H_INCLUDED */
