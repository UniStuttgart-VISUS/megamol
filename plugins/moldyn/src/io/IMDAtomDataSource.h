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

#include "mmcore/CalleeSlot.h"
#include "mmcore/Module.h"
#include "mmcore/param/ParamSlot.h"
#include "vislib/Array.h"
#include "vislib/PtrArray.h"
#include "vislib/RawStorage.h"
#include "vislib/RawStorageWriter.h"
#include "vislib/String.h"
#include "vislib/sys/File.h"


namespace megamol::moldyn::io {


/**
 * Data loader module for the IMD atom data file format (Itap Molecular
 * Dynamic; Stuttgart, Germany).
 *
 * http://www.itap.physik.uni-stuttgart.de/~imd/userguide/output.html
 * header definition:
 * http://www.itap.physik.uni-stuttgart.de/~imd/userguide/header.html
 */
class IMDAtomDataSource : public core::Module {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName() {
        return "IMDAtomData";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description() {
        return "Data source module for IMD atom files.";
    }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable() {
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
    static const char* FilenameExtensions() {
        return ".imd;.imdbin;.crist;.chkpt";
    }

    /**
     * Answer the file name slot name
     *
     * @return The file name slot name
     */
    static const char* FilenameSlotName() {
        return "filename";
    }

    /**
     * Answer the file type name (e. g. "Particle Data")
     *
     * @return The file type name
     */
    static const char* FileTypeName() {
        return "IMD Atom";
    }

    /** Ctor. */
    IMDAtomDataSource();

    /** Dtor. */
    ~IMDAtomDataSource() override;

protected:
    /**
     * Implementation of 'Create'.
     *
     * @return 'true' on success, 'false' otherwise.
     */
    bool create() override;

    /**
     * Implementation of 'Release'.
     */
    void release() override;

private:
    /**
     * Utility struct storing all information from the header
     */
    typedef struct _headerdata_t {
        char format;                             //< 'A', 'B', 'b', 'L', or 'l'
        bool id;                                 //< atom ids present
        bool type;                               //< atom types present
        bool mass;                               //< atom masses present
        int pos;                                 //< dimension of position vectors
        int vel;                                 //< dimension of velocity vectors
        int dat;                                 //< data dimensions
        vislib::Array<vislib::StringA> captions; //< The column captions
    } HeaderData;

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
     * Removes all data
     */
    void clear();

    /**
     * Ensures that the data file is loaded into memory, if possible
     */
    void assertData();

    /**
     * Reads the header of the imd file.
     *
     * @param file The file object to read from
     * @param header The struct receiving the data
     *
     * @return 'true' on success
     */
    bool readHeader(vislib::sys::File& file, HeaderData& header);

    // TODO: document
    // read a value and ADVANCE column
    template<typename T>
    void readToIntColumn(T& reader, bool& fail, unsigned int* column, ...);
    template<typename T>
    void readToFloatColumn(T& reader, bool& fail, unsigned int* column, ...);

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
     * @param dir The writer receiving all directed particle data
     * @param loadDir Flag to activate the use of 'dir'
     * @param splitDir Particles with direction NULL vector will be stored
     *                 in pos and col, while all others will be stored in
     *                 dir if (loadDir==true)
     *
     * @return 'true' on success
     */
    template<typename T>
    bool readData(vislib::sys::File& file, const HeaderData& header, bool loadDir, bool splitDir);

    /**
     * Updates the posX filter data (decrese only!)
     */
    bool posXFilterUpdate(core::param::ParamSlot& slot);

    /** The file name */
    core::param::ParamSlot filenameSlot;

    /** enable bbox */
    core::param::ParamSlot bboxEnabledSlot;

    /** filter boundingbox min */
    core::param::ParamSlot bboxMinSlot;

    /** filter boundingbox max */
    core::param::ParamSlot bboxMaxSlot;

    /** The slot for requesting data */
    core::CalleeSlot getDataSlot;

    /** The global radius */
    core::param::ParamSlot radiusSlot;

    /** The colouring option */
    core::param::ParamSlot colourModeSlot;

    /** The default colour to be used */
    core::param::ParamSlot colourSlot;

    /** The colour column */
    core::param::ParamSlot colourColumnSlot;

    /** Whether or not to automatically calculate the column value range */
    core::param::ParamSlot autoColumnRangeSlot;

    /** The minimum value for the colour mapping of the column */
    core::param::ParamSlot minColumnValSlot;

    /** The maximum value for the colour mapping of the column */
    core::param::ParamSlot maxColumnValSlot;

    /** The type column slot */
    core::param::ParamSlot typeColumnSlot;

    // TODO: Document
    core::param::ParamSlot posXFilterNow;
    core::param::ParamSlot posXFilter;
    core::param::ParamSlot posXMinFilter;
    core::param::ParamSlot posXMaxFilter;

    core::param::ParamSlot splitLoadDiredDataSlot;
    core::param::ParamSlot dirXColNameSlot;
    core::param::ParamSlot dirYColNameSlot;
    core::param::ParamSlot dirZColNameSlot;

    core::param::ParamSlot dircolourModeSlot;
    core::param::ParamSlot dircolourSlot;
    core::param::ParamSlot dircolourColumnSlot;
    core::param::ParamSlot dirautoColumnRangeSlot;
    core::param::ParamSlot dirminColumnValSlot;
    core::param::ParamSlot dirmaxColumnValSlot;
    unsigned char dirdefCol[3];
    core::param::ParamSlot dirradiusSlot;
    core::param::ParamSlot dirNormDirSlot;

    /** The xyz position data */
    //vislib::RawStorage posData;
    vislib::PtrArray<vislib::RawStorage> posData;

    /** The colour data */
    //vislib::RawStorage colData;
    vislib::PtrArray<vislib::RawStorage> colData;

    /** The position bounding box as read from the data file header */
    float headerMinX, headerMinY, headerMinZ, headerMaxX, headerMaxY, headerMaxZ;

    /** The bounding box of positions*/
    float minX, minY, minZ, maxX, maxY, maxZ;

    /** The default colour to be used */
    unsigned char defCol[3];

    /** The bounding values of the colour column */
    vislib::Array<float> minC, maxC;

    /** The hash value of the loaded data */
    SIZE_T datahash;

    /** All data for directional particles */
    //vislib::RawStorage allDirData;
    vislib::PtrArray<vislib::RawStorage> allDirData;

    // TODO: Document
    vislib::Array<unsigned int> typeData;
};

} // namespace megamol::moldyn::io

#endif /* MEGAMOLCORE_IMDATOMDATASOURCE_H_INCLUDED */
