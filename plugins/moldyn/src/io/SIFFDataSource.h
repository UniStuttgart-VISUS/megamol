/*
 * SIFFDataSource.h
 *
 * Copyright (C) 2009 by Universitaet Stuttgart (VISUS).
 * Alle Rechte vorbehalten.
 */

#pragma once

#include "geometry_calls/MultiParticleDataCall.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/Module.h"
#include "mmcore/param/ParamSlot.h"
#include "vislib/RawStorage.h"
#include "vislib/math/Cuboid.h"


namespace megamol::moldyn::io {


/**
 * Data source for the SImple File Format
 *
 * File Format:
 *  Header (5+ Bytes):
 *      0..3    char*   Header-ID   "SIFF"
 *      4       char    Type        "b" for binary, "a" for ascii
 *    if binary:
 *      5..8    uint32  Version     100 (version 1.0)
 *    if ascii:
 *      x       till-NL Version     1.0 (or equivalent)
 *
 *  Body (if binary - 19 Bytes per Sphere):
 *      0..15   4xfloat Position and radius (X, Y, Z, Rad)
 *      16..18  3xbyte  Colour (RGB)
 *
 *  Body (id ascii - 1 line per sphere):
 *      x       till-NL Position (3xFloat), Radius (1xFloat), Colour (3xInt[0..255])
 *
 *  In version 1.1 (101) only position information (X, Y, Z) is written!
 */
class SIFFDataSource : public core::Module {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName() {
        return "SIFFDataSource";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description() {
        return "SImple File Format Data source module";
    }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable() {
        return true;
    }

    /** Ctor. */
    SIFFDataSource();

    /** Dtor. */
    ~SIFFDataSource() override;

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
     * Callback receiving the update of the file name parameter.
     *
     * @param slot The updated ParamSlot.
     *
     * @return Always 'true' to reset the dirty flag.
     */
    bool filenameChanged(core::param::ParamSlot& slot);

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

    /** The file name */
    core::param::ParamSlot filenameSlot;

    /** The radius used when loading a version 1.1 file */
    core::param::ParamSlot radSlot;

    /** The slot for requesting data */
    core::CalleeSlot getDataSlot;

    /** The bounding box */
    vislib::math::Cuboid<float> bbox;

    /** The data */
    vislib::RawStorage data;

    /** The data hash */
    SIZE_T datahash;

    /* The siff data version */
    unsigned int verNum;

    /** Flag whether or not the data also stores color alpha */
    bool hasAlpha;
};

} // namespace megamol::moldyn::io
