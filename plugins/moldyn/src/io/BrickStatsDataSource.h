/*
 * BrickStatsDataSource.h
 *
 * Copyright (C) 2016 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#pragma once

#include "mmcore/CalleeSlot.h"
#include "mmcore/Module.h"
#include "mmcore/param/ParamSlot.h"
#include "moldyn/BrickStatsCall.h"
#include "vislib/RawStorage.h"
#include "vislib/math/Cuboid.h"
#include "vislib/sys/File.h"
#include "vislib/types.h"


namespace megamol::moldyn {

using namespace megamol::core;


/**
 * Data source module for MMPLD files
 */
class BrickStatsDataSource : public Module {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName() {
        return "PTSBrickStatsDataSource";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description() {
        return "Data source module for brick statistics files.";
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
    BrickStatsDataSource();

    /** Dtor. */
    ~BrickStatsDataSource() override;

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

    /**
     * Callback receiving the update of the file name parameter.
     *
     * @param slot The updated ParamSlot.
     *
     * @return Always 'true' to reset the dirty flag.
     */
    bool filenameChanged(param::ParamSlot& slot);

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

private:
    /** The file name */
    param::ParamSlot filename;

    param::ParamSlot skipHeaderLine;

    /** The slot for requesting data */
    CalleeSlot getData;

    /** The opened data file */
    vislib::sys::File* file;

    /** The data set bounding box */
    vislib::math::Cuboid<float> bbox;

    /** The data set clipping box */
    vislib::math::Cuboid<float> clipbox;

    /** Data file load id counter */
    size_t data_hash;

    vislib::Array<BrickStatsCall::BrickInfo> info;
};

} // namespace megamol::moldyn
