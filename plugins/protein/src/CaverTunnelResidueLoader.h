/*
 * CaverTunnelResidueLoader.h
 * Copyright (C) 2006-2017 by MegaMol Team
 * Alle Rechte vorbehalten.
 */
#ifndef MMPROTEINPLUGIN_CAVERTUNNELRESIDUELOADER_H_INCLUDED
#define MMPROTEINPLUGIN_CAVERTUNNELRESIDUELOADER_H_INCLUDED
#pragma once

#include "mmcore/Call.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/Module.h"
#include "mmcore/param/ParamSlot.h"
#include "mmstd/data/AnimDataModule.h"

#include "protein_calls/TunnelResidueDataCall.h"

#include "vislib/math/Cuboid.h"
#include "vislib/sys/File.h"

namespace megamol::protein {

/**
 * Data source for the tunnel-parallel residue files from the Caver software
 */
class CaverTunnelResidueLoader : public core::Module {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName() {
        return "CaverTunnelResidueLoader";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description() {
        return "Data source module for tunnel-residing residue index files outputted by Caver";
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
    CaverTunnelResidueLoader();

    /** Dtor. */
    ~CaverTunnelResidueLoader() override;

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

    /**
     * Splits a line into different parts, seperated by a given char
     *
     * @param line The line to split
     * @param splitChar The char to split after. Default: whitespace
     * @return Vector containing all parts of the line. May be empty, when the line only contains the splitChar or was
     * empty before.
     */
    std::vector<vislib::StringA> splitLine(vislib::StringA line, char splitChar = ' ');

    /** Slot for the filename */
    core::param::ParamSlot filenameSlot;

    /** Slot for the filename containing the tunnel vertices */
    core::param::ParamSlot tunnelFilenameSlot;

    /** The data output callee slot */
    core::CalleeSlot getData;

    /** The file handle */
    vislib::sys::File* file;

    /** The tunnel file handle */
    vislib::sys::File* tunnelFile;

    /** The data hash */
    size_t data_hash;

    /** data storage for all read tunnels */
    std::vector<protein_calls::TunnelResidueDataCall::Tunnel> tunnelVector;

    /** The bounding box of the tunnel voronoi vertices */
    vislib::math::Cuboid<float> boundingBox;
};

} // namespace megamol::protein

#endif
