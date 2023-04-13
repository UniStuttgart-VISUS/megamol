/*
 * MeshSTLDataSource.h
 *
 * Copyright (C) 2019-2022 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */
#pragma once

#include "mmcore/Call.h"

#include "datatools_gl/io/STLDataSource.h"

namespace megamol::mesh_gl {

class MeshSTLDataSource : public datatools_gl::io::STLDataSource {
public:
    /// <summary>
    /// Answer the name of this module
    /// </summary>
    /// <returns>Module name</returns>
    static const char* ClassName() {
        return "MeshSTLDataSource";
    }

    /// <summary>
    /// Answer a human-readable description
    /// </summary>
    /// <returns>Description</returns>
    static const char* Description() {
        return "Reader for triangle data in STL files";
    }

    /// <summary>
    /// Answer the availability of this module
    /// </summary>
    /// <returns>Availability</returns>
    static bool IsAvailable() {
        return true;
    }

    /// <summary>
    /// Constructor
    /// </summary>
    MeshSTLDataSource();

    /// <summary>
    /// Destructor
    /// </summary>
    ~MeshSTLDataSource() override;

protected:
    /// <summary>
    /// Callback function for requesting information
    /// </summary>
    /// <param name="caller">Call for this request</param>
    /// <returns>True on success; false otherwise</returns>
    bool get_extent_callback(core::Call& caller) override;

    /// <summary>
    /// Callback function for requesting data
    /// </summary>
    /// <param name="caller">Call for this request</param>
    /// <returns>True on success; false otherwise</returns>
    bool get_mesh_data_callback(core::Call& caller) override;

    /// Output
    core::CalleeSlot ngmesh_output_slot;
};

} // namespace megamol::mesh_gl
