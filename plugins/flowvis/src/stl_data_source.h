/*
 * stl_data_source.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */
#pragma once

#include "mmcore/Call.h"

#include "mmstd_datatools/io/STLDataSource.h"

namespace megamol {
namespace flowvis {

class stl_data_source : public stdplugin::datatools::io::STLDataSource {
public:
    /// <summary>
    /// Answer the name of this module
    /// </summary>
    /// <returns>Module name</returns>
    static const char* ClassName() { return "stl_data_source"; }

    /// <summary>
    /// Answer a human-readable description
    /// </summary>
    /// <returns>Description</returns>
    static const char* Description() { return "Reader for triangle data in STL files"; }

    /// <summary>
    /// Answer the availability of this module
    /// </summary>
    /// <returns>Availability</returns>
    static bool IsAvailable() { return true; }

    /// <summary>
    /// Constructor
    /// </summary>
    stl_data_source();

    /// <summary>
    /// Destructor
    /// </summary>
    ~stl_data_source();

protected:
    /// <summary>
    /// Callback function for requesting information
    /// </summary>
    /// <param name="caller">Call for this request</param>
    /// <returns>True on success; false otherwise</returns>
    virtual bool get_extent_callback(core::Call& caller) override;

    /// <summary>
    /// Callback function for requesting data
    /// </summary>
    /// <param name="caller">Call for this request</param>
    /// <returns>True on success; false otherwise</returns>
    virtual bool get_mesh_data_callback(core::Call& caller) override;

    /// Output
    core::CalleeSlot ngmesh_output_slot;
};

} // namespace flowvis
} // namespace megamol
