/*
 * TriMeshSTLWriter.h
 *
 * Copyright (C) 2018 by MegaMol Team
 * Alle Rechte vorbehalten.
 */
#ifndef MEGAMOL_DATATOOLS_IO_TRIMESHSTLWRITER_H_INCLUDED
#define MEGAMOL_DATATOOLS_IO_TRIMESHSTLWRITER_H_INCLUDED
#pragma once

#include "datatools/io/AbstractSTLWriter.h"

#include "mmstd/data/AbstractGetData3DCall.h"

#include "geometry_calls_gl/CallTriMeshDataGL.h"

namespace megamol {
namespace datatools_gl {
namespace io {
/// <summary>
/// Triangle STL writer for CallTriMeshData calls
/// </summary>
class TriMeshSTLWriter : public datatools::io::AbstractSTLWriter<geocalls_gl::CallTriMeshDataGLDescription> {
public:
    /// <summary>
    /// Answer the name of this module
    /// </summary>
    /// <returns>Module name</returns>
    static const char* ClassName() {
        return "TriMeshSTLWriter";
    }

    /// <summary>
    /// Answer a human-readable description
    /// </summary>
    /// <returns>Description</returns>
    static const char* Description() {
        return "Writer for triangle data in STL files";
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
    TriMeshSTLWriter();

    /// <summary>
    /// Destructor
    /// </summary>
    ~TriMeshSTLWriter() override;

protected:
    /// <summary>
    /// Create the module
    /// </summary>
    /// <returns>True on success; false otherwise</returns>
    bool create() override;

    /// <summary>
    /// Copy information from the incoming to the outgoing call
    /// </summary>
    /// <param name="caller">Incoming call</param>
    /// <param name="callee">Outgoing call</param>
    /// <returns>True on success; false otherwise</returns>
    bool copy_info_upstream(core::AbstractGetData3DCall& caller, core::AbstractGetData3DCall& callee) override;

    /// <summary>
    /// Copy information from the outgoing to the incoming call
    /// </summary>
    /// <param name="caller">Incoming call</param>
    /// <param name="callee">Outgoing call</param>
    /// <returns>True on success; false otherwise</returns>
    bool copy_info_downstream(core::AbstractGetData3DCall& caller, core::AbstractGetData3DCall& callee) override;

    /// <summary>
    /// Copy data to incoming call
    /// </summary>
    /// <param name="caller">Incoming call</param>
    /// <param name="callee">Outgoing call</param>
    /// <returns>True on success; false otherwise</returns>
    bool copy_data(core::AbstractGetData3DCall& caller, core::AbstractGetData3DCall& callee) override;

    /// <summary>
    /// Write data from outgoing call to file
    /// </summary>
    /// <param name="callee">Outgoing call</param>
    /// <returns>True on success; false otherwise</returns>
    bool write_data(core::AbstractGetData3DCall& callee) override;

    /// <summary>
    /// Release the module
    /// </summary>
    void release() override;
};
} // namespace io
} // namespace datatools_gl
} // namespace megamol

#endif // !MEGAMOL_DATATOOLS_IO_TRIMESHSTLWRITER_H_INCLUDED
