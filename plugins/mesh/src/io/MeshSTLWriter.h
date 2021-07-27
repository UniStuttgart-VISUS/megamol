#pragma once

#include "mmstd_datatools/io/AbstractSTLWriter.h"

#include "mmcore/AbstractGetData3DCall.h"

#include "mesh/MeshCalls.h"

namespace megamol::mesh::io {

/// <summary>
/// Triangle STL writer for CallTriMeshData calls
/// </summary>
class MeshSTLWriter : public stdplugin::datatools::io::AbstractSTLWriter<CallMesh, CallMeshDescription> {
public:
    /// <summary>
    /// Answer the name of this module
    /// </summary>
    /// <returns>Module name</returns>
    static const char* ClassName() {
        return "MeshSTLWriter";
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
    MeshSTLWriter();

    /// <summary>
    /// Destructor
    /// </summary>
    ~MeshSTLWriter();

protected:
    /// <summary>
    /// Create the module
    /// </summary>
    /// <returns>True on success; false otherwise</returns>
    virtual bool create() override;

    /// <summary>
    /// Copy information from the incoming to the outgoing call
    /// </summary>
    /// <param name="caller">Incoming call</param>
    /// <param name="callee">Outgoing call</param>
    /// <returns>True on success; false otherwise</returns>
    virtual bool copy_info_upstream(core::Call& caller, core::Call& callee) override;

    /// <summary>
    /// Copy information from the outgoing to the incoming call
    /// </summary>
    /// <param name="caller">Incoming call</param>
    /// <param name="callee">Outgoing call</param>
    /// <returns>True on success; false otherwise</returns>
    virtual bool copy_info_downstream(
        core::Call& caller, core::Call& callee) override;

    virtual bool check_update(core::Call& caller, core::Call& callee) override;

    /// <summary>
    /// Copy data to incoming call
    /// </summary>
    /// <param name="caller">Incoming call</param>
    /// <param name="callee">Outgoing call</param>
    /// <returns>True on success; false otherwise</returns>
    virtual bool copy_data(core::Call& caller, core::Call& callee) override;

    /// <summary>
    /// Write data from outgoing call to file
    /// </summary>
    /// <param name="callee">Outgoing call</param>
    /// <returns>True on success; false otherwise</returns>
    virtual bool write_data(core::Call& callee) override;

    /// <summary>
    /// Release the module
    /// </summary>
    virtual void release() override;
};
} // namespace megamol::mesh::io
