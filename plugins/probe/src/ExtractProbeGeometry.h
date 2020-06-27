/*
 * ExtractProbeGeometry.h
 * Copyright (C) 2019 by MegaMol Team
 * Alle Rechte vorbehalten.
 */

#pragma once

#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/Module.h"
#include "mmcore/param/ParamSlot.h"

#include "ProbeCollection.h"
#include "mesh/MeshDataAccessCollection.h"

namespace megamol {
namespace probe {

class ExtractProbeGeometry : public core::Module {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName() { return "ExtractProbeGeometry"; }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description() { return "..."; }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable(void) { return true; }

    /** Ctor. */
    ExtractProbeGeometry();

    /** Dtor. */
    virtual ~ExtractProbeGeometry();

protected:
    virtual bool create();
    virtual void release();

    core::CalleeSlot m_mesh_slot;
    core::CallerSlot m_probe_slot;
    
private:
    std::shared_ptr<mesh::MeshDataAccessCollection> convertToLine(core::Call& call);
    bool getData(core::Call& call);

    bool getMetaData(core::Call& call);

    uint32_t _version;

    std::shared_ptr<ProbeCollection> _probes;

    std::shared_ptr<mesh::MeshDataAccessCollection> _line;
    std::vector<std::vector<mesh::MeshDataAccessCollection::VertexAttribute>> _line_attribs;
    mesh::MeshDataAccessCollection::IndexData _line_indices;

    std::vector<std::array<float, 4>> _vertex_data;
    std::array<uint32_t, 1> _index_data;
};


} // namespace probe
} // namespace megamol
