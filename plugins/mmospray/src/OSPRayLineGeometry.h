/*
 * OSPRayLineGeometry.h
 * Copyright (C) 2009-2017 by MegaMol Team
 * Alle Rechte vorbehalten.
 */
#pragma once

#include "mmcore/CallerSlot.h"
#include "mmcore/param/ParamSlot.h"
#include "mmospray/AbstractOSPRayStructure.h"

namespace megamol {
namespace ospray {

class OSPRayLineGeometry : public AbstractOSPRayStructure {

public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName(void) {
        return "OSPRayLineGeometry";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description(void) {
        return "Creator for OSPRay Line Geometry.";
    }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable(void) {
        return true;
    }

    /** Dtor. */
    virtual ~OSPRayLineGeometry(void);

    /** Ctor. */
    OSPRayLineGeometry(void);

protected:
    virtual bool create();
    virtual void release();

    virtual bool readData(core::Call& call);
    virtual bool getExtends(core::Call& call);


private:
    /** detects interface dirtyness */
    bool InterfaceIsDirty();

    /** The call for data */
    core::CallerSlot getDataSlot;
    core::CallerSlot getLineDataSlot;

    core::param::ParamSlot globalRadiusSlot;
    core::param::ParamSlot representationSlot;

    std::shared_ptr<mesh::MeshDataAccessCollection> _converted_data;
    std::vector<std::vector<mesh::MeshDataAccessCollection::VertexAttribute>> _converted_attribs;
    std::vector<mesh::MeshDataAccessCollection::IndexData> _converted_indices;

    std::vector<std::vector<uint32_t>> _converted_index;
    std::vector<std::vector<std::array<float,3>>> _converted_vertices;
    
};

} // namespace ospray
} // namespace megamol
