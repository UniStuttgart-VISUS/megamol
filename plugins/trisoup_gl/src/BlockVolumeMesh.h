/*
 * BlockVolumeMesh.h
 *
 * Copyright (C) 2010 by Sebastian Grottel
 * Copyright (C) 2010 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#pragma once

#include "AbstractTriMeshDataSource.h"
#include "mmcore/CallerSlot.h"


namespace megamol::trisoup_gl {


/**
 * Class creating a tri mesh from binary volume showing the blocks opaque
 */
class BlockVolumeMesh : public AbstractTriMeshDataSource {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName() {
        return "BlockVolumeMesh";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description() {
        return "Converts a binary volume into a tri mesh showing opaque blocks.";
    }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable() {
        return true;
    }

    /** Ctor */
    BlockVolumeMesh();

    /** Dtor */
    ~BlockVolumeMesh() override;

protected:
    /** Ensures that the data is loaded */
    void assertData() override;

private:
    /** The in data slot */
    core::CallerSlot inDataSlot;

    /** The incoming data hash */
    SIZE_T inDataHash;
};

} // namespace megamol::trisoup_gl
