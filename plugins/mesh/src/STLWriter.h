/*
 * implicit_topology_writer.h
 *
 * Copyright (C) 2021 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */
#pragma once

#include "mmcore/Call.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/Module.h"

#include "mmcore/param/ParamSlot.h"

namespace megamol {
namespace mesh {
/**
 * Writer for triangle mesh to STL file.
 *
 * @author Alexander Straub
 */
class STLWriter : public core::Module {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName() {
        return "STLWriter";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description() {
        return "Writer for triangle mesh to STL file.";
    }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable() {
        return true;
    }

    /**
     * Global unique ID that can e.g. be used for hash calculation.
     *
     * @return Unique ID
     */
    static inline SIZE_T GUID() {
        return 903641637uLL;
    }

    /**
     * Constructor
     */
    STLWriter();

    /**
     * Destructor
     */
    ~STLWriter();

protected:
    /**
     * Implementation of 'Create'.
     *
     * @return 'true' on success, 'false' otherwise.
     */
    virtual bool create() override;

    /**
     * Implementation of 'Release'.
     */
    virtual void release() override;

    /** Callbacks for piping through data, and saving data on demand */
    bool getMeshDataCallback(core::Call& call);
    bool getMeshMetaDataCallback(core::Call& call);

    /** Callback to register button pressed */
    bool setButtonPressed(core::param::ParamSlot&);

    /** Function for writing to file */
    bool write(const std::string& filename, const std::vector<float>& vertices, const std::vector<float>& normals,
        const std::vector<unsigned int>& indices) const;

    /** The slots for requesting data from this module, i.e., lhs connection */
    core::CalleeSlot mesh_lhs_slot;

    /** The slots for querying data, i.e., a rhs connection */
    core::CallerSlot mesh_rhs_slot;

    /** Parameters for setting the output file name and triggering the writing process */
    core::param::ParamSlot filename;
    core::param::ParamSlot filetype;
    core::param::ParamSlot save;

    bool triggered;
};
} // namespace mesh
} // namespace megamol
