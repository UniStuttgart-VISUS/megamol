/*
 * SimplifyMesh.h
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

#include "mesh/MeshDataCall.h"

#include <array>
#include <memory>
#include <vector>

namespace megamol {
namespace mesh {
/**
 * Module for simplifying a mesh.
 *
 * @author Alexander Straub
 */
class SimplifyMesh : public core::Module {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static inline const char* ClassName() {
        return "SimplifyMesh";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static inline const char* Description() {
        return "Simplify a triangle mesh to reduce number of vertices and faces";
    }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static inline bool IsAvailable() {
#ifdef WITH_CGAL
        return true;
#else
        return false;
#endif
    }

    /**
     * Global unique ID that can e.g. be used for hash calculation.
     *
     * @return Unique ID
     */
    static inline SIZE_T GUID() {
        return 472520021uLL;
    }

#ifdef WITH_CGAL
    /**
     * Initialises a new instance.
     */
    SimplifyMesh();

    /**
     * Finalises an instance.
     */
    virtual ~SimplifyMesh();
#endif

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

private:
#ifdef WITH_CGAL
    /** Callbacks for performing the computation and exposing data */
    bool getMeshDataCallback(core::Call& call);
    bool getMeshMetaDataCallback(core::Call& call);

    /** Function to start the computation */
    bool compute();

    /** Compute input hash */
    SIZE_T compute_hash(SIZE_T data_hash) const;

    /** The slots for requesting data from this module, i.e., lhs connection */
    megamol::core::CalleeSlot mesh_lhs_slot;

    /** The slots for querying data, i.e., a rhs connection */
    megamol::core::CallerSlot mesh_rhs_slot;

    /** Parameter slots */
    core::param::ParamSlot stop_ratio;

    /** Input */
    struct input_t {
        std::shared_ptr<std::vector<float>> vertices;
        std::shared_ptr<std::vector<float>> normals;
        std::shared_ptr<std::vector<unsigned int>> indices;
    } input;

    SIZE_T input_hash;

    /** Output */
    struct output_t {
        std::shared_ptr<std::vector<float>> vertices;
        std::shared_ptr<std::vector<float>> normals;
        std::shared_ptr<std::vector<unsigned int>> indices;
    } output;
#endif
};
} // namespace mesh
} // namespace megamol
