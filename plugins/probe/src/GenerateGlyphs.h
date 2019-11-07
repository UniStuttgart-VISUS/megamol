/*
 * GenerateGlyphs.h
 * Copyright (C) 2019 by MegaMol Team
 * Alle Rechte vorbehalten.
 */

#pragma once

#include "ProbeCollection.h"
#include "mesh/MeshDataAccessCollection.h"
#include "mmcore/Call.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/Module.h"
#include "mesh/ImageDataAccessCollection.h"
#include "DrawTextureUtility.h"

namespace megamol {
namespace probe {

class GenerateGlyphs : public core::Module {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName(void) { return "GenerateGlyphs"; }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description(void) { return "Creator for GenerateGlyphs."; }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable(void) { return true; }

    /** Dtor. */
    virtual ~GenerateGlyphs(void);

    /** Ctor. */
    GenerateGlyphs(void);

    core::CalleeSlot _deploy_texture;
    core::CalleeSlot _deploy_mesh;
    core::CallerSlot _get_probes;

protected:

    bool create() override {return true;};
    void release() override {};

private:
    bool getMesh(core::Call& call);
    bool getMetaData(core::Call& call);

    bool getTexture(core::Call& call);
    bool getTextureMetaData(core::Call& call);

    void doGlyphGeneration(float scale);

    uint32_t _data_hash;
    std::shared_ptr<ProbeCollection> _probe_data;
    std::shared_ptr<mesh::MeshDataAccessCollection> _mesh_data;
    std::shared_ptr<mesh::ImageDataAccessCollection> _tex_data;

    std::vector<std::array<float,3>> _generated_mesh;
    std::array<uint32_t,6> _generated_mesh_indices;
    std::array<std::array<float,2>,4> _generated_texture_coordinates;

    std::vector<DrawTextureUtility> _dtu;
};

} // namespace probe
} // namespace megamol
