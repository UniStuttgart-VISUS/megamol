/*
 * ElementColoring.h
 * Copyright (C) 2021 by MegaMol Team
 * Alle Rechte vorbehalten.
 */


#pragma once

#include "glm/glm.hpp"

#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/Module.h"

#include "mesh/MeshCalls.h"
#include "mmcore/param/ParamSlot.h"
#include "probe/ProbeCollection.h"


namespace megamol {
namespace probe {


class ElementColoring : public core::Module {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName() {
        return "ElementColoring";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description() {
        return "...";
    }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable(void) {
        return true;
    }

    ElementColoring();
    virtual ~ElementColoring();

protected:
    virtual bool create();
    virtual void release();

    uint32_t _version;

    core::param::ParamSlot _fallbackColor;

    core::CallerSlot _probe_rhs_slot;
    core::CallerSlot _elements_rhs_slot;
    core::CalleeSlot _mesh_lhs_slot;

private:
    bool getData(core::Call& call);
    bool getMetaData(core::Call& call);
    bool parameterChanged(core::param::ParamSlot& p);

    std::vector<std::string> split(std::string, const char);
    std::vector<std::vector<std::vector<std::array<float, 4>>>> _colors;
    std::vector<std::array<float, 4>> _vertColors;
    glm::vec3 hsvSpiralColor(int color_idx, int total_colors);
    glm::vec3 hsv2rgb(glm::vec3 c);

    mesh::MeshDataAccessCollection _mesh_collection_copy;
    std::vector<mesh::MeshDataAccessCollection::Mesh> _mesh_copy;

    bool _recalc;
};

} // namespace probe
} // namespace megamol
