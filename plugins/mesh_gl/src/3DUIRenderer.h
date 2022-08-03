/*
 * 3DUIRenderer.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VISUS).
 * All rights reserved.
 */

#ifndef THREE_DIMENSIONAL_UI_RENDERER_H_INCLUDED
#define THREE_DIMENSIONAL_UI_RENDERER_H_INCLUDED

#include <array>
#include <list>
#include <utility>

#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"

#include "mesh/MeshCalls.h"
#include "mesh_gl/BaseMeshRenderer.h"
#include "vislib/math/Matrix.h"

#include "mesh/3DInteractionCollection.h"

namespace megamol {
namespace mesh_gl {

class ThreeDimensionalUIRenderer : public BaseMeshRenderer {
public:
    ThreeDimensionalUIRenderer();
    ~ThreeDimensionalUIRenderer();

    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName(void) {
        return "3DUIRenderer";
    }
    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description(void) {
        return "....TODO...";
    }

protected:
    void createMaterialCollection() override;

    void updateRenderTaskCollection(mmstd_gl::CallRender3DGL& call, bool force_update) override;

    bool getInteractionCallback(core::Call& caller);

private:
    struct PerObjectShaderParams {
        vislib::math::Matrix<GLfloat, 4, vislib::math::COLUMN_MAJOR> object_transform;

        std::array<float, 4> color;

        int id;

        int highlighted;

        float padding0;
        float padding1;
    };

    uint32_t m_version;

    std::array<std::pair<GPUMeshCollection::SubMeshData, std::array<PerObjectShaderParams, 1>>, 4>
        m_UI_template_elements;

    std::list<std::pair<std::string, std::array<PerObjectShaderParams, 1>>> m_scene;

    std::shared_ptr<mesh::ThreeDimensionalInteractionCollection> m_interaction_collection;

    megamol::core::CalleeSlot m_3DInteraction_calleeSlot;
    megamol::core::CallerSlot m_3DInteraction_callerSlot;
    megamol::core::CallerSlot m_glTF_callerSlot;
};

} // namespace mesh_gl
} // namespace megamol

#endif // !THREE_DIMENSIONAL_UI_RENDERER_H_INCLUDED
