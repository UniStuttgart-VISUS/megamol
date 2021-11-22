/*
 * 3DUIRenderTaskDataSource.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VISUS).
 * All rights reserved.
 */

#ifndef THREE_DIMENSIONAL_UI_RENDER_TASK_DATA_SOURCE
#define THREE_DIMENSIONAL_UI_RENDER_TASK_DATA_SOURCE

#include <array>
#include <list>
#include <utility>

#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"

#include "mesh/MeshCalls.h"
#include "mesh_gl/AbstractGPURenderTaskDataSource.h"
#include "mesh_gl/GPUMeshCollection.h"
#include "vislib/math/Matrix.h"

#include "mesh/3DInteractionCollection.h"

namespace megamol {
namespace mesh_gl {

class ThreeDimensionalUIRenderTaskDataSource : public AbstractGPURenderTaskDataSource {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName(void) {
        return "UIRenderTaskDataSource";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description(void) {
        return "....TODO...";
    }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable(void) {
        return true;
    }

    ThreeDimensionalUIRenderTaskDataSource();
    ~ThreeDimensionalUIRenderTaskDataSource();

protected:
    /**
     * Implementation of 'Create'.
     *
     * @return 'true' on success, 'false' otherwise.
     */
    virtual bool create(void);

    bool getDataCallback(core::Call& caller);

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

    std::shared_ptr<GPUMaterialCollection> m_material_collection;

    megamol::core::CalleeSlot m_3DInteraction_calleeSlot;
    megamol::core::CallerSlot m_3DInteraction_callerSlot;
    megamol::core::CallerSlot m_glTF_callerSlot;
};

} // namespace mesh_gl
} // namespace megamol

#endif // !THREE_DIMENSIONAL_UI_RENDER_TASK_DATA_SOURCE
