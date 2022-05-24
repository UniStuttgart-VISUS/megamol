#pragma once

#include "CallClustering_2.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/param/ParamSlot.h"
#include "mmcore_gl/view/CallRender2DGL.h"
#include "mmcore_gl/view/Renderer2DModuleGL.h"

#include <glowl/BufferObject.hpp>
#include <glowl/GLSLProgram.hpp>

namespace megamol {
namespace molsurfmapcluster_gl {

class ClusterGraphRenderer : public core_gl::view::Renderer2DModuleGL {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName(void) {
        return "ClusterGraphRenderer";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description(void) {
        return "Render the Graph of a given Clustering";
    }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable(void) {
        return true;
    }

    /** Ctor. */
    ClusterGraphRenderer(void);

    /** Dtor. */
    virtual ~ClusterGraphRenderer(void);

    /** The mouse button pressed/released callback. */
    virtual bool OnMouseButton(megamol::core::view::MouseButton button, megamol::core::view::MouseButtonAction action,
        megamol::core::view::Modifiers mods) override;

    /** The mouse movement callback. */
    virtual bool OnMouseMove(double x, double y) override;

protected:
    /**
     * Implementation of 'Create'.
     *
     * @return 'true' on success, 'false' otherwise.
     */
    virtual bool create(void) override;

    /**
     * Implementation of 'Release'.
     */
    virtual void release(void) override;

    /**
     * The get extents callback. The module should set the members of
     * 'call' to tell the caller the extents of its data (bounding boxes
     * and times).
     */
    virtual bool GetExtents(core_gl::view::CallRender2DGL& call) override;

    /**
     * The render callback.
     */
    virtual bool Render(core_gl::view::CallRender2DGL& call) override;

private:
    void calculateNodePositions(ClusteringData const& cluster_data);

    void calculateNodeConnections(ClusteringData const& cluster_data);

    void uploadDataToGPU();

    std::vector<int64_t> getLeaveIndicesInDFSOrder(
        int64_t const start_idx, std::vector<ClusterNode_2> const& nodes) const;

    /** Slot for the cluster data */
    core::CallerSlot cluster_data_slot_;

    /** Parameter setting the height of the used viewport */
    core::param::ParamSlot viewport_height_param_;

    /** Parameter setting the width of the used viewport */
    core::param::ParamSlot viewport_width_param_;

    /** Parameter setting the size of the rendered nodes */
    core::param::ParamSlot vert_size_param_;

    /** Parameter setting the width of the renderer lines */
    core::param::ParamSlot line_size_param_;

    /** Parameter enabling the drawing of the pdb ids */
    core::param::ParamSlot draw_pdb_ids_param_;

    /** Parameter enabling the drawing of the miniature maps */
    core::param::ParamSlot draw_minimap_param_;

    /** Parameter enabling the drawing of the brenda classes */
    core::param::ParamSlot draw_brenda_param_;

    /** The positions of the nodes */
    std::vector<glm::vec2> node_positions_;

    /** The colors of the nodes */
    std::vector<glm::vec3> node_colors_;

    /** The positions of the line vertices */
    std::vector<glm::vec2> line_positions_;

    /** The colors of the line vertices */
    std::vector<glm::vec3> line_colors_;

    /** The last processed data hash */
    size_t last_data_hash_;

    /** The shader used for the vertices */
    std::unique_ptr<glowl::GLSLProgram> point_shader_;

    /** The shader used for the connecting lines */
    std::unique_ptr<glowl::GLSLProgram> line_shader_;

    std::unique_ptr<glowl::BufferObject> vert_pos_buffer_;
    std::unique_ptr<glowl::BufferObject> vert_col_buffer_;
    std::unique_ptr<glowl::BufferObject> line_pos_buffer_;
    std::unique_ptr<glowl::BufferObject> line_col_buffer_;
    GLuint vert_vao_;
    GLuint line_vao_;
};

} // namespace molsurfmapcluster_gl
} // namespace megamol
