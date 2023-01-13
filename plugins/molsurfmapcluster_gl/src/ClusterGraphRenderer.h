#pragma once

#include "CallClustering_2.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/param/ParamSlot.h"
#include "mmcore/view/CallGetTransferFunction.h"
#include "mmcore_gl/utility/SDFFont.h"
#include "mmcore_gl/view/CallRender2DGL.h"
#include "mmcore_gl/view/Renderer2DModuleGL.h"

#include <glowl/BufferObject.hpp>
#include <glowl/FramebufferObject.hpp>
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

    /** Method to request resources from the frontend */
    void requested_lifetime_resources(frontend_resources::ResourceRequest& req) override {
        ModuleGL::requested_lifetime_resources(req);
        req.require<core::MegaMolGraph>();
    }

private:
    void calculateNodePositions(ClusteringData const& cluster_data);

    void calculateNodeConnections(ClusteringData const& cluster_data);

    void applyClusterColoring(ClusteringData const& cluster_data, std::vector<glm::vec3> const& color_table);

    void uploadDataToGPU();

    void extractColorMap(
        core::view::CallGetTransferFunction const& cgtf, std::vector<glm::vec3>& OUT_color_table) const;

    std::vector<int64_t> getLeaveIndicesInDFSOrder(
        int64_t const start_idx, std::vector<ClusterNode_2> const& nodes) const;

    /** Slot for the cluster data */
    core::CallerSlot cluster_data_slot_;

    /** Slot for the transfer function */
    core::CallerSlot color_input_slot_;

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

    /** Parameter setting the cluster cutoff value */
    core::param::ParamSlot cluster_cutoff_param_;

    /** Parameter for the default node color */
    core::param::ParamSlot default_color_param_;

    /** Parameter for the text color */
    core::param::ParamSlot text_color_param_;

    /** Parameter slot for the selection of the coloring mode */
    core::param::ParamSlot color_mode_selection_param_;

    /** Parameter slot for the file path pointing to the comparison matrix */
    core::param::ParamSlot comparison_matrix_file_param_;

    /** The positions of the nodes */
    std::vector<glm::vec2> node_positions_;

    /** The colors of the nodes */
    std::vector<glm::vec3> node_colors_;

    /** The ids of the nodes */
    std::vector<int> node_ids_;

    /** The positions of the line vertices */
    std::vector<glm::vec2> line_positions_;

    /** The colors of the line vertices */
    std::vector<glm::vec3> line_colors_;

    /** The node ids for all lines */
    std::vector<int> line_node_ids_;

    /** The pdb ids for all nodes */
    std::vector<std::string> node_pdb_ids_;

    /** The last processed data hash */
    size_t last_data_hash_;

    /** The shader used for the vertices */
    std::unique_ptr<glowl::GLSLProgram> point_shader_;

    /** The shader used for the connecting lines */
    std::unique_ptr<glowl::GLSLProgram> line_shader_;

    /** The shader used for the molecular surface maps */
    std::unique_ptr<glowl::GLSLProgram> map_shader_;

    /** The shader used for copying texture contents to the normal framebuffer */
    std::unique_ptr<glowl::GLSLProgram> texture_copy_shader_ptr_;

    /** Buffer containing the node vertex positions */
    std::unique_ptr<glowl::BufferObject> vert_pos_buffer_;

    /** Buffer containing the node vertex color */
    std::unique_ptr<glowl::BufferObject> vert_col_buffer_;

    /** Buffer containing the node vertex id */
    std::unique_ptr<glowl::BufferObject> vert_id_buffer_;

    /** Buffer containing the lien vertex positions */
    std::unique_ptr<glowl::BufferObject> line_pos_buffer_;

    /** Buffer containing the node vertex colors */
    std::unique_ptr<glowl::BufferObject> line_col_buffer_;

    /** Buffer containing the node vertex ids */
    std::unique_ptr<glowl::BufferObject> line_id_buffer_;

    /** FBO for picking */
    std::unique_ptr<glowl::FramebufferObject> fbo_;

    /** Vector containing the pointer to the images */
    std::map<std::filesystem::path, std::shared_ptr<glowl::Texture2D>> picture_storage_;

    /** The font used for font rendering */
    core::utility::SDFFont font_;

    /** rainbow color table to color the clusters */
    std::vector<glm::vec3> cluster_colors_;

    /** ids of the leaf nodes */
    std::vector<int64_t> leaf_ids_;

    /** VAO for the node vertices */
    GLuint vert_vao_;

    /** VAO for the line vertices */
    GLuint line_vao_;

    /** Index of the selected cluster */
    int64_t selected_cluster_id_;

    /** Index of the previous selected cluster */
    int64_t last_selected_cluster_id;

    /** Index of the hovered cluster */
    int64_t hovered_cluster_id_;

    /** Index of the root node */
    int64_t root_id_;

    /** Position of the mouse in screen coordinates */
    glm::dvec2 mouse_pos_;

    /** prefix of the data path */
    std::filesystem::path path_prefix_;
};

} // namespace molsurfmapcluster_gl
} // namespace megamol
