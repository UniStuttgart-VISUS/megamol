#include "ClusterGraphRenderer.h"
#include "CallClustering_2.h"

#include "mmcore/CoreInstance.h"
#include "mmcore/param/BoolParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/IntParam.h"
#include "mmcore_gl/utility/ShaderFactory.h"

#include <algorithm>
#include <deque>
#include <set>

using namespace megamol;
using namespace megamol::core;
using namespace megamol::molsurfmapcluster_gl;

/*
 * ClusterGraphRenderer::ClusterGraphRenderer
 */
ClusterGraphRenderer::ClusterGraphRenderer(void)
        : core_gl::view::Renderer2DModuleGL()
        , cluster_data_slot_("clusterData", "Input slot for the cluster data")
        , viewport_height_param_("viewport::height", "Height of the viewport")
        , viewport_width_param_("viewport::width", "Width of the viewport")
        , vert_size_param_("graph::vertexSize", "Size of the rendered vertices")
        , line_size_param_("graph::lineWidth", "Width of the rendered line")
        , draw_pdb_ids_param_("information::PDBId", "Enable the drawing of the PDB IDs")
        , draw_minimap_param_("information::minimap", "Enable the drawing of the minimaps")
        , draw_brenda_param_("information::brenda", "Enable the drawing of the BRENDA classes")
        , cluster_cutoff_param_("clusterCutoff", "Cutoff value to determine the size of clusters")
        , last_data_hash_(0)
        , line_vao_(0)
        , vert_vao_(0)
        , selected_cluster_id_(-1) {
    // Caller Slot
    cluster_data_slot_.SetCompatibleCall<CallClustering_2Description>();
    this->MakeSlotAvailable(&cluster_data_slot_);

    // Parameter Slots
    viewport_height_param_.SetParameter(new param::IntParam(1440, 1000, 10800));
    this->MakeSlotAvailable(&viewport_height_param_);

    viewport_width_param_.SetParameter(new param::IntParam(2560, 1000, 10800));
    this->MakeSlotAvailable(&viewport_width_param_);

    vert_size_param_.SetParameter(new param::IntParam(15, 1, 200));
    this->MakeSlotAvailable(&vert_size_param_);

    line_size_param_.SetParameter(new param::IntParam(3, 1, 10));
    this->MakeSlotAvailable(&line_size_param_);

    draw_pdb_ids_param_.SetParameter(new param::BoolParam(true));
    this->MakeSlotAvailable(&draw_pdb_ids_param_);

    draw_minimap_param_.SetParameter(new param::BoolParam(true));
    this->MakeSlotAvailable(&draw_minimap_param_);

    draw_brenda_param_.SetParameter(new param::BoolParam(true));
    this->MakeSlotAvailable(&draw_brenda_param_);

    cluster_cutoff_param_.SetParameter(new param::FloatParam(0.7f, 0.01f, 1.0f));
    this->MakeSlotAvailable(&cluster_cutoff_param_);
}

/*
 * ClusterGraphRenderer::~ClusterGraphRenderer
 */
ClusterGraphRenderer::~ClusterGraphRenderer(void) {
    this->Release();
}

/*
 * ClusterGraphRenderer::OnMouseButton
 */
bool ClusterGraphRenderer::OnMouseButton(
    view::MouseButton button, view::MouseButtonAction action, view::Modifiers mods) {
    // TODO
    return false;
}

/*
 * ClusterGraphRenderer::OnMouseMove
 */
bool ClusterGraphRenderer::OnMouseMove(double x, double y) {
    // TODO
    return false;
}

/*
 * ClusterGraphRenderer::create
 */
bool ClusterGraphRenderer::create(void) {

    try {
        auto const shdr_options = msf::ShaderFactoryOptionsOpenGL(instance()->GetShaderPaths());

        point_shader_ = core::utility::make_glowl_shader("vertices", shdr_options,
            std::filesystem::path("molsurfmapcluster_gl/passthrough.vert.glsl"),
            std::filesystem::path("molsurfmapcluster_gl/passthrough.frag.glsl"));

        line_shader_ = core::utility::make_glowl_shader("vertices", shdr_options,
            std::filesystem::path("molsurfmapcluster_gl/passthrough.vert.glsl"),
            std::filesystem::path("molsurfmapcluster_gl/passthrough.frag.glsl"));

    } catch (glowl::GLSLProgramException const& ex) {
        megamol::core::utility::log::Log::DefaultLog.WriteMsg(
            megamol::core::utility::log::Log::LEVEL_ERROR, "[ClusterGraphRenderer] %s", ex.what());
    } catch (std::exception const& ex) {
        megamol::core::utility::log::Log::DefaultLog.WriteMsg(megamol::core::utility::log::Log::LEVEL_ERROR,
            "[ClusterGraphRenderer] Unable to compile shader: Unknown exception: %s", ex.what());
    } catch (...) {
        megamol::core::utility::log::Log::DefaultLog.WriteMsg(megamol::core::utility::log::Log::LEVEL_ERROR,
            "[ClusterGraphRenderer] Unable to compile shader: Unknown exception.");
    }

    vert_pos_buffer_ = std::make_unique<glowl::BufferObject>(GL_ARRAY_BUFFER, nullptr, 0, GL_DYNAMIC_DRAW);
    vert_col_buffer_ = std::make_unique<glowl::BufferObject>(GL_ARRAY_BUFFER, nullptr, 0, GL_DYNAMIC_DRAW);
    line_pos_buffer_ = std::make_unique<glowl::BufferObject>(GL_ARRAY_BUFFER, nullptr, 0, GL_DYNAMIC_DRAW);
    line_col_buffer_ = std::make_unique<glowl::BufferObject>(GL_ARRAY_BUFFER, nullptr, 0, GL_DYNAMIC_DRAW);

    glGenVertexArrays(1, &vert_vao_);
    glBindVertexArray(vert_vao_);

    vert_pos_buffer_->bind();
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, nullptr);

    vert_col_buffer_->bind();
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, nullptr);

    glBindVertexArray(0);
    glDisableVertexAttribArray(0);
    glDisableVertexAttribArray(1);

    glGenVertexArrays(1, &line_vao_);
    glBindVertexArray(line_vao_);

    line_pos_buffer_->bind();
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, nullptr);

    line_col_buffer_->bind();
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, nullptr);

    glBindVertexArray(0);
    glDisableVertexAttribArray(0);
    glDisableVertexAttribArray(1);

    return true;
}

/*
 * ClusterGraphRenderer::release
 */
void ClusterGraphRenderer::release(void) {
    // TODO
}

/*
 * ClusterGraphRenderer::GetExtents
 */
bool ClusterGraphRenderer::GetExtents(core_gl::view::CallRender2DGL& call) {
    call.AccessBoundingBoxes().SetBoundingBox(0, 0,
        static_cast<float>(viewport_width_param_.Param<param::IntParam>()->Value()),
        static_cast<float>(viewport_height_param_.Param<param::IntParam>()->Value()));

    auto const cc = cluster_data_slot_.CallAs<CallClustering_2>();
    if (cc == nullptr)
        return false;

    if (!(*cc)(CallClustering_2::CallForGetExtent))
        return false;

    return true;
}

/*
 * ClusterGraphRenderer::Render
 */
bool ClusterGraphRenderer::Render(core_gl::view::CallRender2DGL& call) {
    auto const cc = cluster_data_slot_.CallAs<CallClustering_2>();
    if (cc == nullptr)
        return false;

    if (!(*cc)(CallClustering_2::CallForGetData))
        return false;

    auto const& cam = call.GetCamera();
    auto const view = cam.getViewMatrix();
    auto const proj = cam.getProjectionMatrix();
    auto const mvp = proj * view;

    auto const& meta = cc->GetMetaData();
    auto const& clustering_data = cc->GetData();

    if (last_data_hash_ != meta.dataHash) {
        last_data_hash_ = meta.dataHash;
        calculateNodePositions(clustering_data);
        calculateNodeConnections(clustering_data);
        uploadDataToGPU();
    }

    // render lines
    glBindVertexArray(line_vao_);
    line_shader_->use();

    line_shader_->setUniform("mvp", mvp);
    glLineWidth(static_cast<GLfloat>(line_size_param_.Param<param::IntParam>()->Value()));

    glDrawArrays(GL_LINES, 0, line_positions_.size());

    glLineWidth(1.0f);
    glUseProgram(0);
    glBindVertexArray(0);

    // render nodes
    glBindVertexArray(vert_vao_);
    point_shader_->use();

    point_shader_->setUniform("mvp", mvp);
    glPointSize(static_cast<GLfloat>(vert_size_param_.Param<param::IntParam>()->Value()));

    glDrawArrays(GL_POINTS, 0, node_positions_.size());

    glPointSize(1.0);
    glUseProgram(0);
    glBindVertexArray(0);

    return true;
}

void ClusterGraphRenderer::calculateNodePositions(ClusteringData const& cluster_data) {
    node_positions_.clear();
    node_colors_.clear();
    if (cluster_data.nodes == nullptr) {
        return;
    }

    // step 1: identify root
    auto const& nodes = *cluster_data.nodes;
    auto const& root_location =
        std::find_if(nodes.begin(), nodes.end(), [](ClusterNode_2 const& node) { return node.parent < 0; });
    int64_t root_idx = -1;
    if (root_location != nodes.end()) {
        root_idx = (*root_location).id;
    } else {
        utility::log::Log::DefaultLog.WriteError("[ClusterGraphRenderer]: No tree root could be identified!");
        return;
    }

    glm::vec2 const min_coord(0.05f * viewport_width_param_.Param<param::IntParam>()->Value(),
        0.05f * viewport_height_param_.Param<param::IntParam>()->Value());
    glm::vec2 const max_coord(0.95f * viewport_width_param_.Param<param::IntParam>()->Value(),
        0.95f * viewport_height_param_.Param<param::IntParam>()->Value());

    float constexpr min_height_value = 0.0f;
    float const max_height_value = std::max_element(nodes.begin(), nodes.end(), [](auto n_1, auto n_2) {
        return n_1.height < n_2.height;
    })->height;

    // process all nodes
    std::vector to_process_queue = getLeaveIndicesInDFSOrder(root_idx, nodes);
    node_positions_.resize(nodes.size(), glm::vec2(0.0f, 0.0f));
    node_colors_.resize(nodes.size(), glm::vec3(0.0f, 0.0f, 0.0f));
    const auto& root_node = nodes[root_idx];
    float const width_between_nodes = (max_coord.x - min_coord.x) / static_cast<float>(root_node.numLeafNodes - 1);
    // process leaf nodes
    for (int64_t i = 0; i < to_process_queue.size(); ++i) {
        auto const node_idx = to_process_queue[i];
        node_positions_[node_idx] = glm::vec2(min_coord.x + i * width_between_nodes, min_coord.y);
    }
    // the following nodes are already sorted correctly, so we can process them directly
    for (int64_t cur_idx = to_process_queue.size(); cur_idx < nodes.size(); ++cur_idx) {
        auto const& cur_node = nodes[cur_idx];
        auto const& left_node = nodes[cur_node.left];
        auto const& right_node = nodes[cur_node.right];
        float const a = (cur_node.height - min_height_value) / (max_height_value - min_height_value);
        float const x_coord = (node_positions_[right_node.id].x + node_positions_[left_node.id].x) * 0.5f;
        float const y_coord = glm::mix(min_coord.y, max_coord.y, a);
        node_positions_[cur_idx] = glm::vec2(x_coord, y_coord);
    }
}

void ClusterGraphRenderer::calculateNodeConnections(ClusteringData const& cluster_data) {
    line_positions_.clear();
    line_colors_.clear();
    if (cluster_data.nodes == nullptr) {
        return;
    }

    auto const& nodes = *cluster_data.nodes;
    for (auto const& node : nodes) {
        if (node.left >= 0) {
            auto const start_pos = node_positions_[node.left];
            auto const end_pos = node_positions_[node.id];
            line_positions_.emplace_back(start_pos);
            line_positions_.emplace_back(glm::vec2(start_pos.x, end_pos.y));
            line_positions_.emplace_back(glm::vec2(start_pos.x, end_pos.y));
            line_positions_.emplace_back(end_pos);
        }
        if (node.right >= 0) {
            auto const start_pos = node_positions_[node.right];
            auto const end_pos = node_positions_[node.id];
            line_positions_.emplace_back(start_pos);
            line_positions_.emplace_back(glm::vec2(start_pos.x, end_pos.y));
            line_positions_.emplace_back(glm::vec2(start_pos.x, end_pos.y));
            line_positions_.emplace_back(end_pos);
        }
    }

    line_colors_.resize(line_positions_.size(), glm::vec3(0.0f, 0.0f, 0.0f));
}

void ClusterGraphRenderer::uploadDataToGPU() {
    vert_pos_buffer_->rebuffer(node_positions_);
    vert_col_buffer_->rebuffer(node_colors_);
    line_pos_buffer_->rebuffer(line_positions_);
    line_col_buffer_->rebuffer(line_colors_);
}

void ClusterGraphRenderer::applyClusterColoring(ClusteringData const& cluster_data) {
    // TODO
}


std::vector<int64_t> ClusterGraphRenderer::getLeaveIndicesInDFSOrder(
    int64_t const start_idx, std::vector<ClusterNode_2> const& nodes) const {
    std::vector<int64_t> result;
    if (start_idx < 0 || start_idx >= nodes.size()) {
        utility::log::Log::DefaultLog.WriteError(
            "[ClusterGraphRenderer]: Wrong input index for the depth first search");
        return result;
    }
    auto const& cur_node = nodes[start_idx];
    if (cur_node.left < 0 && cur_node.right < 0) {
        // leaf node
        result = {start_idx};
    } else if (cur_node.left < 0 && cur_node.right >= 0) {
        // this should not happen (leaf to the left)
        auto const left_res = {start_idx};
        auto const right_res = getLeaveIndicesInDFSOrder(cur_node.right, nodes);
        result.insert(result.end(), left_res.begin(), left_res.end());
        result.insert(result.end(), right_res.begin(), right_res.end());
    } else if (cur_node.left >= 0 && cur_node.right < 0) {
        // this should not happen (leaf to the right)
        auto const left_res = getLeaveIndicesInDFSOrder(cur_node.left, nodes);
        auto const right_res = {start_idx};
        result.insert(result.end(), left_res.begin(), left_res.end());
        result.insert(result.end(), right_res.begin(), right_res.end());
    } else {
        auto const left_res = getLeaveIndicesInDFSOrder(cur_node.left, nodes);
        auto const right_res = getLeaveIndicesInDFSOrder(cur_node.right, nodes);
        result.insert(result.end(), left_res.begin(), left_res.end());
        result.insert(result.end(), right_res.begin(), right_res.end());
    }
    return result;
}
