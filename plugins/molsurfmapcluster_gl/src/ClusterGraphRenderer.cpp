#include "ClusterGraphRenderer.h"
#include "CallClustering_2.h"
#include "EnzymeClassProvider.h"

#include "mmcore/CoreInstance.h"
#include "mmcore/param/BoolParam.h"
#include "mmcore/param/ColorParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/IntParam.h"
#include "mmcore/utility/graphics/BitmapCodecCollection.h"
#include "mmcore_gl/utility/RenderUtils.h"
#include "mmcore_gl/utility/ShaderFactory.h"
#include "protein_calls/ProteinColor.h"

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
        , default_color_param_("color::defaultColor", "Default node color")
        , text_color_param_("color::textColor", "Default text color")
        , font_(utility::SDFFont::PRESET_ROBOTO_SANS)
        , last_data_hash_(0)
        , line_vao_(0)
        , vert_vao_(0)
        , point_shader_(nullptr)
        , line_shader_(nullptr)
        , map_shader_(nullptr)
        , selected_cluster_id_(-1)
        , hovered_cluster_id_(-1)
        , last_selected_cluster_id(-1)
        , root_id_(-1) {
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

    default_color_param_.SetParameter(new param::ColorParam(0.3f, 0.3f, 0.3f, 1.0f));
    this->MakeSlotAvailable(&default_color_param_);

    text_color_param_.SetParameter(new param::ColorParam(0.0f, 0.0f, 0.0f, 1.0f));
    this->MakeSlotAvailable(&text_color_param_);
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
    if (button == view::MouseButton::BUTTON_RIGHT && action == view::MouseButtonAction::PRESS) {
        selected_cluster_id_ = hovered_cluster_id_;
    }
    return false;
}

/*
 * ClusterGraphRenderer::OnMouseMove
 */
bool ClusterGraphRenderer::OnMouseMove(double x, double y) {
    if (fbo_ != nullptr) {
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        fbo_->bindToRead(1);
        int read_value = -1;
        glReadPixels(static_cast<GLint>(x), fbo_->getHeight() - static_cast<GLint>(y), 1, 1, GL_RED_INTEGER, GL_INT,
            &read_value);
        glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);
        hovered_cluster_id_ = read_value - 1; // move the value back. 0 corresponds to no id.
    }
    mouse_pos_ = glm::vec2(x, y);
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

        line_shader_ = core::utility::make_glowl_shader("linevertices", shdr_options,
            std::filesystem::path("molsurfmapcluster_gl/passthrough.vert.glsl"),
            std::filesystem::path("molsurfmapcluster_gl/passthrough.frag.glsl"));

        texture_copy_shader_ptr_ = core::utility::make_glowl_shader("texture", shdr_options,
            std::filesystem::path("molsurfmapcluster_gl/texturecopy.vert.glsl"),
            std::filesystem::path("molsurfmapcluster_gl/texturecopy.frag.glsl"));

        map_shader_ = core::utility::make_glowl_shader("map", shdr_options,
            std::filesystem::path("molsurfmapcluster_gl/texture.vert.glsl"),
            std::filesystem::path("molsurfmapcluster_gl/texture.frag.glsl"));

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
    vert_id_buffer_ = std::make_unique<glowl::BufferObject>(GL_ARRAY_BUFFER, nullptr, 0, GL_DYNAMIC_DRAW);
    line_pos_buffer_ = std::make_unique<glowl::BufferObject>(GL_ARRAY_BUFFER, nullptr, 0, GL_DYNAMIC_DRAW);
    line_col_buffer_ = std::make_unique<glowl::BufferObject>(GL_ARRAY_BUFFER, nullptr, 0, GL_DYNAMIC_DRAW);
    line_id_buffer_ = std::make_unique<glowl::BufferObject>(GL_ARRAY_BUFFER, nullptr, 0, GL_DYNAMIC_DRAW);

    glGenVertexArrays(1, &vert_vao_);
    glBindVertexArray(vert_vao_);

    vert_pos_buffer_->bind();
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, nullptr);

    vert_col_buffer_->bind();
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, nullptr);

    vert_id_buffer_->bind();
    glEnableVertexAttribArray(2);
    glVertexAttribIPointer(2, 1, GL_INT, 0, nullptr);

    glBindVertexArray(0);
    glDisableVertexAttribArray(0);
    glDisableVertexAttribArray(1);
    glDisableVertexAttribArray(2);

    glGenVertexArrays(1, &line_vao_);
    glBindVertexArray(line_vao_);

    line_pos_buffer_->bind();
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, nullptr);

    line_col_buffer_->bind();
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, nullptr);

    line_id_buffer_->bind();
    glEnableVertexAttribArray(2);
    glVertexAttribIPointer(2, 1, GL_INT, 0, nullptr);

    glBindVertexArray(0);
    glDisableVertexAttribArray(0);
    glDisableVertexAttribArray(1);
    glDisableVertexAttribArray(2);

    fbo_ = std::make_unique<glowl::FramebufferObject>(1, 1);
    fbo_->createColorAttachment(GL_RGBA16F, GL_RGBA, GL_HALF_FLOAT);
    fbo_->createColorAttachment(GL_R32I, GL_RED, GL_INT);

    if (!font_.Initialise(instance())) {
        megamol::core::utility::log::Log::DefaultLog.WriteMsg(
            megamol::core::utility::log::Log::LEVEL_ERROR, "[ClusterGraphRenderer]: Unable to initialize the font");
    }

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

    // apply necessary changes
    bool changes = false;
    if (last_data_hash_ != meta.dataHash) {
        last_data_hash_ = meta.dataHash;
        changes = true;
        calculateNodePositions(clustering_data);
        calculateNodeConnections(clustering_data);
    }
    changes = selected_cluster_id_ != last_selected_cluster_id ? true : changes;
    if (changes || cluster_cutoff_param_.IsDirty()) {
        last_selected_cluster_id = selected_cluster_id_;
        cluster_cutoff_param_.ResetDirty();
        applyClusterColoring(clustering_data);
        uploadDataToGPU();
    }

    // resize fbo
    auto const view_res = call.GetViewResolution();
    if (fbo_->getWidth() != view_res.x || fbo_->getHeight() != view_res.y) {
        fbo_->resize(view_res.x, view_res.y);
    }

    glDisable(GL_DEPTH_TEST);

    fbo_->bind();
    glClearColor(0, 0, 0, 0);
    glClearDepth(1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // we have to do a specific stunt to clear the id texture to -1
    int constexpr tex_initval = -1;
    auto const texname = fbo_->getColorAttachment(1)->getTextureHandle();
    //glClearTexImage(texname, 0, GL_RED, GL_INT, &tex_initval);

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

    call.GetFramebuffer()->bind();

    // render the graph into the correct fbo
    glActiveTexture(GL_TEXTURE0);
    fbo_->bindColorbuffer(0);
    glActiveTexture(GL_TEXTURE1);
    fbo_->bindColorbuffer(1);

    texture_copy_shader_ptr_->use();
    texture_copy_shader_ptr_->setUniform("input_tx2D", 0);
    texture_copy_shader_ptr_->setUniform("input_txid", 1);

    glDrawArrays(GL_TRIANGLES, 0, 6);

    glUseProgram(0);

    if (root_id_ < 0) {
        return true;
    }

    float top_value = 0.0f;
    auto const& root_node = clustering_data.nodes->at(root_id_);
    float const width_between_nodes =
        static_cast<float>(this->viewport_width_param_.Param<param::IntParam>()->Value()) /
        static_cast<float>(root_node.numLeafNodes - 1);
    float const map_width = 0.8f * width_between_nodes;
    float const map_height = 0.5f * map_width;
    auto text_col = text_color_param_.Param<param::ColorParam>()->Value();

    // draw the minimaps below the graph
    if (draw_minimap_param_.Param<param::BoolParam>()->Value()) {
        map_shader_->use();
        map_shader_->setUniform("mvp", mvp);

        // if necessary, load the textures
        for (const auto& node_id : leaf_ids_) {
            auto& node = clustering_data.nodes->at(node_id);
            if (picture_storage_.count(node.picturePath) == 0) {
                picture_storage_[node.picturePath] = nullptr;
                core_gl::utility::RenderUtils::LoadTextureFromFile(
                    picture_storage_[node.picturePath], node.picturePath, GL_LINEAR, GL_LINEAR);
            }
        }

        for (const auto& node_id : leaf_ids_) {
            auto const node_pos = node_positions_[node_id];
            auto const& node = clustering_data.nodes->at(node_id);
            glm::vec2 lower_left(node_pos.x - 0.5f * map_width, top_value - map_height);
            glm::vec2 upper_right(node_pos.x + 0.5f * map_width, top_value);

            map_shader_->setUniform("lowerleft", lower_left);
            map_shader_->setUniform("upperright", upper_right);

            glActiveTexture(GL_TEXTURE0);
            if (picture_storage_.count(node.picturePath) == 0) {
                continue;
            }
            picture_storage_.at(node.picturePath)->bindTexture();

            map_shader_->setUniform("tex", 0);

            glDrawArrays(GL_TRIANGLES, 0, 6);

            glBindTexture(GL_TEXTURE_2D, 0);
        }
        glUseProgram(0);
        top_value -= map_height;
    }

    // draw the pdb ids
    if (draw_pdb_ids_param_.Param<param::BoolParam>()->Value()) {
        float const text_size = 0.75f * map_height;
        float const text_height = font_.LineHeight(text_size);
        for (const auto& node_id : leaf_ids_) {
            auto const cur_node = clustering_data.nodes->at(node_id);
            auto const node_pos = node_positions_[node_id];
            auto const text = cur_node.pdbID;
            glm::vec2 top_center(node_pos.x, top_value);
            font_.DrawString(mvp, text_col.data(), top_center.x, top_center.y, text_size, false, text.c_str(),
                utility::SDFFont::ALIGN_CENTER_TOP);
        }
        top_value -= text_height;
    }

    // draw the brenda class information
    if (draw_brenda_param_.Param<param::BoolParam>()->Value()) {
        float const text_size = 0.3f * map_height;
        float const text_height = font_.LineHeight(text_size);
        for (const auto& node_id : leaf_ids_) {
            auto const cur_node = clustering_data.nodes->at(node_id);
            auto const node_pos = node_positions_[node_id];
            auto const classes = EnzymeClassProvider::RetrieveClassesForPdbId(cur_node.pdbID, *instance());
            float cur_top = top_value;
            for (auto const brenda_class : classes) {
                auto text = EnzymeClassProvider::ConvertEnzymeClassToString(brenda_class);
                glm::vec2 top_center(node_pos.x, cur_top);
                font_.DrawString(mvp, text_col.data(), top_center.x, top_center.y, text_size, false, text.c_str(),
                    utility::SDFFont::ALIGN_CENTER_TOP);
                cur_top -= text_height;
            }
        }
    }

    // draw the mouse hover map and string
    if (hovered_cluster_id_ >= 0) {
        auto cam_pose = cam.getPose();
        auto cam_intrinsics = cam.get<core::view::Camera::OrthographicParameters>();
        glm::dvec2 world_pos(-1.0);
        world_pos.x = ((mouse_pos_.x * 2.0 / static_cast<double>(call.GetViewResolution().x)) - 1.0);
        world_pos.y = 1.0 - (mouse_pos_.y * 2.0 / static_cast<double>(call.GetViewResolution().y));
        world_pos.x = world_pos.x * 0.5 * cam_intrinsics.frustrum_height * cam_intrinsics.aspect + cam_pose.position.x;
        world_pos.y = world_pos.y * 0.5 * cam_intrinsics.frustrum_height + cam_pose.position.y;

        auto const& hovered_node = clustering_data.nodes->at(hovered_cluster_id_);

        auto const hover_map_width = 0.4f * static_cast<float>(call.GetViewResolution().x);
        auto const hover_map_height = 0.5f * hover_map_width;

        glm::vec2 const lower_left = world_pos + glm::dvec2(0.01 * hover_map_width);
        glm::vec2 const upper_right = lower_left + glm::vec2(hover_map_width, hover_map_height);
        glm::vec2 const middle = 0.5f * (lower_left + upper_right);

        map_shader_->use();
        map_shader_->setUniform("mvp", mvp);
        map_shader_->setUniform("lowerleft", lower_left);
        map_shader_->setUniform("upperright", upper_right);

        std::string path = hovered_node.picturePath;
        if (hovered_node.representative >= 0) {
            auto const& node = clustering_data.nodes->at(hovered_node.representative);
            path = node.picturePath;
        }

        glActiveTexture(GL_TEXTURE0);
        picture_storage_.at(path)->bindTexture();

        glDrawArrays(GL_TRIANGLES, 0, 6);

        glBindTexture(GL_TEXTURE_2D, 0);
        glUseProgram(0);

        std::string const text = hovered_node.pdbID;
        auto const text_size = 0.3f * hover_map_height;
        font_.DrawString(mvp, text_col.data(), middle.x, middle.y, text_size, false, text.c_str(),
            utility::SDFFont::ALIGN_CENTER_MIDDLE);
    }

    glEnable(GL_DEPTH_TEST);
    return true;
}

void ClusterGraphRenderer::calculateNodePositions(ClusteringData const& cluster_data) {
    node_positions_.clear();
    node_colors_.clear();
    node_ids_.clear();
    leaf_ids_.clear();
    root_id_ = -1;
    selected_cluster_id_ = -1;
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
    root_id_ = root_idx;

    glm::vec2 const min_coord(0.05f * viewport_width_param_.Param<param::IntParam>()->Value(),
        0.05f * viewport_height_param_.Param<param::IntParam>()->Value());
    glm::vec2 const max_coord(0.95f * viewport_width_param_.Param<param::IntParam>()->Value(),
        0.95f * viewport_height_param_.Param<param::IntParam>()->Value());

    float constexpr min_height_value = 0.0f;
    float const max_height_value = std::max_element(nodes.begin(), nodes.end(), [](auto n_1, auto n_2) {
        return n_1.height < n_2.height;
    })->height;

    // process all nodes
    leaf_ids_ = getLeaveIndicesInDFSOrder(root_idx, nodes);
    auto to_process_queue = leaf_ids_;
    glm::vec3 const default_color = glm::make_vec3(default_color_param_.Param<param::ColorParam>()->Value().data());
    node_positions_.resize(nodes.size(), glm::vec2(0.0f, 0.0f));
    node_colors_.resize(nodes.size(), default_color);
    node_ids_.resize(nodes.size(), -1);
    const auto& root_node = nodes[root_idx];
    float const width_between_nodes = (max_coord.x - min_coord.x) / static_cast<float>(root_node.numLeafNodes - 1);
    // process leaf nodes
    for (int64_t i = 0; i < to_process_queue.size(); ++i) {
        auto const node_idx = to_process_queue[i];
        node_positions_[node_idx] = glm::vec2(min_coord.x + i * width_between_nodes, min_coord.y);
        node_ids_[node_idx] = static_cast<int>(node_idx);
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
        node_ids_[cur_idx] = static_cast<int>(cur_idx);
    }
}

void ClusterGraphRenderer::calculateNodeConnections(ClusteringData const& cluster_data) {
    line_positions_.clear();
    line_colors_.clear();
    line_node_ids_.clear();
    if (cluster_data.nodes == nullptr) {
        return;
    }

    // the line node order always goes from the root node to the children
    // this ensures the correct cluster id picking later on
    auto const& nodes = *cluster_data.nodes;
    for (auto const& node : nodes) {
        int const parent_id = static_cast<int>(node.id);
        if (node.left >= 0) {
            auto const start_pos = node_positions_[node.id];
            auto const end_pos = node_positions_[node.left];
            line_positions_.emplace_back(start_pos);
            line_positions_.emplace_back(glm::vec2(end_pos.x, start_pos.y));
            line_positions_.emplace_back(glm::vec2(end_pos.x, start_pos.y));
            line_positions_.emplace_back(end_pos);
            int const id = static_cast<int>(node.left);
            std::vector ids = {parent_id, parent_id, parent_id, id};
            line_node_ids_.insert(line_node_ids_.end(), ids.begin(), ids.end());
        }
        if (node.right >= 0) {
            auto const start_pos = node_positions_[node.id];
            auto const end_pos = node_positions_[node.right];
            line_positions_.emplace_back(start_pos);
            line_positions_.emplace_back(glm::vec2(end_pos.x, start_pos.y));
            line_positions_.emplace_back(glm::vec2(end_pos.x, start_pos.y));
            line_positions_.emplace_back(end_pos);
            int const id = static_cast<int>(node.right);
            std::vector ids = {parent_id, parent_id, parent_id, id};
            line_node_ids_.insert(line_node_ids_.end(), ids.begin(), ids.end());
        }
    }
    glm::vec3 const default_color = glm::make_vec3(default_color_param_.Param<param::ColorParam>()->Value().data());
    line_colors_.resize(line_positions_.size(), default_color);
}

void ClusterGraphRenderer::uploadDataToGPU() {
    vert_pos_buffer_->rebuffer(node_positions_);
    vert_col_buffer_->rebuffer(node_colors_);
    vert_id_buffer_->rebuffer(node_ids_);
    line_pos_buffer_->rebuffer(line_positions_);
    line_col_buffer_->rebuffer(line_colors_);
    line_id_buffer_->rebuffer(line_node_ids_);
}

void ClusterGraphRenderer::applyClusterColoring(ClusteringData const& cluster_data) {
    int64_t const cur_selected_id = selected_cluster_id_ >= 0 ? selected_cluster_id_ : root_id_;
    if (cur_selected_id < 0) {
        return;
    }
    float const cluster_factor = cluster_cutoff_param_.Param<param::FloatParam>()->Value();
    auto const& root_node = cluster_data.nodes->at(cur_selected_id);
    float const init_height = root_node.height;
    float const border_height = init_height * cluster_factor;
    auto const& node_vec = *cluster_data.nodes;

    // setup filtering set containing all ids of the searched subtree
    std::set<int64_t> subtree_set;
    std::deque<int64_t> nodes_to_process = {cur_selected_id};
    while (!nodes_to_process.empty()) {
        auto const cur_id = nodes_to_process.front();
        nodes_to_process.pop_front();
        auto const& cur_node = node_vec[cur_id];
        if (cur_node.left >= 0) {
            nodes_to_process.push_back(cur_node.left);
        }
        if (cur_node.right >= 0) {
            nodes_to_process.push_back(cur_node.right);
        }
        subtree_set.insert(cur_id);
    }

    // search for the cluster roots
    std::vector<int64_t> cluster_roots;
    for (const auto& node : node_vec) {
        if (node.parent >= 0 && subtree_set.count(node.id) > 0) {
            auto const& parent_node = node_vec[node.parent];
            if (node.height <= border_height && parent_node.height > border_height) {
                cluster_roots.emplace_back(node.id);
            }
        }
    }

    // find out the cluster id for each node
    std::vector<int32_t> cluster_assignment(node_vec.size(), -1);
    int32_t cur_cluster_id = 0;
    for (const auto& cur_root_id : cluster_roots) {
        std::deque<int64_t> to_process = {cur_root_id};
        while (!to_process.empty()) {
            auto const cur_id = to_process.front();
            to_process.pop_front();
            auto const& node = node_vec[cur_id];
            cluster_assignment[cur_id] = cur_cluster_id;
            if (node.left >= 0) {
                to_process.push_back(node.left);
            }
            if (node.right >= 0) {
                to_process.push_back(node.right);
            }
        }
        ++cur_cluster_id;
    }

    cluster_colors_.clear();
    protein_calls::ProteinColor::MakeRainbowColorTable(cluster_roots.size(), cluster_colors_);

    // apply the colors to the nodes
    glm::vec3 const default_color = glm::make_vec3(default_color_param_.Param<param::ColorParam>()->Value().data());
    for (int64_t node_id = 0; node_id < node_colors_.size(); ++node_id) {
        auto const cluster_id = cluster_assignment[node_id];
        node_colors_[node_id] = default_color;
        if (cluster_id >= 0) {
            node_colors_[node_id] = cluster_colors_[cluster_id];
        }
    }

    // apply the colors to the lines
    for (int64_t line_id = 0; line_id < line_colors_.size(); ++line_id) {
        auto const node_id = line_node_ids_[line_id];
        auto const cluster_id = cluster_assignment[node_id];
        line_colors_[line_id] = default_color;
        if (cluster_id >= 0) {
            line_colors_[line_id] = cluster_colors_[cluster_id];
        }
    }
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
