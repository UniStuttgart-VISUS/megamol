#include "3DUIRenderer.h"

#include "tiny_gltf.h"
#include "vislib/math/Matrix.h"
#include "vislib_gl/graphics/gl/IncludeAllGL.h"

#include "mesh_gl/MeshCalls_gl.h"


megamol::mesh_gl::ThreeDimensionalUIRenderer::ThreeDimensionalUIRenderer()
        : m_version(0)
        , m_interaction_collection(new mesh::ThreeDimensionalInteractionCollection)
        , m_3DInteraction_calleeSlot(
              "getInteraction", "The slot publishing available interactions and receiving pending manipulations")
        , m_3DInteraction_callerSlot("", "")
        , m_glTF_callerSlot("getGlTFFile", "Connects the data source with a loaded glTF file") {
    this->m_3DInteraction_calleeSlot.SetCallback(mesh::Call3DInteraction::ClassName(), "GetData", &ThreeDimensionalUIRenderer::getInteractionCallback);
    this->m_3DInteraction_calleeSlot.SetCallback(mesh::Call3DInteraction::ClassName(), "GetMetaData", &ThreeDimensionalUIRenderer::getInteractionCallback);
    this->MakeSlotAvailable(&this->m_3DInteraction_calleeSlot);

    this->m_glTF_callerSlot.SetCompatibleCall<mesh::CallGlTFDataDescription>();
    this->MakeSlotAvailable(&this->m_glTF_callerSlot);

    this->m_3DInteraction_callerSlot.SetCompatibleCall<mesh::Call3DInteractionDescription>();
    this->MakeSlotAvailable(&this->m_3DInteraction_callerSlot);
}

megamol::mesh_gl::ThreeDimensionalUIRenderer::~ThreeDimensionalUIRenderer() {}

void megamol::mesh_gl::ThreeDimensionalUIRenderer::createMaterialCollection() {
    material_collection_ = std::make_shared<GPUMaterialCollection>();
    material_collection_->addMaterial(this->instance(), "3DUIMaterial",
        {"mesh_gl/3DUI.vert.glsl", "mesh_gl/3DUI.geom.glsl", "mesh_gl/3DUI.frag.glsl"});
}

void megamol::mesh_gl::ThreeDimensionalUIRenderer::updateRenderTaskCollection(
    mmstd_gl::CallRender3DGL& call, bool force_update) {
    bool something_has_changed = force_update;

    mesh::CallGlTFData* gltf_call = this->m_glTF_callerSlot.CallAs<mesh::CallGlTFData>();
    if (gltf_call != nullptr) {
        if (!(*gltf_call)(0)) {
            return;
        }

        something_has_changed |= gltf_call->hasUpdate();

        if (something_has_changed) {
            ++m_version;

            render_task_collection_->clear();

            auto model = gltf_call->getData().second;

            for (size_t node_idx = 0; node_idx < model->nodes.size(); node_idx++) {
                if (node_idx < model->nodes.size() && model->nodes[node_idx].mesh != -1) {

                    std::array<PerObjectShaderParams, 1> per_obj_data;

                    if (model->nodes[node_idx].matrix.size() != 0) // has matrix transform
                    {
                        // TODO
                    } else {
                        auto& translation = model->nodes[node_idx].translation;
                        auto& scale = model->nodes[node_idx].scale;
                        auto& rotation = model->nodes[node_idx].rotation;

                        if (translation.size() != 0) {
                            per_obj_data[0].object_transform.SetAt(0, 3, translation[0]);
                            per_obj_data[0].object_transform.SetAt(1, 3, translation[1]);
                            per_obj_data[0].object_transform.SetAt(2, 3, translation[2]);
                        }

                        if (scale.size() != 0) {}

                        if (rotation.size() != 0) {}
                    }

                    // TODO check node name for UI element names
                    if (model->nodes[node_idx].name == "axisX_arrow") {
                        per_obj_data[0].color = {1.0f, 0.0f, 0.0f, 1.0f};
                        per_obj_data[0].id = 0;

                        m_UI_template_elements[0].first = mesh_collection_->getSubMesh("axisX_arrow");
                        m_UI_template_elements[0].second = per_obj_data;

                    } else if (model->nodes[node_idx].name == "axisY_arrow") {
                        per_obj_data[0].color = {0.0f, 1.0f, 0.0f, 1.0f};
                        per_obj_data[0].id = 0;

                        m_UI_template_elements[1].first = mesh_collection_->getSubMesh("axisY_arrow");
                        m_UI_template_elements[1].second = per_obj_data;

                    } else if (model->nodes[node_idx].name == "axisZ_arrow") {
                        per_obj_data[0].color = {0.0f, 0.0f, 1.0f, 1.0f};
                        per_obj_data[0].id = 0;

                        m_UI_template_elements[2].first = mesh_collection_->getSubMesh("axisZ_arrow");
                        m_UI_template_elements[2].second = per_obj_data;

                    } else if (model->nodes[node_idx].name == "slider_arrow") {
                        per_obj_data[0].color = {1.0f, 0.0f, 1.0f, 1.0f};
                        per_obj_data[0].id = 0;

                        m_UI_template_elements[3].first = mesh_collection_->getSubMesh("slider_arrow");
                        m_UI_template_elements[3].second = per_obj_data;
                    }
                }
            }

            auto const& shader = material_collection_->getMaterials().find("3DUI")->second.shader_program;

            int render_task_index = 0;
            {
                // TODO create debug scene from UI template obejcts
                m_scene.push_back({0, {PerObjectShaderParams()}});
                m_scene.back().second = m_UI_template_elements[3].second;
                m_scene.back().second[0].id = 1;

                auto const& sub_mesh = m_UI_template_elements[3].first;
                auto const& gpu_batch_mesh = sub_mesh.mesh->mesh;

                std::string rt_identifier(std::string(this->FullName()) + "_" + std::to_string(++render_task_index));
                render_task_collection_->addRenderTask(
                    rt_identifier, shader, gpu_batch_mesh, sub_mesh.sub_mesh_draw_command, m_scene.back().second);

                m_scene.back().first = rt_identifier;

                m_interaction_collection->addInteractionObject(
                    1, {mesh::ThreeDimensionalInteraction{
                           mesh::InteractionType::MOVE_ALONG_AXIS, 1, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f}});
            }
            {
                m_scene.push_back({0, {PerObjectShaderParams()}});
                m_scene.back().second = m_UI_template_elements[2].second;
                m_scene.back().second[0].id = 2;

                auto const& sub_mesh = m_UI_template_elements[2].first;
                auto const& gpu_batch_mesh = sub_mesh.mesh->mesh;

                std::string rt_identifier(std::string(this->FullName()) + "_" + std::to_string(++render_task_index));
                render_task_collection_->addRenderTask(
                    rt_identifier, shader, gpu_batch_mesh, sub_mesh.sub_mesh_draw_command, m_scene.back().second);

                m_scene.back().first = rt_identifier;

                m_interaction_collection->addInteractionObject(
                    2, {mesh::ThreeDimensionalInteraction{
                           mesh::InteractionType::MOVE_ALONG_AXIS, 2, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f}});
            }
            {
                m_scene.push_back({0, {PerObjectShaderParams()}});
                m_scene.back().second = m_UI_template_elements[1].second;
                m_scene.back().second[0].id = 3;

                auto const& sub_mesh = m_UI_template_elements[1].first;
                auto const& gpu_batch_mesh = sub_mesh.mesh->mesh;

                std::string rt_identifier(std::string(this->FullName()) + "_" + std::to_string(++render_task_index));
                render_task_collection_->addRenderTask(
                    rt_identifier, shader, gpu_batch_mesh, sub_mesh.sub_mesh_draw_command, m_scene.back().second);

                m_scene.back().first = rt_identifier;

                m_interaction_collection->addInteractionObject(
                    3, {mesh::ThreeDimensionalInteraction{
                           mesh::InteractionType::MOVE_ALONG_AXIS, 3, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f}});
            }
            {
                m_scene.push_back({0, {PerObjectShaderParams()}});
                m_scene.back().second = m_UI_template_elements[0].second;
                m_scene.back().second[0].id = 4;

                auto const& sub_mesh = m_UI_template_elements[0].first;
                auto const& gpu_batch_mesh = sub_mesh.mesh->mesh;

                std::string rt_identifier(std::string(this->FullName()) + "_" + std::to_string(++render_task_index));
                render_task_collection_->addRenderTask(
                    rt_identifier, shader, gpu_batch_mesh, sub_mesh.sub_mesh_draw_command, m_scene.back().second);

                m_scene.back().first = rt_identifier;

                m_interaction_collection->addInteractionObject(
                    4, {mesh::ThreeDimensionalInteraction{
                           mesh::InteractionType::MOVE_ALONG_AXIS, 4, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}});
            }


            // add some lights to the scene to test the per frame buffers
            struct LightParams {
                float x, y, z, intensity;
            };

            // Place lights in icosahedron pattern
            float x = 0.525731112119133606f * 500.0f;
            float z = 0.850650808352039932f * 500.0f;

            std::vector<LightParams> lights = {{-x, 0.0f, z, 1.0f}, {x, 0.0f, z, 1.0f}, {-x, 0.0f, -z, 1.0f},
                {x, 0.0f, -z, 1.0f}, {0.0f, z, x, 1.0f}, {0.0f, z, -x, 1.0f}, {0.0f, -z, x, 1.0f}, {0.0f, -z, -x, 1.0f},
                {z, x, 0.0f, 1.0f}, {-z, x, 0.0f, 1.0f}, {z, -x, 0.0f, 1.0f}, {-z, -x, 0.0f, 1.0f}};

            // Add a key light
            lights.push_back({-5000.0, 5000.0, -5000.0, 1000.0f});

            render_task_collection_->deletePerFrameDataBuffer(1);
            render_task_collection_->addPerFrameDataBuffer("lights", lights, 1);
        }
    }
}

bool megamol::mesh_gl::ThreeDimensionalUIRenderer::getInteractionCallback(core::Call& caller) {

    mesh::Call3DInteraction* ci = dynamic_cast<mesh::Call3DInteraction*>(&caller);
    if (ci == NULL)
        return false;

    if (ci->getData() == nullptr) {
        ci->setData(this->m_interaction_collection, m_version);
    }

    // clear non persistent changes, such has highlighting
    for (auto& entity : m_scene) {
        entity.second[0].highlighted = 0;
    }

    // TODO consume pending manipulations
    while (!ci->getData()->accessPendingManipulations().empty()) {
        mesh::ThreeDimensionalManipulation manipulation = ci->getData()->accessPendingManipulations().front();

        std::list<std::pair<std::string, std::array<PerObjectShaderParams, 1>>>::iterator it = m_scene.begin();
        std::advance(it, manipulation.obj_id - 1);

        std::array<PerObjectShaderParams, 1> per_obj_data = it->second;

        vislib::math::Vector<float, 4> translate_col;

        switch (manipulation.type) {
        case mesh::MOVE_ALONG_AXIS:
            translate_col = per_obj_data[0].object_transform.GetColumn(3);
            translate_col += vislib::math::Vector<float, 4>(manipulation.axis_x * manipulation.value,
                manipulation.axis_y * manipulation.value, manipulation.axis_Z * manipulation.value, 0.0f);
            per_obj_data[0].object_transform.SetAt(0, 3, translate_col.X());
            per_obj_data[0].object_transform.SetAt(1, 3, translate_col.Y());
            per_obj_data[0].object_transform.SetAt(2, 3, translate_col.Z());
            break;
        case mesh::MOVE_IN_PLANE:
            break;
        case mesh::ROTATE_AROUND_AXIS:
            break;
        case mesh::SELECT:
            break;
        case mesh::DESELET:
            break;
        case mesh::HIGHLIGHT:
            std::cout << "Hightlight: " << manipulation.obj_id << std::endl;
            per_obj_data[0].highlighted = 1;
            // per_obj_data[0].color = {1.0f, 1.0f, 0.0f, 1.0f};a
            break;
        default:
            break;
        }

        it->second = per_obj_data; // overwrite object data for persistent change

        ci->getData()->accessPendingManipulations().pop();
    }

    // update all per obj data buffers
    for (auto& entity : m_scene) {
        render_task_collection_->updatePerDrawData(entity.first, entity.second);
    }

    return true;
}
