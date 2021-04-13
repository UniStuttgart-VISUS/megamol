#include "stdafx.h"

#include "3DUIRenderTaskDataSource.h"

#include "tiny_gltf.h"
#include "vislib/graphics/gl/IncludeAllGL.h"
#include "vislib/math/Matrix.h"

#include "mesh/MeshCalls.h"


megamol::mesh::ThreeDimensionalUIRenderTaskDataSource::ThreeDimensionalUIRenderTaskDataSource()
        : m_version(0)
        , m_interaction_collection(new ThreeDimensionalInteractionCollection)
        , m_material_collection(nullptr)
        , m_3DInteraction_calleeSlot(
              "getInteraction", "The slot publishing available interactions and receiving pending manipulations")
        , m_3DInteraction_callerSlot("", "")
        , m_glTF_callerSlot("getGlTFFile", "Connects the data source with a loaded glTF file") {
    this->m_3DInteraction_calleeSlot.SetCallback(
        Call3DInteraction::ClassName(), "GetData", &ThreeDimensionalUIRenderTaskDataSource::getInteractionCallback);
    this->m_3DInteraction_calleeSlot.SetCallback(
        Call3DInteraction::ClassName(), "GetMetaData", &ThreeDimensionalUIRenderTaskDataSource::getInteractionCallback);
    this->MakeSlotAvailable(&this->m_3DInteraction_calleeSlot);

    this->m_glTF_callerSlot.SetCompatibleCall<CallGlTFDataDescription>();
    this->MakeSlotAvailable(&this->m_glTF_callerSlot);

    this->m_3DInteraction_callerSlot.SetCompatibleCall<Call3DInteractionDescription>();
    this->MakeSlotAvailable(&this->m_3DInteraction_callerSlot);
}

megamol::mesh::ThreeDimensionalUIRenderTaskDataSource::~ThreeDimensionalUIRenderTaskDataSource() {}

bool megamol::mesh::ThreeDimensionalUIRenderTaskDataSource::create(void) {
    AbstractGPURenderTaskDataSource::create();

    m_material_collection = std::make_shared<GPUMaterialCollection>();
    m_material_collection->addMaterial(this->instance(), "3DUI", "3DUI");

    return true;
}

bool megamol::mesh::ThreeDimensionalUIRenderTaskDataSource::getDataCallback(core::Call& caller) {
    CallGPURenderTaskData* lhs_rtc = dynamic_cast<CallGPURenderTaskData*>(&caller);
    if (lhs_rtc == nullptr) {
        return false;
    }

    CallGPURenderTaskData* rhs_rtc = this->m_renderTask_rhs_slot.CallAs<CallGPURenderTaskData>();

    std::vector<std::shared_ptr<GPURenderTaskCollection>> gpu_render_tasks;
    if (rhs_rtc != nullptr) {
        if (!(*rhs_rtc)(0)) {
            return false;
        }
        if (rhs_rtc->hasUpdate()) {
            ++m_version;
        }
        gpu_render_tasks = rhs_rtc->getData();
    }
    gpu_render_tasks.push_back(m_rendertask_collection.first);

    CallGPUMeshData* mc = this->m_mesh_slot.CallAs<CallGPUMeshData>();
    if (mc == NULL)
        return false;
    if (!(*mc)(0))
        return false;

    CallGlTFData* gltf_call = this->m_glTF_callerSlot.CallAs<CallGlTFData>();
    if (gltf_call == NULL)
        return false;
    if (!(*gltf_call)(0))
        return false;

    bool something_has_changed = mc->hasUpdate() || gltf_call->hasUpdate();

    if (something_has_changed) {
        ++m_version;

        clearRenderTaskCollection();

        auto model = gltf_call->getData().second;
        auto gpu_mesh_storage = mc->getData();

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

                    m_UI_template_elements[0].first = gpu_mesh_storage[0]->getSubMesh("axisX_arrow");
                    m_UI_template_elements[0].second = per_obj_data;

                } else if (model->nodes[node_idx].name == "axisY_arrow") {
                    per_obj_data[0].color = {0.0f, 1.0f, 0.0f, 1.0f};
                    per_obj_data[0].id = 0;

                    m_UI_template_elements[1].first = gpu_mesh_storage[0]->getSubMesh("axisY_arrow");
                    m_UI_template_elements[1].second = per_obj_data;

                } else if (model->nodes[node_idx].name == "axisZ_arrow") {
                    per_obj_data[0].color = {0.0f, 0.0f, 1.0f, 1.0f};
                    per_obj_data[0].id = 0;

                    m_UI_template_elements[2].first = gpu_mesh_storage[0]->getSubMesh("axisZ_arrow");
                    m_UI_template_elements[2].second = per_obj_data;

                } else if (model->nodes[node_idx].name == "slider_arrow") {
                    per_obj_data[0].color = {1.0f, 0.0f, 1.0f, 1.0f};
                    per_obj_data[0].id = 0;

                    m_UI_template_elements[3].first = gpu_mesh_storage[0]->getSubMesh("slider_arrow");
                    m_UI_template_elements[3].second = per_obj_data;
                }
            }
        }

        auto const& shader = m_material_collection->getMaterials().find("3DUI")->second.shader_program;

        int render_task_index = 0;
        {
            // TODO create debug scene from UI template obejcts
            m_scene.push_back({0, {PerObjectShaderParams()}});
            m_scene.back().second = m_UI_template_elements[3].second;
            m_scene.back().second[0].id = 1;

            auto const& sub_mesh = m_UI_template_elements[3].first;
            auto const& gpu_batch_mesh = sub_mesh.mesh->mesh;

            std::string rt_identifier(std::string(this->FullName()) + "_" + std::to_string(++render_task_index));
            m_rendertask_collection.first->addRenderTask(
                rt_identifier, shader, gpu_batch_mesh, sub_mesh.sub_mesh_draw_command, m_scene.back().second);
            m_rendertask_collection.second.push_back(rt_identifier);

            m_scene.back().first = m_rendertask_collection.second.back();

            m_interaction_collection->addInteractionObject(1,
                {ThreeDimensionalInteraction{InteractionType::MOVE_ALONG_AXIS, 1, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f}});
        }
        {
            m_scene.push_back({0, {PerObjectShaderParams()}});
            m_scene.back().second = m_UI_template_elements[2].second;
            m_scene.back().second[0].id = 2;

            auto const& sub_mesh = m_UI_template_elements[2].first;
            auto const& gpu_batch_mesh = sub_mesh.mesh->mesh;

            std::string rt_identifier(std::string(this->FullName()) + "_" + std::to_string(++render_task_index));
            m_rendertask_collection.first->addRenderTask(
                rt_identifier, shader, gpu_batch_mesh, sub_mesh.sub_mesh_draw_command, m_scene.back().second);
            m_rendertask_collection.second.push_back(rt_identifier);

            m_scene.back().first = m_rendertask_collection.second.back();

            m_interaction_collection->addInteractionObject(2,
                {ThreeDimensionalInteraction{InteractionType::MOVE_ALONG_AXIS, 2, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f}});
        }
        {
            m_scene.push_back({0, {PerObjectShaderParams()}});
            m_scene.back().second = m_UI_template_elements[1].second;
            m_scene.back().second[0].id = 3;

            auto const& sub_mesh = m_UI_template_elements[1].first;
            auto const& gpu_batch_mesh = sub_mesh.mesh->mesh;

            std::string rt_identifier(std::string(this->FullName()) + "_" + std::to_string(++render_task_index));
            m_rendertask_collection.first->addRenderTask(
                rt_identifier, shader, gpu_batch_mesh, sub_mesh.sub_mesh_draw_command, m_scene.back().second);
            m_rendertask_collection.second.push_back(rt_identifier);

            m_scene.back().first = m_rendertask_collection.second.back();

            m_interaction_collection->addInteractionObject(3,
                {ThreeDimensionalInteraction{InteractionType::MOVE_ALONG_AXIS, 3, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f}});
        }
        {
            m_scene.push_back({0, {PerObjectShaderParams()}});
            m_scene.back().second = m_UI_template_elements[0].second;
            m_scene.back().second[0].id = 4;

            auto const& sub_mesh = m_UI_template_elements[0].first;
            auto const& gpu_batch_mesh = sub_mesh.mesh->mesh;

            std::string rt_identifier(std::string(this->FullName()) + "_" + std::to_string(++render_task_index));
            m_rendertask_collection.first->addRenderTask(
                rt_identifier, shader, gpu_batch_mesh, sub_mesh.sub_mesh_draw_command, m_scene.back().second);
            m_rendertask_collection.second.push_back(rt_identifier);

            m_scene.back().first = m_rendertask_collection.second.back();

            m_interaction_collection->addInteractionObject(4,
                {ThreeDimensionalInteraction{InteractionType::MOVE_ALONG_AXIS, 4, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}});
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

        m_rendertask_collection.first->deletePerFrameDataBuffer(1);
        m_rendertask_collection.first->addPerFrameDataBuffer("lights", lights, 1);
    }

    lhs_rtc->setData(gpu_render_tasks, m_version);

    return true;
}

bool megamol::mesh::ThreeDimensionalUIRenderTaskDataSource::getInteractionCallback(core::Call& caller) {

    Call3DInteraction* ci = dynamic_cast<Call3DInteraction*>(&caller);
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
        ThreeDimensionalManipulation manipulation = ci->getData()->accessPendingManipulations().front();

        std::list<std::pair<std::string, std::array<PerObjectShaderParams, 1>>>::iterator it = m_scene.begin();
        std::advance(it, manipulation.obj_id - 1);

        std::array<PerObjectShaderParams, 1> per_obj_data = it->second;

        vislib::math::Vector<float, 4> translate_col;

        switch (manipulation.type) {
        case MOVE_ALONG_AXIS:
            translate_col = per_obj_data[0].object_transform.GetColumn(3);
            translate_col += vislib::math::Vector<float, 4>(manipulation.axis_x * manipulation.value,
                manipulation.axis_y * manipulation.value, manipulation.axis_Z * manipulation.value, 0.0f);
            per_obj_data[0].object_transform.SetAt(0, 3, translate_col.X());
            per_obj_data[0].object_transform.SetAt(1, 3, translate_col.Y());
            per_obj_data[0].object_transform.SetAt(2, 3, translate_col.Z());
            break;
        case MOVE_IN_PLANE:
            break;
        case ROTATE_AROUND_AXIS:
            break;
        case SELECT:
            break;
        case DESELET:
            break;
        case HIGHLIGHT:
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
        this->m_rendertask_collection.first->updatePerDrawData(entity.first, entity.second);
    }

    return true;
}
