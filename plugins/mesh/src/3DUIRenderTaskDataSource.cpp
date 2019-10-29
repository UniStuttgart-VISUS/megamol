#include "stdafx.h"

#include "mesh/3DUIRenderTaskDataSource.h"

#include "tiny_gltf.h"
#include "vislib/graphics/gl/IncludeAllGL.h"
#include "vislib/math/Matrix.h"

#include "mesh/MeshCalls.h"


megamol::mesh::ThreeDimensionalUIRenderTaskDataSource::ThreeDimensionalUIRenderTaskDataSource()
    : m_interaction_collection(new ThreeDimensionalInteractionCollection)
    , m_3DInteraction_calleeSlot("getInteraction", "The slot publishing available interactions and receiving pending manipulations")
    , m_3DInteraction_callerSlot("","")
    , m_glTF_callerSlot("getGlTFFile", "Connects the data source with a loaded glTF file")
    , m_glTF_cached_hash(0)
{
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

bool megamol::mesh::ThreeDimensionalUIRenderTaskDataSource::getDataCallback(core::Call& caller)
{
    CallGPURenderTaskData* lhs_rtc = dynamic_cast<CallGPURenderTaskData*>(&caller);
    if (lhs_rtc == NULL) return false;

    std::shared_ptr<GPURenderTaskCollection> rt_collection(nullptr);

    if (lhs_rtc->getData() == nullptr) {
        rt_collection = this->m_gpu_render_tasks;
        lhs_rtc->setData(rt_collection);
    } else {
        rt_collection = lhs_rtc->getData();
    }

    CallGPUMaterialData* mtlc = this->m_material_slot.CallAs<CallGPUMaterialData>();
    if (mtlc == NULL) return false;
    if (!(*mtlc)(0)) return false;

    CallGPUMeshData* mc = this->m_mesh_slot.CallAs<CallGPUMeshData>();
    if (mc == NULL) return false;
    if (!(*mc)(0)) return false;

    CallGlTFData* gltf_call = this->m_glTF_callerSlot.CallAs<CallGlTFData>();
    if (gltf_call == NULL) return false;
    if (!(*gltf_call)(0)) return false;

    auto gpu_mtl_storage = mtlc->getData();
    auto gpu_mesh_storage = mc->getData();

    // TODO nullptr check

    if (gltf_call->getMetaData().m_data_hash > m_glTF_cached_hash)
    {
        m_glTF_cached_hash = gltf_call->getMetaData().m_data_hash;

        // rt_collection->clear();
        if (!m_rt_collection_indices.empty()) {
            // TODO delete all exisiting render task from this module
            for (auto& rt_idx : m_rt_collection_indices) {
                rt_collection->deleteSingleRenderTask(rt_idx);
            }

            m_rt_collection_indices.clear();
        }

        auto model = gltf_call->getData();

        for (size_t node_idx = 0; node_idx < model->nodes.size(); node_idx++) {
            if (node_idx < model->nodes.size() && model->nodes[node_idx].mesh != -1) {

                std::array<PerObjectShaderParams,1> per_obj_data;

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

                    if (scale.size() != 0) {
                    }

                    if (rotation.size() != 0) {
                    }
                }

                // compute submesh offset by iterating over all meshes before the given mesh and summing up their
                // primitive counts
                size_t submesh_offset = 0;
                for (int mesh_idx = 0; mesh_idx < model->nodes[node_idx].mesh; ++mesh_idx) {
                    submesh_offset += model->meshes[mesh_idx].primitives.size();
                }

                GPUMeshCollection::SubMeshData sub_mesh_data;

                auto primitive_cnt = model->meshes[model->nodes[node_idx].mesh].primitives.size();
                for (size_t primitive_idx = 0; primitive_idx < primitive_cnt; ++primitive_idx) {
                    // auto const& sub_mesh = gpu_mesh_storage->getSubMeshData()[model->nodes[node_idx].mesh];
                    auto const& sub_mesh = gpu_mesh_storage->getSubMeshData()[submesh_offset + primitive_idx];
                    auto const& gpu_batch_mesh = gpu_mesh_storage->getMeshes()[sub_mesh.batch_index].mesh;
                    auto const& shader = gpu_mtl_storage->getMaterials().front().shader_program;

                    sub_mesh_data = sub_mesh;

                    //size_t rt_idx = rt_collection->addSingleRenderTask(
                    //    shader, gpu_batch_mesh, sub_mesh.sub_mesh_draw_command, per_obj_data);
                    //
                    //m_rt_collection_indices.push_back(rt_idx);
                }


                // TODO check node name for UI element names
                if (model->nodes[node_idx].name == "axisX_arrow") {
                    per_obj_data[0].color = {1.0f, 0.0f, 0.0f, 1.0f};
                    per_obj_data[0].id = 0;

                    m_UI_template_elements[0].first = sub_mesh_data;
                    m_UI_template_elements[0].second = per_obj_data;

                } else if (model->nodes[node_idx].name == "axisY_arrow") {
                    per_obj_data[0].color = {0.0f, 1.0f, 0.0f, 1.0f};
                    per_obj_data[0].id = 0;

                    m_UI_template_elements[1].first = sub_mesh_data;
                    m_UI_template_elements[1].second = per_obj_data;

                } else if (model->nodes[node_idx].name == "axisZ_arrow") {
                    per_obj_data[0].color = {0.0f, 0.0f, 1.0f, 1.0f};
                    per_obj_data[0].id = 0;

                    m_UI_template_elements[2].first = sub_mesh_data;
                    m_UI_template_elements[2].second = per_obj_data;

                } else if (model->nodes[node_idx].name == "slider_arrow") {
                    per_obj_data[0].color = {1.0f, 0.0f, 1.0f, 1.0f};
                    per_obj_data[0].id = 0;

                    m_UI_template_elements[3].first = sub_mesh_data;
                    m_UI_template_elements[3].second = per_obj_data;
                }

            }
        }

        {
            //TODO create debug scene from UI template obejcts
            m_scene.push_back({0,{PerObjectShaderParams()}});
            m_scene.back().second = m_UI_template_elements[3].second;
            m_scene.back().second[0].id = 1;
            
            auto const& sub_mesh = m_UI_template_elements[3].first;
            auto const& gpu_batch_mesh = gpu_mesh_storage->getMeshes()[sub_mesh.batch_index].mesh;
            auto const& shader = gpu_mtl_storage->getMaterials().front().shader_program;

            size_t rt_idx = rt_collection->addSingleRenderTask(shader, gpu_batch_mesh, sub_mesh.sub_mesh_draw_command, m_scene.back().second);
            
            m_rt_collection_indices.push_back(rt_idx);
            m_scene.back().first = rt_idx;

            m_interaction_collection->addInteractionObject(1,{ThreeDimensionalInteraction{InteractionType::MOVE_ALONG_AXIS, 1, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f}});
        }
        {
            m_scene.push_back({0, {PerObjectShaderParams()}});
            m_scene.back().second = m_UI_template_elements[2].second;
            m_scene.back().second[0].id = 2;

            auto const& sub_mesh = m_UI_template_elements[2].first;
            auto const& gpu_batch_mesh = gpu_mesh_storage->getMeshes()[sub_mesh.batch_index].mesh;
            auto const& shader = gpu_mtl_storage->getMaterials().front().shader_program;

            size_t rt_idx = rt_collection->addSingleRenderTask(
                shader, gpu_batch_mesh, sub_mesh.sub_mesh_draw_command, m_scene.back().second);

            m_rt_collection_indices.push_back(rt_idx);
            m_scene.back().first = rt_idx;

            m_interaction_collection->addInteractionObject(2,
                {ThreeDimensionalInteraction{InteractionType::MOVE_ALONG_AXIS, 2, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f}});
        }
        {
            m_scene.push_back({0, {PerObjectShaderParams()}});
            m_scene.back().second = m_UI_template_elements[1].second;
            m_scene.back().second[0].id = 3;

            auto const& sub_mesh = m_UI_template_elements[1].first;
            auto const& gpu_batch_mesh = gpu_mesh_storage->getMeshes()[sub_mesh.batch_index].mesh;
            auto const& shader = gpu_mtl_storage->getMaterials().front().shader_program;

            size_t rt_idx = rt_collection->addSingleRenderTask(
                shader, gpu_batch_mesh, sub_mesh.sub_mesh_draw_command, m_scene.back().second);

            m_rt_collection_indices.push_back(rt_idx);
            m_scene.back().first = rt_idx;

            m_interaction_collection->addInteractionObject(3,
                {ThreeDimensionalInteraction{InteractionType::MOVE_ALONG_AXIS, 3, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f}});
        }
        {
            m_scene.push_back({0, {PerObjectShaderParams()}});
            m_scene.back().second = m_UI_template_elements[0].second;
            m_scene.back().second[0].id = 4;

            auto const& sub_mesh = m_UI_template_elements[0].first;
            auto const& gpu_batch_mesh = gpu_mesh_storage->getMeshes()[sub_mesh.batch_index].mesh;
            auto const& shader = gpu_mtl_storage->getMaterials().front().shader_program;

            size_t rt_idx = rt_collection->addSingleRenderTask(
                shader, gpu_batch_mesh, sub_mesh.sub_mesh_draw_command, m_scene.back().second);

            m_rt_collection_indices.push_back(rt_idx);
            m_scene.back().first = rt_idx;

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

        rt_collection->addPerFrameDataBuffer(lights, 1);

    }

    CallGPURenderTaskData* rhs_rtc = this->m_renderTask_rhs_slot.CallAs<CallGPURenderTaskData>();
    if (rhs_rtc != NULL) {
        rhs_rtc->setData(rt_collection);

        (*rhs_rtc)(0);
    }

    return true;
}

bool megamol::mesh::ThreeDimensionalUIRenderTaskDataSource::getInteractionCallback(core::Call& caller) {

    Call3DInteraction* ci = dynamic_cast<Call3DInteraction*>(&caller);
    if (ci == NULL) return false;

    if (ci->getData() == nullptr){
        ci->setData(this->m_interaction_collection);
    }

    // clear non persistent changes, such has highlighting
    for (auto& entity : m_scene) {
        entity.second[0].highlighted = 0;
    }

    // TODO consume pending manipulations
    while(!ci->getData()->accessPendingManipulations().empty())
    {
        ThreeDimensionalManipulation manipulation = ci->getData()->accessPendingManipulations().front();

        std::list<std::pair<uint32_t, std::array<PerObjectShaderParams, 1>>>::iterator it = m_scene.begin();
        std::advance(it, manipulation.obj_id-1);

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
            //per_obj_data[0].color = {1.0f, 1.0f, 0.0f, 1.0f};a
            break;
        default:
            break;
        }

        it->second = per_obj_data; // overwrite object data for persistent change

        ci->getData()->accessPendingManipulations().pop();
    }

    // update all per obj data buffers
    for (auto& entity : m_scene) {
        this->m_gpu_render_tasks->updatePerDrawData(entity.first, entity.second);
    }

    return true; 
}
