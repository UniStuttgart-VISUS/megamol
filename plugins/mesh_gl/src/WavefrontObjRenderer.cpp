#include "WavefrontObjRenderer.h"

#include "mmstd/light/CallLight.h"
#include "mmstd/light/DistantLight.h"
#include "mmstd/light/PointLight.h"
#include "mmstd/light/TriDirectionalLighting.h"

#include "mesh/MeshCalls.h"

megamol::mesh_gl::WavefrontObjRenderer::WavefrontObjRenderer()
        : BaseMeshRenderer()
        , lights_slot_("lights", "Connects a chain of lights") {
    lights_slot_.SetCompatibleCall<megamol::core::view::light::CallLightDescription>();
    megamol::core::Module::MakeSlotAvailable(&this->lights_slot_);
}

megamol::mesh_gl::WavefrontObjRenderer::~WavefrontObjRenderer() {}

void megamol::mesh_gl::WavefrontObjRenderer::createMaterialCollection() {
    material_collection_ = std::make_shared<GPUMaterialCollection>();
    material_collection_->addMaterial(frontend_resources.get<megamol::frontend_resources::RuntimeConfig>(),
        "wavefrontObjMaterial", {"mesh_gl/wavefrontObj_example.vert.glsl", "mesh_gl/wavefrontObj_example.frag.glsl"});
}

void megamol::mesh_gl::WavefrontObjRenderer::updateRenderTaskCollection(
    mmstd_gl::CallRender3DGL& call, bool force_update) {

    // Obtain camera information
    core::view::Camera cam = call.GetCamera();
    auto cam_pose = cam.getPose();
    glm::mat4 view_mx = cam.getViewMatrix();

    bool something_has_changed = force_update;

    auto call_light = lights_slot_.CallAs<core::view::light::CallLight>();
    if (call_light != nullptr) {
        if (!(*call_light)(0)) {
            // TODO throw error
        }
    }

    something_has_changed |= call_light->hasUpdate();

    if (something_has_changed) {
        render_task_collection_->clear();

        std::vector<std::vector<std::string>> identifiers;
        std::vector<std::vector<glowl::DrawElementsCommand>> draw_commands;
        std::vector<std::vector<std::array<float, 16>>> object_transforms;
        std::vector<std::shared_ptr<glowl::Mesh>> batch_meshes;

        std::shared_ptr<glowl::Mesh> prev_mesh(nullptr);

        for (auto& sub_mesh : mesh_collection_->getSubMeshData()) {
            auto const& gpu_batch_mesh = sub_mesh.second.mesh->mesh;

            if (gpu_batch_mesh != prev_mesh) {
                identifiers.emplace_back(std::vector<std::string>());
                draw_commands.emplace_back(std::vector<glowl::DrawElementsCommand>());
                object_transforms.emplace_back(std::vector<std::array<float, 16>>());
                batch_meshes.push_back(gpu_batch_mesh);

                prev_mesh = gpu_batch_mesh;
            }

            float scale = 1.0f;
            std::array<float, 16> obj_xform = {
                scale, 0.0f, 0.0f, 0.0f, 0.0f, scale, 0.0f, 0.0f, 0.0f, 0.0f, scale, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f};

            identifiers.back().emplace_back(std::string(this->FullName()) + sub_mesh.first);
            draw_commands.back().push_back(sub_mesh.second.sub_mesh_draw_command);
            object_transforms.back().push_back(obj_xform);
        }

        // just take the first available shader as shader selection will eventually be integrated into this module

        for (int i = 0; i < batch_meshes.size(); ++i) {
            auto const& shader = material_collection_->getMaterial("wavefrontObjMaterial").shader_program;
            render_task_collection_->addRenderTasks(
                identifiers[i], shader, batch_meshes[i], draw_commands[i], object_transforms[i]);
        }

        {
            auto lights = call_light->getData();

            std::vector<std::array<float, 4>> point_lights_data;
            std::vector<std::array<float, 4>> distant_lights_data;

            auto point_lights = lights.get<core::view::light::PointLightType>();
            auto distant_lights = lights.get<core::view::light::DistantLightType>();
            auto tridirection_lights = lights.get<core::view::light::TriDirectionalLightType>();

            for (auto pl : point_lights) {
                point_lights_data.push_back({pl.position[0], pl.position[1], pl.position[2], pl.intensity});
            }

            for (auto dl : distant_lights) {
                if (dl.eye_direction) {
                    auto cam_dir = glm::normalize(cam_pose.direction);
                    distant_lights_data.push_back({cam_dir.x, cam_dir.y, cam_dir.z, dl.intensity});
                } else {
                    distant_lights_data.push_back({dl.direction[0], dl.direction[1], dl.direction[2], dl.intensity});
                }
            }

            for (auto tdl : tridirection_lights) {
                if (tdl.in_view_space) {
                    auto inverse_view = glm::transpose(glm::mat3(view_mx));
                    auto key_dir =
                        inverse_view * glm::vec3(tdl.key_direction[0], tdl.key_direction[1], tdl.key_direction[2]);
                    auto fill_dir =
                        inverse_view * glm::vec3(tdl.fill_direction[0], tdl.fill_direction[1], tdl.fill_direction[2]);
                    auto back_dir =
                        inverse_view * glm::vec3(tdl.back_direction[0], tdl.back_direction[1], tdl.back_direction[2]);
                    distant_lights_data.push_back({key_dir[0], key_dir[1], key_dir[2], tdl.intensity});
                    distant_lights_data.push_back({fill_dir[0], fill_dir[1], fill_dir[2], tdl.intensity});
                    distant_lights_data.push_back({back_dir[0], back_dir[1], back_dir[2], tdl.intensity});
                } else {
                    distant_lights_data.push_back(
                        {tdl.key_direction[0], tdl.key_direction[1], tdl.key_direction[2], tdl.intensity});
                    distant_lights_data.push_back(
                        {tdl.fill_direction[0], tdl.fill_direction[1], tdl.fill_direction[2], tdl.intensity});
                    distant_lights_data.push_back(
                        {tdl.back_direction[0], tdl.back_direction[1], tdl.back_direction[2], tdl.intensity});
                }
            }

            // add some lights to the scene to test the per frame buffers
            struct LightMetaInfo {
                int point_light_cnt;
                int directional_light_cnt;
            };
            std::array<LightMetaInfo, 1> light_meta_info{point_lights_data.size(), distant_lights_data.size()};
            render_task_collection_->deletePerFrameDataBuffer(1);
            render_task_collection_->addPerFrameDataBuffer("light_meta_info", light_meta_info, 1);

            render_task_collection_->deletePerFrameDataBuffer(2);
            render_task_collection_->addPerFrameDataBuffer("point_lights", point_lights_data, 2);

            render_task_collection_->deletePerFrameDataBuffer(3);
            render_task_collection_->addPerFrameDataBuffer("directional_lights", distant_lights_data, 3);
        }
    }
}
