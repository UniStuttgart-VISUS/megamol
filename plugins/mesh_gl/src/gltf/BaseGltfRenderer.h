/*
 * BaseGltfRenderer.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VISUS).
 * All rights reserved.
 */

#ifndef BASE_GLTF_RENDERER_H_INCLUDED
#define BASE_GLTF_RENDERER_H_INCLUDED

#include "mmcore/CallerSlot.h"

#include "mesh/MeshCalls.h"
#include "mesh_gl/BaseMeshRenderer.h"

namespace megamol::mesh_gl {

class BaseGltfRenderer : public BaseMeshRenderer {
public:
    BaseGltfRenderer();
    ~BaseGltfRenderer() override = default;

protected:
    void updateRenderTaskCollection(mmstd_gl::CallRender3DGL& call, bool force_update) override;

private:
    /** Slot to retrieve the gltf model */
    megamol::core::CallerSlot glTF_callerSlot_;
};

inline BaseGltfRenderer::BaseGltfRenderer()
        : BaseMeshRenderer()
        , glTF_callerSlot_("gltfModels", "Connects a collection of loaded glTF files") {
    glTF_callerSlot_.SetCompatibleCall<mesh::CallGlTFDataDescription>();
    megamol::core::Module::MakeSlotAvailable(&this->glTF_callerSlot_);
}

inline void BaseGltfRenderer::updateRenderTaskCollection(mmstd_gl::CallRender3DGL& call, bool force_update) {

    bool something_has_changed = force_update;

    mesh::CallGlTFData* gltf_call = this->glTF_callerSlot_.CallAs<mesh::CallGlTFData>();

    if (gltf_call != nullptr) {

        if (!(*gltf_call)(0)) {
            //return false;
        }

        something_has_changed |= gltf_call->hasUpdate();

        if (something_has_changed) {
            render_task_collection_->clear();

            auto model = gltf_call->getData().second;

            for (size_t node_idx = 0; node_idx < model->nodes.size(); node_idx++) {
                if (node_idx < model->nodes.size() && model->nodes[node_idx].mesh != -1) {
                    vislib::math::Matrix<GLfloat, 4, vislib::math::COLUMN_MAJOR> object_transform;

                    if (model->nodes[node_idx].matrix.size() != 0) // has matrix transform
                    {
                        // TODO
                    } else {
                        auto& translation = model->nodes[node_idx].translation;
                        auto& scale = model->nodes[node_idx].scale;
                        auto& rotation = model->nodes[node_idx].rotation;

                        if (translation.size() != 0) {
                            object_transform.SetAt(0, 3, translation[0]);
                            object_transform.SetAt(1, 3, translation[1]);
                            object_transform.SetAt(2, 3, translation[2]);
                        }

                        if (scale.size() != 0) {}

                        if (rotation.size() != 0) {}
                    }

                    auto primitive_cnt = model->meshes[model->nodes[node_idx].mesh].primitives.size();
                    for (size_t primitive_idx = 0; primitive_idx < primitive_cnt; ++primitive_idx) {
                        std::string sub_mesh_identifier = gltf_call->getData().first +
                                                          model->meshes[model->nodes[node_idx].mesh].name + "_" +
                                                          std::to_string(primitive_idx);

                        GPUMeshCollection::SubMeshData sub_mesh = mesh_collection_->getSubMesh(sub_mesh_identifier);

                        if (sub_mesh.mesh != nullptr) {
                            auto const& gpu_batch_mesh = sub_mesh.mesh->mesh;
                            auto const& shader = material_collection_->getMaterial("gltfMaterial").shader_program;
                            auto const& textured_shader =
                                material_collection_->getMaterial("gltfTexturedMaterial").shader_program;

                            auto material_idx =
                                model->meshes[model->nodes[node_idx].mesh].primitives[primitive_idx].material;

                            if (material_idx != -1) {
                                std::string material_name = "";
                                std::array<float, 4> base_colour = {1.0f, 0.0f, 1.0f, 1.0f};
                                float metalness = 0.0f;
                                float roughness = 0.8f;
                                std::array<float, 4> specular_colour = {1.0f, 1.0f, 1.0f, 1.0f};

                                std::vector<std::shared_ptr<glowl::Texture>> textures;
                                GLuint64 albedo_texture_handle;
                                GLuint64 normal_texture_handle;
                                GLuint64 metallicRoughness_texture_handle;

                                std::string identifier_string = "ga_" + model->nodes[node_idx].name + "_n_" +
                                                                std::to_string(node_idx) + "_p_" +
                                                                std::to_string(primitive_idx);

                                material_name = model->materials[material_idx].name.empty()
                                                    ? identifier_string
                                                    : model->materials[material_idx].name;

                                //TODO check if material has already been added

                                //std::copy_n(model->materials[material_idx].pbrMetallicRoughness.baseColorFactor.begin(),4, base_colour.begin());
                                metalness = model->materials[material_idx].pbrMetallicRoughness.metallicFactor;
                                roughness = model->materials[material_idx].pbrMetallicRoughness.roughnessFactor;

                                auto c = model->materials[material_idx].pbrMetallicRoughness.baseColorFactor;
                                base_colour = {static_cast<float>(c[0]) * (1.0f - metalness),
                                    static_cast<float>(c[1]) * (1.0f - metalness),
                                    static_cast<float>(c[2]) * (1.0f - metalness), static_cast<float>(c[3])};
                                // assume a specular color value of 0.04 (around plastic) as default value for dielectrics
                                specular_colour = {(static_cast<float>(c[0]) * metalness) + 0.04f * (1.0f - metalness),
                                    (static_cast<float>(c[1]) * metalness) + 0.04f * (1.0f - metalness),
                                    (static_cast<float>(c[2]) * metalness) + 0.04f * (1.0f - metalness),
                                    static_cast<float>(c[3])};

                                if (model->materials[material_idx].pbrMetallicRoughness.baseColorTexture.index != -1) {
                                    // base color texture (diffuse albedo)
                                    glowl::TextureLayout layout;

                                    auto& img =
                                        model->images[model
                                                          ->textures[model->materials[material_idx]
                                                                         .pbrMetallicRoughness.baseColorTexture.index]
                                                          .source];

                                    layout.width = img.width;
                                    layout.height = img.height;
                                    layout.depth = 1;
                                    layout.levels = 1;
                                    layout.type = img.pixel_type;
                                    layout.format =
                                        0x1908; // GL_RGBA, apparently tinygltf enforces 4 components for better vulkan compability anyway
                                    layout.internal_format = 0x8058; // GL_RGBA8

                                    layout.int_parameters = {{
                                                                 0x2801, // GL_TEXTURE_MIN_FILTER
                                                                 0x2703  //GL_LINEAR_MIPMAP_LINEAR
                                                             },
                                        {
                                            0x2800, //GL_TEXTURE_MAG_FILTER
                                            0x2601  //GL_LINEAR
                                        }};
                                    layout.float_parameters = {{0x84FE, //GL_TEXTURE_MAX_ANISOTROPY_EXT
                                        8.0f}};

                                    auto texture = std::make_shared<glowl::Texture2D>(
                                        material_name + "_baseColor", layout, img.image.data(), true);

                                    textures.emplace_back(texture);

                                    albedo_texture_handle = texture->getTextureHandle();
                                    if (!glIsTextureHandleResidentARB(albedo_texture_handle)) {
                                        texture->makeResident();
                                    }
                                }

                                if (model->materials[material_idx]
                                        .pbrMetallicRoughness.metallicRoughnessTexture.index != -1) {
                                    // metallic roughness texture
                                    glowl::TextureLayout layout;

                                    auto& img =
                                        model->images
                                            [model->textures[model->materials[material_idx]
                                                                 .pbrMetallicRoughness.metallicRoughnessTexture.index]
                                                    .source];

                                    layout.width = img.width;
                                    layout.height = img.height;
                                    layout.depth = 1;
                                    layout.levels = 1;
                                    layout.type = img.pixel_type;
                                    layout.format =
                                        0x1908; // GL_RGBA, apparently tinygltf enforces 4 components for better vulkan compability anyway
                                    layout.internal_format = 0x8058; // GL_RGBA8

                                    layout.int_parameters = {{
                                                                 0x2801, // GL_TEXTURE_MIN_FILTER
                                                                 0x2703  //GL_LINEAR_MIPMAP_LINEAR
                                                             },
                                        {
                                            0x2800, //GL_TEXTURE_MAG_FILTER
                                            0x2601  //GL_LINEAR
                                        }};
                                    layout.float_parameters = {{0x84FE, //GL_TEXTURE_MAX_ANISOTROPY_EXT
                                        8.0f}};

                                    auto texture = std::make_shared<glowl::Texture2D>(
                                        material_name + "_metallicRoughness", layout, img.image.data(), true);

                                    textures.emplace_back(texture);

                                    metallicRoughness_texture_handle = texture->getTextureHandle();
                                    if (!glIsTextureHandleResidentARB(metallicRoughness_texture_handle)) {
                                        texture->makeResident();
                                    }
                                }

                                if (model->materials[material_idx].normalTexture.index != -1) {
                                    // normal map texture
                                    glowl::TextureLayout layout;

                                    auto& img =
                                        model
                                            ->images[model->textures[model->materials[material_idx].normalTexture.index]
                                                         .source];

                                    layout.width = img.width;
                                    layout.height = img.height;
                                    layout.depth = 1;
                                    layout.levels = 1;
                                    layout.type = img.pixel_type;
                                    layout.format =
                                        0x1908; // GL_RGBA, apparently tinygltf enforces 4 components for better vulkan compability anyway
                                    layout.internal_format = 0x8058; // GL_RGBA8

                                    layout.int_parameters = {{
                                                                 0x2801, // GL_TEXTURE_MIN_FILTER
                                                                 0x2703  //GL_LINEAR_MIPMAP_LINEAR
                                                             },
                                        {
                                            0x2800, //GL_TEXTURE_MAG_FILTER
                                            0x2601  //GL_LINEAR
                                        }};
                                    layout.float_parameters = {{0x84FE, //GL_TEXTURE_MAX_ANISOTROPY_EXT
                                        8.0f}};

                                    auto texture = std::make_shared<glowl::Texture2D>(
                                        material_name + "_normal", layout, img.image.data(), true);

                                    textures.emplace_back(texture);

                                    normal_texture_handle = texture->getTextureHandle();
                                    if (!glIsTextureHandleResidentARB(normal_texture_handle)) {
                                        texture->makeResident();
                                    }
                                }

                                material_collection_->addMaterial(material_name, textured_shader, textures);

                                struct PerObjectData {
                                    vislib::math::Matrix<GLfloat, 4, vislib::math::COLUMN_MAJOR> transform;
                                    GLuint64 albedo_texture_handle;
                                    GLuint64 normal_texture_handle;
                                    GLuint64 metallicRoughness_texture_handle;
                                };
                                PerObjectData per_obj_data;
                                per_obj_data.transform = object_transform;
                                per_obj_data.albedo_texture_handle = albedo_texture_handle;
                                per_obj_data.metallicRoughness_texture_handle = metallicRoughness_texture_handle;
                                per_obj_data.normal_texture_handle = normal_texture_handle;

                                std::string rt_identifier(std::string(this->FullName()) + "_" + sub_mesh_identifier);
                                render_task_collection_->addRenderTask(rt_identifier, textured_shader, gpu_batch_mesh,
                                    sub_mesh.sub_mesh_draw_command, per_obj_data);
                            } else {
                                struct PerObjectData {
                                    vislib::math::Matrix<GLfloat, 4, vislib::math::COLUMN_MAJOR> transform;
                                };
                                PerObjectData per_obj_data;
                                per_obj_data.transform = object_transform;

                                std::string rt_identifier(std::string(this->FullName()) + "_" + sub_mesh_identifier);
                                render_task_collection_->addRenderTask(rt_identifier, shader, gpu_batch_mesh,
                                    sub_mesh.sub_mesh_draw_command, per_obj_data);
                            }
                        }
                    }
                }
            }
        }
    } else {
        render_task_collection_->clear();
    }
}


} // namespace megamol::mesh_gl

#endif // !BASE_GLTF_RENDERER_H_INCLUDED
