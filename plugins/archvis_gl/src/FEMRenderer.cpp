/**
 * MegaMol
 * Copyright (c) 2019, MegaMol Dev Team
 * All rights reserved.
 */

#include "FEMRenderer.h"

#include <variant>

#include "ArchVisCalls.h"
#include "FEMRenderer.h"
#include "mesh_gl/GPUMeshCollection.h"

megamol::archvis_gl::FEMRenderer::FEMRenderer()
        : m_fem_callerSlot("getFEMFile", "Connects the data source with loaded FEM data") {
    this->m_fem_callerSlot.SetCompatibleCall<FEMModelCallDescription>();
    this->MakeSlotAvailable(&this->m_fem_callerSlot);
}

megamol::archvis_gl::FEMRenderer::~FEMRenderer() {}

bool megamol::archvis_gl::FEMRenderer::GetExtents(mmstd_gl::CallRender3DGL& call) {
    return false;
}

void megamol::archvis_gl::FEMRenderer::createMaterialCollection() {
    material_collection_ = std::make_shared<mesh_gl::GPUMaterialCollection>();
    material_collection_->addMaterial(frontend_resources.get<megamol::frontend_resources::RuntimeConfig>(),
        "ArchVisFEM", {"archvis_gl/FEM/fem.vert.glsl", "archvis_gl/FEM/fem.geom.glsl", "archvis_gl/FEM/fem_frag.glsl"});
}

bool megamol::archvis_gl::FEMRenderer::updateMeshCollection() {
    bool retval = false;

    CallFEMModel* fem_call = this->m_fem_callerSlot.CallAs<CallFEMModel>();
    if (fem_call != nullptr) {
        if (!(*fem_call)(0)) {
            return false;
        }

        retval = fem_call->hasUpdate();

        if (fem_call->hasUpdate()) {
            mesh_collection_->clear();
            retval = true;

            auto fem_data = fem_call->getData();

            // TODO generate vertex and index data

            // Create std-container for holding vertex data
            std::vector<std::vector<float>> vbs(1);
            vbs[0].reserve(fem_data->getNodes().size() * 3);
            for (auto& node : fem_data->getNodes()) {
                vbs[0].push_back(node.X()); // position data buffer
                vbs[0].push_back(node.Y());
                vbs[0].push_back(node.Z());
            }
            // Create std-container holding vertex attribute descriptions
            std::vector<glowl::VertexLayout::Attribute> attribs = {
                glowl::VertexLayout::Attribute(3, GL_FLOAT, GL_FALSE, 0)};
            glowl::VertexLayout vertex_descriptor(0, attribs);

            // Create std-container holding index data
            std::vector<uint32_t> indices;
            std::vector<size_t> node_indices;

            for (auto& element : fem_data->getElements()) {
                switch (element.getType()) {
                case FEMModel::CUBE:

                    // TODO indices for a cube....
                    node_indices = element.getNodeIndices();

                    indices.insert(indices.end(),
                        {// front
                            static_cast<uint32_t>(node_indices[0] - 1), static_cast<uint32_t>(node_indices[1] - 1),
                            static_cast<uint32_t>(node_indices[2] - 1), static_cast<uint32_t>(node_indices[2] - 1),
                            static_cast<uint32_t>(node_indices[3] - 1), static_cast<uint32_t>(node_indices[0] - 1),
                            // right
                            static_cast<uint32_t>(node_indices[1] - 1), static_cast<uint32_t>(node_indices[5] - 1),
                            static_cast<uint32_t>(node_indices[6] - 1), static_cast<uint32_t>(node_indices[6] - 1),
                            static_cast<uint32_t>(node_indices[2] - 1), static_cast<uint32_t>(node_indices[1] - 1),
                            // back
                            static_cast<uint32_t>(node_indices[7] - 1), static_cast<uint32_t>(node_indices[6] - 1),
                            static_cast<uint32_t>(node_indices[5] - 1), static_cast<uint32_t>(node_indices[5] - 1),
                            static_cast<uint32_t>(node_indices[4] - 1), static_cast<uint32_t>(node_indices[7] - 1),
                            // left
                            static_cast<uint32_t>(node_indices[4] - 1), static_cast<uint32_t>(node_indices[0] - 1),
                            static_cast<uint32_t>(node_indices[3] - 1), static_cast<uint32_t>(node_indices[3] - 1),
                            static_cast<uint32_t>(node_indices[7] - 1), static_cast<uint32_t>(node_indices[4] - 1),
                            // bottom
                            static_cast<uint32_t>(node_indices[4] - 1), static_cast<uint32_t>(node_indices[5] - 1),
                            static_cast<uint32_t>(node_indices[1] - 1), static_cast<uint32_t>(node_indices[1] - 1),
                            static_cast<uint32_t>(node_indices[0] - 1), static_cast<uint32_t>(node_indices[4] - 1),
                            // top
                            static_cast<uint32_t>(node_indices[3] - 1), static_cast<uint32_t>(node_indices[2] - 1),
                            static_cast<uint32_t>(node_indices[6] - 1), static_cast<uint32_t>(node_indices[6] - 1),
                            static_cast<uint32_t>(node_indices[7] - 1), static_cast<uint32_t>(node_indices[3] - 1)});

                    break;
                default:
                    break;
                }
            }

            std::vector<glowl::VertexLayout> vb_layouts = {vertex_descriptor};
            std::vector<std::pair<std::vector<float>::iterator, std::vector<float>::iterator>> vb_iterators = {
                {vbs[0].begin(), vbs[0].end()}};
            std::pair<std::vector<uint32_t>::iterator, std::vector<uint32_t>::iterator> ib_iterators = {
                indices.begin(), indices.end()};

            std::string identifier = std::string(this->FullName());

            try {
                mesh_collection_->addMesh(identifier, vb_layouts, vb_iterators, ib_iterators, GL_UNSIGNED_INT,
                    GL_STATIC_DRAW, GL_TRIANGLES, true);
            } catch (glowl::MeshException const& exc) {
                megamol::core::utility::log::Log::DefaultLog.WriteError(
                    "Failed to add GPU mesh \"%s\": %s. [%s, %s, line %d]\n", identifier.c_str(), exc.what(), __FILE__,
                    __FUNCTION__, __LINE__);

            } catch (glowl::BufferObjectException const& exc) {
                megamol::core::utility::log::Log::DefaultLog.WriteError(
                    "Failed to add GPU mesh \"%s\": %s. [%s, %s, line %d]\n", identifier.c_str(), exc.what(), __FILE__,
                    __FUNCTION__, __LINE__);
            }
        }

    } else {
        if (mesh_collection_->getMeshes().size() > 0) {
            mesh_collection_->clear();
            retval = true;
        }
    }
}

void megamol::archvis_gl::FEMRenderer::updateRenderTaskCollection(mmstd_gl::CallRender3DGL& call, bool force_update) {
    bool something_has_changed = force_update;

    CallFEMModel* fem_call = this->m_fem_callerSlot.CallAs<CallFEMModel>();
    if (fem_call != nullptr) {
        if (!(*fem_call)(0)) {
            // TODO throw error
        }

        something_has_changed |= fem_call->hasUpdate();

        if (something_has_changed) {
            render_task_collection_->clear();

            for (auto& sub_mesh : mesh_collection_->getSubMeshData()) {
                auto const& gpu_batch_mesh = sub_mesh.second.mesh->mesh;

                auto const& shader = material_collection_->getMaterial("ArchVisFEM").shader_program;

                vislib::math::Matrix<GLfloat, 4, vislib::math::COLUMN_MAJOR> object_transform;
                typedef std::vector<vislib::math::Matrix<GLfloat, 4, vislib::math::COLUMN_MAJOR>> PerTaskData;

                GLfloat scale = 1.0f;
                object_transform.SetAt(0, 0, scale);
                object_transform.SetAt(1, 1, scale);
                object_transform.SetAt(2, 2, scale);

                object_transform.SetAt(3, 3, 1.0f);

                object_transform.SetAt(0, 3, 0.0f);
                object_transform.SetAt(1, 3, 0.0f);
                object_transform.SetAt(2, 3, 0.0f);

                auto identifier = std::string(this->FullName()) + sub_mesh.first;
                render_task_collection_->addRenderTask(
                    identifier, shader, gpu_batch_mesh, sub_mesh.second.sub_mesh_draw_command, object_transform);
            }

            auto const& node_deformation = fem_call->getData()->getDynamicData();

            render_task_collection_->addPerFrameDataBuffer("", node_deformation, 1);

            //{
            //    // TODO get transfer function texture and add as per frame data
            //    std::vector<GLuint64> texture_handles;
            //    auto textures = gpu_mtl_storage->getMaterials().front().textures;
            //    for (auto texture : textures) {
            //
            //        texture_handles.push_back(texture->getTextureHandle());
            //        // base_texture->makeResident();
            //    }
            //    m_gpu_render_tasks->updatePerFrameDataBuffer("", texture_handles, 2);
            //}
        }
    }
}
