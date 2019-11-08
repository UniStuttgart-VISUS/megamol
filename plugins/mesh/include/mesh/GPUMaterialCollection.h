/*
 * GPURenderTaskCollection.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VISUS).
 * All rights reserved.
 */

#ifndef GPU_MATERIAL_COLLECTION_H_INCLUDED
#define GPU_MATERIAL_COLLECTION_H_INCLUDED

//#include "vislib/graphics/gl/GLSLShader.h"
#include "mmcore/CoreInstance.h"
#include "vislib/graphics/gl/GLSLGeometryShader.h"

#include <memory>
#include <variant>
#include <vector>

#include "mesh.h"

#include "glowl/GLSLProgram.hpp"
#include "glowl/Texture.hpp"
#include "glowl/Texture2D.hpp"
#include "glowl/Texture2DArray.hpp"
#include "glowl/Texture3D.hpp"
#include "glowl/TextureCubemapArray.hpp"

namespace megamol {
namespace mesh {

typedef glowl::GLSLProgram Shader;

class MESH_API GPUMaterialCollecton {
public:
    using TexturePtrType =
        std::variant <
            std::shared_ptr<glowl::Texture>,
            std::shared_ptr<glowl::Texture2D>,
            std::shared_ptr<glowl::Texture2DArray>,
            std::shared_ptr<glowl::Texture3D>,
            std::shared_ptr<glowl::TextureCubemapArray> >;

    struct Material {
        std::shared_ptr<Shader> shader_program;
        std::vector<std::shared_ptr<glowl::Texture>> textures;
    };

    void addMaterial(
        megamol::core::CoreInstance* mm_core_inst,
        std::string const& shader_btf_name,
        std::vector<std::shared_ptr<glowl::Texture>> const& textures = {});

    void addMaterial(
        std::shared_ptr<Shader> const& shader, 
        std::vector<std::shared_ptr<glowl::Texture>> const& textures = {});

    void updateMaterialTexture(size_t mtl_idx, size_t tex_idx, std::shared_ptr<glowl::Texture> const& texture);

    void clearMaterials();

    inline std::vector<Material> const& getMaterials();

private:
    std::vector<Material> m_materials;
};

inline std::vector<GPUMaterialCollecton::Material> const& GPUMaterialCollecton::getMaterials() { return m_materials; }

} // namespace mesh
} // namespace megamol

#endif // !GPU_MATERIAL_COLLECTION_H_INCLUDED
