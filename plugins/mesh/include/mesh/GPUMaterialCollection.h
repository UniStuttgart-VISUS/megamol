/*
 * GPURenderTaskCollection.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VISUS).
 * All rights reserved.
 */

#ifndef GPU_MATERIAL_COLLECTION_H_INCLUDED
#define GPU_MATERIAL_COLLECTION_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#    pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

//#include "vislib/graphics/gl/GLSLShader.h"
#include "mmcore/CoreInstance.h"
#include "vislib/graphics/gl/GLSLGeometryShader.h"

#include <memory>
#include <vector>

#include "mesh.h"

#include "glowl/Texture2D.h"

namespace megamol {
namespace mesh {

typedef vislib::graphics::gl::GLSLGeometryShader Shader;

class MESH_API GPUMaterialCollecton {
public:
    struct Material {
        std::shared_ptr<Shader> shader_program;
        std::vector<GLuint> textures_names;
    };

    void addMaterial(
        megamol::core::CoreInstance* mm_core_inst, std::string shader_btf_name, std::vector<GLuint> texture_names = {});

    void addMaterial(std::shared_ptr<Shader> const& shader, std::vector<GLuint> texture_names);

    void updateMaterialTexture(size_t mtl_idx, size_t tex_idx, GLuint texture_name);

    void clearMaterials();

    inline std::vector<Material> const& getMaterials();

private:
    std::vector<Material> m_materials;

    /**
     * Storage for textures created and managed by this class
     */
    std::vector<std::shared_ptr<Texture2D>> m_texture;
};

inline std::vector<GPUMaterialCollecton::Material> const& GPUMaterialCollecton::getMaterials() { return m_materials; }

} // namespace mesh
} // namespace megamol

#endif // !GPU_MATERIAL_COLLECTION_H_INCLUDED
