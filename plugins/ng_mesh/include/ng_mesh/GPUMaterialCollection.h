/*
* GPURenderTaskCollection.h
*
* Copyright (C) 2019 by Universitaet Stuttgart (VISUS).
* All rights reserved.
*/

#ifndef GPU_MATERIAL_DATA_STORAGE_H_INCLUDED
#define GPU_MATERIAL_DATA_STORAGE_H_INCLUDED

//#include "vislib/graphics/gl/GLSLShader.h"
#include "vislib/graphics/gl/GLSLGeometryShader.h"
#include "mmcore/CoreInstance.h"

#include <memory>
#include <vector>

#include "glowl/Texture2D.h"

namespace megamol {
	namespace ngmesh {

		typedef vislib::graphics::gl::GLSLGeometryShader Shader;

		class GPUMaterialCollecton
		{
		public:
			struct Material
			{
				std::shared_ptr<Shader> shader_program;
				std::vector<std::shared_ptr<Texture2D>> textures;
			};

			void addMaterial(megamol::core::CoreInstance* mm_core_inst, std::string shader_btf_name);

			void addMaterial(std::shared_ptr<Shader> const& shader);

			void clearMaterials();

			inline std::vector<Material> const& getMaterials();

		private:
			std::vector<Material> m_materials;
		};

		inline std::vector<megamol::ngmesh::GPUMaterialCollecton::Material> const & megamol::ngmesh::GPUMaterialCollecton::getMaterials()
		{
			return m_materials;
		}
	}
}

#endif // !GPU_MATERIAL_DATA_STORAGE_H_INCLUDED
