/*
* GPURenderTaskDataStorage.h
*
* Copyright (C) 2019 by Universitaet Stuttgart (VISUS).
* All rights reserved.
*/

#ifndef GPU_MATERIAL_DATA_STORAGE_H_INCLUDED
#define GPU_MATERIAL_DATA_STORAGE_H_INCLUDED

#include "vislib/graphics/gl/GLSLShader.h"

#include <memory>
#include <vector>

#include "glowl/Texture2D.h"

namespace megamol {
	namespace ngmesh {

		typedef vislib::graphics::gl::GLSLShader GLSLShader;

		class GPUMaterialDataStorage
		{
		public:
			struct Material
			{
				std::shared_ptr<GLSLShader>             shader_program;
				std::vector<std::shared_ptr<Texture2D>> textures;
			};

			void addMaterial(std::string shader_btf_name);

			void addMaterial(std::shared_ptr<GLSLShader> const& shader);

		private:
			std::vector<Material> m_materials;
		};
	}
}

#endif // !GPU_MATERIAL_DATA_STORAGE_H_INCLUDED
