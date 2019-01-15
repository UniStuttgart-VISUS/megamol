/*
* MaterialDataStorage.h
*
* Copyright (C) 2019 by Universitaet Stuttgart (VISUS).
* All rights reserved.
*/

#ifndef MATERIAL_DATA_STORAGE_H_INCLUDED
#define MATERIAL_DATA_STORAGE_H_INLCUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include <list>
#include <vector>

#include "NGMeshStructs.h"

namespace megamol {
	namespace ngmesh {

		class MaterialDataStorage
		{
		public:

			template<typename T>
			using IteratorPair = std::pair< T, T>;

			template<typename TextureDataIterator>
			void addMaterial(
				std::string const&                                    btf_filename,
				std::vector<TextureLayout> const&                     texture_layouts,
				std::vector<IteratorPair<TextureDataIterator>> const& texture_data);

		private:
			struct TextureData
			{
				std::vector<std::byte> texture_data;
				TextureLayout          texture_descriptor;
			};

			struct Material
			{
				std::string              btf_filename;
				std::vector<TextureData> textures;
			};

			std::list<Material> m_materials;
		};


		template<typename TextureDataIterator>
		inline void MaterialDataStorage::addMaterial(
			std::string const&                                    btf_filename,
			std::vector<TextureLayout> const&                     texture_layouts,
			std::vector<IteratorPair<TextureDataIterator>> const& texture_data)
		{
		}
	}
}

#endif // !MATERIAL_DATA_STORAGE_H_INCLUDED
