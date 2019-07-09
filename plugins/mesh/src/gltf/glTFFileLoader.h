/*
* glTFFileLoader.h
*
* Copyright (C) 2019 by Universitaet Stuttgart (VISUS).
* All rights reserved.
*/

#ifndef GLTF_FILE_LOADER_H_INCLUDED
#define GLTF_FILE_LOADER_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "mmcore/CalleeSlot.h"
#include "mmcore/param/ParamSlot.h"
#include "mesh/mesh.h"
#include "mesh/CallGltfData.h"

namespace megamol
{
	namespace mesh
	{

		class MESH_API GlTFFileLoader : public core::Module
		{
		public:
			/**
			* Answer the name of this module.
			*
			* @return The name of this module.
			*/
			static const char *ClassName(void) {
				return "GlTFFileLoader";
			}

			/**
			* Answer a human readable description of this module.
			*
			* @return A human readable description of this module.
			*/
			static const char *Description(void) {
				return "Data source for simply loading a glTF file from disk";
			}

			/**
			* Answers whether this module is available on the current system.
			*
			* @return 'true' if the module is available, 'false' otherwise.
			*/
			static bool IsAvailable(void) {
				return true;
			}

			GlTFFileLoader();
			~GlTFFileLoader();

		protected:
			/**
			* Implementation of 'Create'.
			*
			* @return 'true' on success, 'false' otherwise.
			*/
			bool create(void);

			/**
			* Gets the data from the source.
			*
			* @param caller The calling call.
			*
			* @return 'true' on success, 'false' on failure.
			*/
			bool getDataCallback(core::Call& caller);

			/**
			* Implementation of 'Release'.
			*/
			void release();

		private:
			std::shared_ptr<tinygltf::Model> m_gltf_model;

			int                              m_update_flag;

			/** The gltf file name */
			core::param::ParamSlot           m_glTFFilename_slot;

			/** The slot for requesting data */
			megamol::core::CalleeSlot        m_getData_slot;
		};

	}
}


#endif // !GLTF_FILE_LOADER_H_INCLUDED
