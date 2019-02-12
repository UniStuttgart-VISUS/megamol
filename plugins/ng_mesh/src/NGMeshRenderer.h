/*
* NGMeshRenderer.h
*
* Copyright (C) 2017 by Universitaet Stuttgart (VISUS).
* All rights reserved.
*/

#ifndef NG_MESH_RENDERER_H_INCLUDED
#define NG_MESH_RENDERER_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "vislib/graphics/gl/GLSLShader.h"
#include "vislib/math/Matrix.h"

#include "mmcore/CallerSlot.h"
#include "mmcore/view/Renderer3DModule.h"
#include "mmcore/view/CallRender3D.h"

//#include "ng_mesh/CallNGMeshRenderBatches.h"
#include "ng_mesh/BatchedMeshesDataCall.h"
#include "ng_mesh/MaterialDataStorage.h"
#include "ng_mesh/NGMeshStructs.h"
#include "ng_mesh/RenderTaskDataStorage.h"
#include "ng_mesh/GPUMaterialDataStorage.h"

#include "glowl/BufferObject.h"
#include "glowl/Mesh.h"

namespace megamol {
namespace ngmesh {


	/**
	 * Renderer module for rendering geometry with modern (OpenGL 4.3+) features.
	 * Objects for rendering are supplied in batches. Each  render batch can contain
	 * many objects that use the same shader program and also share the same geometry
	 * or at least the same vertex format.
	 * Per render batch, a single call of glMultiDrawElementsIndirect is made. The data
	 * for the indirect draw call is stored and accessed via SSBOs.
	 */
	class NGMeshRenderer : public megamol::core::view::Renderer3DModule
	{
	public:
		/**
		* Answer the name of this module.
		*
		* @return The name of this module.
		*/
		static const char *ClassName(void) {
			return "NGMeshRenderer";
		}

		/**
		* Answer a human readable description of this module.
		*
		* @return A human readable description of this module.
		*/
		static const char *Description(void) {
			return "Modern renderer for meshes. Objects are rendered in batches using indirect draw calls.";
		}

		/**
		* Answers whether this module is available on the current system.
		*
		* @return 'true' if the module is available, 'false' otherwise.
		*/
		static bool IsAvailable(void) {
#ifdef _WIN32
#if defined(DEBUG) || defined(_DEBUG)
			HDC dc = ::wglGetCurrentDC();
			HGLRC rc = ::wglGetCurrentContext();
			ASSERT(dc != NULL);
			ASSERT(rc != NULL);
#endif // DEBUG || _DEBUG
#endif // _WIN32
			return vislib::graphics::gl::GLSLShader::AreExtensionsAvailable()
				&& isExtAvailable("GL_ARB_shader_draw_parameters")
				&& ogl_IsVersionGEQ(4, 3);
		}

		/** Ctor. */
		NGMeshRenderer();

		/** Dtor. */
		~NGMeshRenderer();

	protected:

		/**
		* Implementation of 'Create'.
		*
		* @return 'true' on success, 'false' otherwise.
		*/
		bool create();

		/**
		* Implementation of 'Release'.
		*/
		void release();

		/**
		* The get extents callback. The module should set the members of
		* 'call' to tell the caller the extents of its data (bounding boxes
		* and times).
		*
		* @param call The calling call.
		*
		* @return The return value of the function.
		*/
		bool GetExtents(core::Call& call);

		//	/**
		//	 * Adds a new render batch by translating the given render batch data
		//	 * into a GPU RenderBatch ready for rendering.
		//	 *
		//	 * @param shader_prgm_data The data for loading a shader program, i.e. the shader filename
		//	 * @param mesh_data The data required for creating a mesh, i.e. vertex and index data, and vertex layout
		//	 * @param draw_command_data The data that describes the draw calls executed by glMultiDrawElementsIndirect
		//	 * @param obj_shader_params Additional data used by the shader during rendering that is specific to the rendered object
		//	 * @param mtl_shader_params Additional used by the shader during rendering that is specific to the material in use, i.e. texture handles
		//	 */
		//	void addRenderBatch(
		//		ShaderPrgmDataAccessor const&		shader_prgm_data,
		//		MeshDataAccessor const&				mesh_data,
		//		DrawCommandDataAccessor const&		draw_command_data,
		//		ObjectShaderParamsDataAccessor const&	obj_shader_params,
		//		MaterialShaderParamsDataAccessor const&	mtl_shader_params);
		//	
		//	/**
		//	* Selectively updates an existing render batch with new data.
		//	*
		//	* @param idx The index of the render batch that is updated
		//	* @param shader_prgm_data The data for loading a shader program, i.e. the shader filename
		//	* @param mesh_data The data required for creating a mesh, i.e. vertex and index data, and vertex layout
		//	* @param draw_command_data The data that describes the draw calls executed by glMultiDrawElementsIndirect
		//	* @param obj_shader_params Additional data used by the shader during rendering that is specific to the rendered object
		//	* @param mtl_shader_params Additional used by the shader during rendering that is specific to the material in use, i.e. texture handles
		//	* @param update_flags The bit flags that signal which parts of the render batch data needs to be updated
		//	*/
		//	void updateRenderBatch(
		//		size_t						idx,
		//		ShaderPrgmDataAccessor const&		shader_prgm_data,
		//		MeshDataAccessor const&				mesh_data,
		//		DrawCommandDataAccessor const&		draw_command_data,
		//		ObjectShaderParamsDataAccessor const&	obj_shader_params,
		//		MaterialShaderParamsDataAccessor const&	mtl_shader_params,
		//		uint32_t					update_flags);

		void updateMeshes(BatchedMeshesDataAccessor const& meshes, uint32_t update_flags);

		void updateMaterials(std::shared_ptr<MaterialsDataStorage> const& materials, uint32_t update_flags);

		void updateRenderTasks(std::shared_ptr<RenderTaskDataStorage> const& render_tasks, uint32_t update_flags);
	
		/**
		* The render callback.
		*
		* @param call The calling call.
		*
		* @return The return value of the function.
		*/
		bool Render(core::Call& call);

	private:

		std::unique_ptr<BufferObject> m_per_frame_data; //< GPU buffer object that stores per frame data, i.e. camera parameters

		/** Render batches caller slot */
		//megamol::core::CallerSlot m_renderBatches_callerSlot;

		//megamol::core::CallerSlot m_mesh_callerSlot;
		//megamol::core::CallerSlot m_material_callerSlot;
		megamol::core::CallerSlot m_render_task_callerSlot;


		std::shared_ptr<Mesh> createMesh(BatchedMeshesDataAccessor const & meshes, size_t mesh_idx);
	};

}
}

#endif