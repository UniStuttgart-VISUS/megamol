/*
* mmvtkmRenderer.h
*
* Copyright (C) 2018 by VISUS (Universitaet Stuttgart)
* Alle Rechte vorbehalten.
*/

#ifndef MEGAMOL_MMVTKM_MMVTKMRENDERER_H_INCLUDED
#define MEGAMOL_MMVTKM_MMVTKMRENDERER_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

//#include "vtkm/cont/DataSet.h"
#include "vtkm/rendering/Actor.h"
#include "vtkm/rendering/Scene.h"
//#include "vtkm/rendering/Canvas.h"
#include "vtkm/rendering/CanvasRayTracer.h"
//#include "vtkm/rendering/Mapper.h"
#include "vtkm/rendering/MapperRayTracer.h"
#include "vtkm/rendering/MapperVolume.h"
#include "vtkm/rendering/View3D.h"

#include "mmcore/view/Renderer3DModule_2.h"
#include "mmcore/CallerSlot.h"


namespace megamol {
namespace mmvtkm {

	/**
	* Renderer for vtkm data
	*/
	class mmvtkmDataRenderer : public core::view::Renderer3DModule_2 {
	public:

		/**
		* Answer the name of this module.
		*
		* @return The name of this module.
		*/
		static const char *ClassName(void) {
			return "vtkmDataRenderer";
		}

		/**
		* Answer a human readable description of this module.
		*
		* @return A human readable description of this module.
		*/
		static const char *Description(void) {
			return "Renderer for vtkm data.";
		}

		/**
		* Answers whether this module is available on the current system.
		*
		* @return 'true' if the module is available, 'false' otherwise.
		*/
		static bool IsAvailable(void) {
			return true;
		}


		/** Ctor. */
		mmvtkmDataRenderer(void);

		/** Dtor. */
		virtual ~mmvtkmDataRenderer(void);

	protected:

		/**
		* Implementation of 'Create'.
		*
		* @return 'true' on success, 'false' otherwise.
		*/
		virtual bool create(void);

		/**
		* Implementation of 'Release'.
		*/
		virtual void release(void);

		/**
		* The get capabilities callback. The module should set the members
		* of 'call' to tell the caller its capabilities.
		*
		* @param call The calling call.
		*
		* @return The return value of the function.
		*/
        virtual bool GetCapabilities(core::view::CallRender3D_2& call);

		/**
		* The get extents callback. The module should set the members of
		* 'call' to tell the caller the extents of its data (bounding boxes
		* and times).
		*
		* @param call The calling call.
		*
		* @return The return value of the function.
		*/
        virtual bool GetExtents(core::view::CallRender3D_2& call);

		/**
		* The render callback.
		*
		* @param call The calling call.
		*
		* @return The return value of the function.
		*/
        virtual bool Render(core::view::CallRender3D_2& call);

		/**
         * The render callback.
         *
         * @param call The calling call.
         *
         * @return The return value of the function.
         */
        virtual bool GetData(core::view::CallRender3D_2& call);


	private:
       /** caller slot */
    core::CallerSlot vtkmDataCallerSlot;

	/** Some test vtkm data set */
    vtkm::cont::DataSet *vtkmDataSet;

	/** The vtkm structures used for rendering */
    vtkm::rendering::Scene scene;
    vtkm::rendering::MapperRayTracer mapper;
    vtkm::rendering::CanvasRayTracer canvas;
    vtkm::rendering::Camera vtkmCamera;
    //vtkm::rendering::View3D view;
    void* colorArray;
    bool dataHasChanged;
	float canvasWidth, canvasHeight, canvasDepth;
    SIZE_T oldHash;
	};

} /* end namespace mmvtkm */
} /* end namespace megamol */

#endif // MEGAMOL_MMVTKM_MMVTKMRENDERER_H_INCLUDED