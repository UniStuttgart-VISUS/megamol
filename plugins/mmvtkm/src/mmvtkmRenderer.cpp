/*
 * mmvtkmDataRenderer.cpp
 *
 * Copyright (C) 2018 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "mmvtkm/mmvtkmRenderer.h"
#include "mmvtkm/mmvtkmDataCall.h"
#include "vtkm/io/reader/VTKDataSetReader.h"

//#define VTKM_DEVICE_ADAPTER VTKM_DEVICE_ADAPTER_CUDA
//#include "vtkm/cont/DeviceAdapter.h"


using namespace megamol;
using namespace megamol::mmvtkm;

mmvtkmDataRenderer::mmvtkmDataRenderer(void)
    : Renderer3DModule_2()
    , vtkmDataCallerSlot("getData", "Connects the vtkm renderer with a vtkm data source")
    , oldHash(0)
    , scene()
    , mapper()
    , canvas()
    , vtkmCamera()
    , colorArray(nullptr)
	, dataHasChanged(true)
{
    this->vtkmDataCallerSlot.SetCompatibleCall<megamol::mmvtkm::mmvtkmDataCallDescription>();
    this->MakeSlotAvailable(&this->vtkmDataCallerSlot);
}

mmvtkmDataRenderer::~mmvtkmDataRenderer(void) { 
	this->Release(); 
}

bool mmvtkmDataRenderer::create() {
    return true;
}

void mmvtkmDataRenderer::release() {
    // Renderer3DModule::release();
}

bool mmvtkmDataRenderer::GetCapabilities(core::view::CallRender3D_2& call) { return true; }

bool mmvtkmDataRenderer::GetExtents(core::view::CallRender3D_2& call) {
    mmvtkmDataCall* cd = this->vtkmDataCallerSlot.CallAs<mmvtkmDataCall>();
    if (cd == NULL) return false;

    if (!(*cd)(1)) {
        return false;
    }

    return true;
}

bool mmvtkmDataRenderer::GetData(core::view::CallRender3D_2& call) {
    mmvtkmDataCall* cd = this->vtkmDataCallerSlot.CallAs<mmvtkmDataCall>();

    if (cd == NULL) {
        return false;
    }

    this->GetExtents(call);

    if (!(*cd)(0)) {
        return false;
    } else {
        if (cd->DataHash() == this->oldHash) {
            return false;
        } else {
            this->oldHash = cd->DataHash();
            vtkmDataSet = cd->GetDataSet();
            dataHasChanged = true;

            // compute the bounds and extends of the input data
            vtkm::cont::ColorTable colorTable("inferno");
            vtkm::rendering::Actor actor(vtkmDataSet->GetCellSet(), vtkmDataSet->GetCoordinateSystem(),
                vtkmDataSet->GetPointField("pointvar"),
                colorTable); // depending on dataset change to getCellField with according FrameID
            scene = vtkm::rendering::Scene();
            scene.AddActor(actor); // makeScene(...)
        }
    }

    return true;
}

bool mmvtkmDataRenderer::Render(core::view::CallRender3D_2& call) {
    this->GetData(call);

	// camera setup
    core::view::Camera_2 cam;
    call.GetCamera(cam);
    cam_type::snapshot_type snapshot;
    cam_type::matrix_type viewTemp, projTemp;

    // Generate complete snapshot
    cam.calc_matrices(snapshot, viewTemp, projTemp, core::thecam::snapshot_content::all);
    glm::vec4 viewport = glm::vec4(0, 0, cam.resolution_gate().width(), cam.resolution_gate().height());
    if (viewport.z < 1.0f) viewport.z = 1.0f;
    if (viewport.w < 1.0f) viewport.w = 1.0f;
    float shaderPointSize = vislib::math::Max(viewport.z, viewport.w);
    viewport = glm::vec4(0, 0, 2.f / viewport.z, 2.f / viewport.w);

    glm::vec4 camView = snapshot.view_vector;
    glm::vec4 camRight = snapshot.right_vector;
    glm::vec4 camUp = snapshot.up_vector;
    glm::vec4 camPos = snapshot.position;

	// vislib::graphics::SceneSpaceCuboid bbox = call.AccessBoundingBoxes().BoundingBox();

	// set camera setting for vtkm
	canvasWidth = cam.resolution_gate().width();
    canvasHeight = cam.resolution_gate().height();
    canvas = vtkm::rendering::CanvasRayTracer(canvasWidth, canvasHeight);
    vtkm::Vec<vtkm::Float64, 3> lookat(camView.x + 1.f, camView.y + 1.f, camView.z + 1.f);
    vtkm::Vec<vtkm::Float32, 3> up(camUp.x, camUp.y, camUp.z);
    vtkm::Vec<vtkm::Float32, 3> position(camPos.x + 1.f, camPos.y + 1.f, camPos.z + 1.f);
    vtkm::Float32 nearPlane = snapshot.frustum_near;
    vtkm::Float32 farPlane = snapshot.frustum_far;
    vtkm::Float32 fovY = cam.aperture_angle();
    vtkm::Bounds coordsBounds = vtkmDataSet->GetCoordinateSystem().GetBounds();
    vtkm::Vec<vtkm::Float64, 3> totalExtent(coordsBounds.X.Length(), coordsBounds.Y.Length(), coordsBounds.Z.Length());
    vtkm::Float64 mag = vtkm::Magnitude(totalExtent);
    vtkm::Normalize(totalExtent);

    // setup a camera and point it to towards the center of the input data
    vtkmCamera.ResetToBounds(coordsBounds);
    vtkmCamera.SetLookAt(lookat /*totalExtent * (mag * .5f)*/);
    vtkmCamera.SetViewUp(up);
    vtkmCamera.SetClippingRange(nearPlane, farPlane);
    vtkmCamera.SetFieldOfView(fovY);
    vtkmCamera.SetPosition(position);

	//vislib::math::Cuboid<float> bbox(coordsBounds.X.Min, coordsBounds.Y.Min, coordsBounds.Z.Min, 
		//coordsBounds.X.Max, coordsBounds.Y.Max, coordsBounds.Z.Max);
    vislib::math::Cuboid<float> bbox(-1.f, -1.f, -1.f, 1.f, 1.f, 1.f);
	call.AccessBoundingBoxes().SetBoundingBox(bbox);
    call.AccessBoundingBoxes().SetClipBox(bbox);

    // default coordinatesystem name = "coordinates"
    // default fieldname = "pointvar"
    // default cellname = "cells"

    // update actor, acutally just the field, each frame
    // store dynamiccellset and coordinatesystem after reading them in GetExtents
    vtkm::rendering::View3D view(scene, mapper, canvas, vtkmCamera);
	view.Initialize(); // required

	view.Paint();
	// the canvas holds the buffer of the offscreen rendered image
	vtkm::cont::ArrayHandle<vtkm::Vec<vtkm::Float32, 4>> colorBuffer = view.GetCanvas().GetColorBuffer();

	// pulling out the c array from the buffer
	// which can just be rendered
	colorArray = colorBuffer.GetStorage().GetBasePointer();

	dataHasChanged = false;

    // Write the C array to an OpenGL buffer
    glDrawPixels((GLint)canvasWidth, (GLint)canvasHeight, GL_RGBA, GL_FLOAT, colorArray);

    // SwapBuffers();

    return true;
}
