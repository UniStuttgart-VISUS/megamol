/*
 * mmvtkmDataRenderer.cpp
 *
 * Copyright (C) 2018 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "mmvtkm/mmvtkmRenderer.h"
#include "mmvtkm/mmvtkmDataCall.h"
#include "vtkm/cont/Timer.h"
#include "vtkm/io/reader/VTKDataSetReader.h"

//#define VTKM_DEVICE_ADAPTER VTKM_DEVICE_ADAPTER_CUDA
//#include "vtkm/cont/DeviceAdapter.h"


using namespace megamol;
using namespace megamol::mmvtkm;

mmvtkmDataRenderer::mmvtkmDataRenderer(void)
    : Renderer3DModule()
    , vtkmDataCallerSlot("getData", "Connects the vtkm renderer with a vtkm data source")
    , oldHash(0)
    ,
    // actor(vtkm::cont::DynamicCellSet(), vtkm::cont::CoordinateSystem(), vtkm::cont::Field()),
    scene()
    , mapper()
    , canvas()
    , camera()
    , colorArray(nullptr)
//, view(scene, mapper, canvas, camera)
{
    this->vtkmDataCallerSlot.SetCompatibleCall<megamol::mmvtkm::mmvtkmDataCallDescription>();
    this->MakeSlotAvailable(&this->vtkmDataCallerSlot);
}

mmvtkmDataRenderer::~mmvtkmDataRenderer(void) { this->Release(); }

bool mmvtkmDataRenderer::create() {

    return true;
}

void mmvtkmDataRenderer::release() {
    // Renderer3DModule::release();
}

bool mmvtkmDataRenderer::GetCapabilities(core::Call& call) { return true; }

bool mmvtkmDataRenderer::GetMetaData(core::Call& call) {
    mmvtkmDataCall* cd = this->vtkmDataCallerSlot.CallAs<mmvtkmDataCall>();
    if (cd == NULL) return false;

    if (!(*cd)(1)) {
        return false;
    } else {
        // cd->FrameID();
    }

    std::cout << "Das ist ein Checksatz: Renderer::GetMetaData0" << '\n';
    core::view::CallRender3D* cr = dynamic_cast<core::view::CallRender3D*>(&call);
    if (cr == NULL) return false;
    std::cout << "Das ist ein Checksatz: Renderer::GetMetaData1" << '\n';
    canvasWidth = cr->GetViewport().Width();
    canvasHeight = cr->GetViewport().Height();

    return true;
}

bool mmvtkmDataRenderer::GetData(core::Call& call) {
    std::cout << "Das ist ein Checksatz: Renderer::GetData0" << '\n';
    mmvtkmDataCall* cd = this->vtkmDataCallerSlot.CallAs<mmvtkmDataCall>();


    if (cd == NULL) {
        std::cout << "Das ist ein Checksatz: Renderer::GetDataNULL" << '\n';
        return false;
    }

    // temp
    this->GetMetaData(call);

    std::cout << "Das ist ein Checksatz: Renderer::GetData1" << '\n';
    if (!(*cd)(0)) {
        std::cout << "Das ist ein Checksatz: Renderer::GetData2" << '\n';
        return false;
    } else {
        if (cd->DataHash() == this->oldHash) {
            std::cout << "Das ist ein Checksatz: Renderer::GetData3" << '\n';
            return false;
        } else {
            // call metadata here to update metadata if new data is available

            std::cout << "Das ist ein Checksatz: Renderer::GetData4" << '\n';
            this->oldHash = cd->DataHash();
            vtkmDataSet = cd->GetDataSet();

            // TODO place all static setups in create
            // compute the bounds and extends of the input data
            vtkm::cont::ColorTable colorTable("inferno");
            vtkm::rendering::Actor actor(vtkmDataSet->GetCellSet(), vtkmDataSet->GetCoordinateSystem(),
                vtkmDataSet->GetPointField("pointvar"),
                colorTable); // depending on dataset change to getCellField with according FrameID
            scene = vtkm::rendering::Scene();
            scene.AddActor(actor); // makeScene(...)
            canvas = vtkm::rendering::CanvasRayTracer(canvasWidth, canvasHeight);
        }
    }

    return true;
}

bool mmvtkmDataRenderer::Render(core::Call& call) {
    // connect camera to plugin to get camera movement
    std::cout << "Das ist ein Checksatz: Renderer::Render" << '\n';

    core::view::CallRender3D* cr = dynamic_cast<core::view::CallRender3D*>(&call);
    if (cr == NULL) return false;

    this->GetData(call);

    vtkm::cont::Timer time;

    // get all necessary camera parameters from render call
    vislib::graphics::SceneSpacePoint3D lookatCr = cr->GetCameraParameters()->LookAt();
    vislib::graphics::SceneSpaceVector3D upCr = cr->GetCameraParameters()->Up();
    vislib::graphics::SceneSpacePoint3D positionCr = cr->GetCameraParameters()->Position();

    // transform parameters to vtkm compatibel structures
    vtkm::Vec<vtkm::Float64, 3> lookat(lookatCr.GetX(), lookatCr.GetY(), lookatCr.GetZ());
    vtkm::Vec<vtkm::Float32, 3> up(upCr.GetX(), upCr.GetY(), upCr.GetZ());
    vtkm::Vec<vtkm::Float32, 3> position(positionCr.GetX(), positionCr.GetY(), positionCr.GetZ());
    vtkm::Float32 nearPlane = cr->GetCameraParameters()->NearClip();
    vtkm::Float32 farPlane = cr->GetCameraParameters()->FarClip();
    vtkm::Float32 fovY = cr->GetCameraParameters()->ApertureAngle();
    vtkm::Bounds coordsBounds = vtkmDataSet->GetCoordinateSystem().GetBounds();
    vtkm::Vec<vtkm::Float64, 3> totalExtent(coordsBounds.X.Length(), coordsBounds.Y.Length(), coordsBounds.Z.Length());
    vtkm::Float64 mag = vtkm::Magnitude(totalExtent);
    vtkm::Normalize(totalExtent);

    // setup a camera and point it to towards the center of the input data
    camera.ResetToBounds(coordsBounds);
    camera.SetLookAt(lookat /*totalExtent * (mag * .5f)*/);
    camera.SetViewUp(up);
    camera.SetClippingRange(nearPlane, farPlane);
    camera.SetFieldOfView(fovY);
    camera.SetPosition(position);

    // set up all vtkm rendering components needed in order to render with vtkm
    // default coordinatesystem name = "coordinates"
    // default fieldname = "pointvar"
    // default cellname = "cells"

    // THIS SECTION SHOULD POSSIBLY BE IN ::Render BUT ONLY .Initialize and .Paint when data has been changed!
    // POSSIBLY DECLARE LOCAL DIRTY FLAG
    // update actor, acutally just the field, each frame
    // store dynamiccellset and coordinatesystem after reading them in GetMetaData
    vtkm::rendering::View3D view(scene, mapper, canvas, camera);
    view.Initialize(); // required

    time.Start();
    view.Paint();
    // view.SaveAs("demo_output.ppm");
    time.Stop();
    std::cout << "Elapsed time 3: " << time.GetElapsedTime() << '\n';
    int numActors = scene.GetNumberOfActors();
    std::cout << "Num Actors: " << numActors << '\n';
    // the canvas holds the buffer of the offscreen rendered image
    vtkm::cont::ArrayHandle<vtkm::Vec<vtkm::Float32, 4>> colorBuffer = view.GetCanvas().GetColorBuffer();

    // pulling out the c array from the buffer
    // which can just be rendered
    colorArray = colorBuffer.GetStorage().GetBasePointer();

    // Write the C array to an OpenGL buffer
    glDrawPixels((GLint)canvasWidth, (GLint)canvasHeight, GL_RGBA, GL_FLOAT, colorArray);

    // SwapBuffers();

    return true;
}
