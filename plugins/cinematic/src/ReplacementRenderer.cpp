/*
 * ReplacementRenderer.cpp
 *
 * Copyright (C) 2017 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "ReplacementRenderer.h"


using namespace megamol;
using namespace megamol::core;
using namespace megamol::cinematic;

using namespace vislib;


ReplacementRenderer::ReplacementRenderer(void) : megamol::core::view::RendererModule<megamol::core::view::CallRender3D_2>(),
    alphaParam("alpha", "The alpha value of the replacement rendering."),
    replacementRenderingParam("replacement", "Show/hide replacement rendering for chained renderer."),
    replacementKeyParam("hotkeyAssignment", "Choose hotkey for replacement rendering button."),
    toggleReplacementParam("toggleReplacement", "Toggle replacement rendering."),
    toggle(false),
    utils(),
    bbox()
{
    // Make render slots available
    this->MakeSlotAvailable(&this->chainRenderSlot);
    this->MakeSlotAvailable(&this->renderSlot);

    // Init parameters
    alphaParam.SetParameter(new param::FloatParam(0.75f, 0.0f, 1.0f));
    this->MakeSlotAvailable(&alphaParam);

    this->toggleReplacementParam.SetParameter(new param::ButtonParam());
    this->MakeSlotAvailable(&this->toggleReplacementParam);

    this->replacementRenderingParam.SetParameter(new param::BoolParam(this->toggle));
    this->MakeSlotAvailable(&this->replacementRenderingParam);

    param::EnumParam *tmpEnum = new param::EnumParam(static_cast<int>(KeyAssignment::KEY_ASSIGN_NONE));
    tmpEnum->SetTypePair(KeyAssignment::KEY_ASSIGN_NONE, "Choose key assignment for button.");
    tmpEnum->SetTypePair(KeyAssignment::KEY_ASSIGN_1, "Alt + 1");
    tmpEnum->SetTypePair(KeyAssignment::KEY_ASSIGN_2, "Alt + 2");
    tmpEnum->SetTypePair(KeyAssignment::KEY_ASSIGN_3, "Alt + 3");
    tmpEnum->SetTypePair(KeyAssignment::KEY_ASSIGN_4, "Alt + 4");
    tmpEnum->SetTypePair(KeyAssignment::KEY_ASSIGN_5, "Alt + 5");
    tmpEnum->SetTypePair(KeyAssignment::KEY_ASSIGN_6, "Alt + 6");
    tmpEnum->SetTypePair(KeyAssignment::KEY_ASSIGN_7, "Alt + 7");
    tmpEnum->SetTypePair(KeyAssignment::KEY_ASSIGN_8, "Alt + 8");
    tmpEnum->SetTypePair(KeyAssignment::KEY_ASSIGN_9, "Alt + 9");
    tmpEnum->SetTypePair(KeyAssignment::KEY_ASSIGN_0, "Alt + 0");
    this->replacementKeyParam << tmpEnum;
    this->MakeSlotAvailable(&this->replacementKeyParam);
}


ReplacementRenderer::~ReplacementRenderer(void) {

    this->Release();
}


void ReplacementRenderer::release(void) {

}


bool ReplacementRenderer::create(void) {

    // Initialise render utils
    if (!this->utils.Initialise(this->GetCoreInstance())) {
        vislib::sys::Log::DefaultLog.WriteError("[TIMELINE RENDERER] [create] Couldn't initialize the font.");
        return false;
    }

    return true;
}


bool ReplacementRenderer::GetExtents(megamol::core::view::CallRender3D_2& call) {

    auto cr3d_out = this->chainRenderSlot.CallAs<view::CallRender3D_2>();

    bool retVal = true;
    if (cr3d_out != nullptr) {
        *cr3d_out = call;
        retVal = (*cr3d_out)(view::AbstractCallRender::FnGetExtents);
        call = *cr3d_out;
        this->bbox = call.AccessBoundingBoxes().BoundingBox();
    }
    else {
        call.SetTimeFramesCount(1);
        call.SetTime(0.0f);
        this->bbox = vislib::math::Cuboid<float>(-1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f);
        call.AccessBoundingBoxes().SetBoundingBox(this->bbox);
    }

    return retVal;
}


bool ReplacementRenderer::Render(megamol::core::view::CallRender3D_2& call) {

    auto leftSlotParent = call.PeekCallerSlot()->Parent();
    std::shared_ptr<const view::AbstractView> viewptr =
        std::dynamic_pointer_cast<const view::AbstractView>(leftSlotParent);
    if (viewptr != nullptr) { // TODO move this behind the fbo magic?
        auto vp = call.GetViewport();
        glViewport(vp.Left(), vp.Bottom(), vp.Width(), vp.Height());
        auto backCol = call.BackgroundColor();
        glClearColor(backCol.x, backCol.y, backCol.z, 0.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    }

    if (this->replacementRenderingParam.IsDirty()) {
        this->replacementRenderingParam.ResetDirty();
        this->toggle = this->replacementRenderingParam.Param<param::BoolParam>()->Value();
    }

    if (this->toggleReplacementParam.IsDirty()) {
        this->toggleReplacementParam.ResetDirty();
        this->toggle = !this->toggle;
        this->replacementRenderingParam.Param<param::BoolParam>()->SetValue(this->toggle, false);
    }

    if (this->replacementKeyParam.IsDirty()) {
        this->replacementKeyParam.ResetDirty();

        KeyAssignment newKey = static_cast<KeyAssignment>(this->replacementKeyParam.Param<param::EnumParam>()->Value());
        core::view::Key key = core::view::Key::KEY_UNKNOWN;
        switch (newKey) {
            case(KeyAssignment::KEY_ASSIGN_1): key = core::view::Key::KEY_1; break;
            case(KeyAssignment::KEY_ASSIGN_2): key = core::view::Key::KEY_2; break;
            case(KeyAssignment::KEY_ASSIGN_3): key = core::view::Key::KEY_3; break;
            case(KeyAssignment::KEY_ASSIGN_4): key = core::view::Key::KEY_4; break;
            case(KeyAssignment::KEY_ASSIGN_5): key = core::view::Key::KEY_5; break;
            case(KeyAssignment::KEY_ASSIGN_6): key = core::view::Key::KEY_6; break;
            case(KeyAssignment::KEY_ASSIGN_7): key = core::view::Key::KEY_7; break;
            case(KeyAssignment::KEY_ASSIGN_8): key = core::view::Key::KEY_8; break;
            case(KeyAssignment::KEY_ASSIGN_9): key = core::view::Key::KEY_9; break;
            case(KeyAssignment::KEY_ASSIGN_0): key = core::view::Key::KEY_0; break;
            default: break;
        }
        // Set hotkey for button param 
        if (key != core::view::Key::KEY_UNKNOWN) {
            this->toggleReplacementParam.Param<param::ButtonParam>()->SetKey(key);
            this->toggleReplacementParam.Param<param::ButtonParam>()->SetModifier(core::view::Modifier::ALT);
        }
        else {
            this->toggleReplacementParam.Param<param::ButtonParam>()->SetKey(core::view::Key::KEY_UNKNOWN);
            this->toggleReplacementParam.Param<param::ButtonParam>()->SetModifier(core::view::Modifier::NONE);
        }
    }

    if (this->toggle) {

        view::Camera_2 cam;
        call.GetCamera(cam);
        cam_type::snapshot_type snapshot;
        cam_type::matrix_type viewTemp, projTemp;
        cam.calc_matrices(snapshot, viewTemp, projTemp);
        glm::mat4 proj = projTemp;
        glm::mat4 view = viewTemp;
        glm::mat4 mvp = proj * view;

        auto viewport = call.GetViewport();
        float vp_fw = static_cast<float>(viewport.Width());
        float vp_fh = static_cast<float>(viewport.Height());

        float alpha = alphaParam.Param<param::FloatParam>()->Value();

        glm::vec4 dark_blue = { 0.0f, 0.0f, 0.25f, alpha };
        glm::vec4 light_blue = { 0.0f, 0.0f, 0.75f, alpha };
        glm::vec4 dark_red = { 0.25f, 0.0f, 0.0f, alpha };
        glm::vec4 light_red = { 0.75f, 0.0f, 0.0f, alpha };
        glm::vec4 dark_green = { 0.0f, 0.25f, 0.0f, alpha };
        glm::vec4 light_green = { 0.0f, 0.75f, 0.0f, alpha };

        glm::vec3 left_top_back = { this->bbox.Left(), this->bbox.Top(), this->bbox.Back() };
        glm::vec3 left_bottom_back = { this->bbox.Left(), this->bbox.Bottom(), this->bbox.Back() };
        glm::vec3 right_top_back = { this->bbox.Right(), this->bbox.Top(), this->bbox.Back() };
        glm::vec3 right_bottom_back = { this->bbox.Right(), this->bbox.Bottom(), this->bbox.Back() };
        glm::vec3 left_top_front = { this->bbox.Left(), this->bbox.Top(), this->bbox.Front() };
        glm::vec3 left_bottom_front = { this->bbox.Left(), this->bbox.Bottom(), this->bbox.Front() };
        glm::vec3 right_top_front = { this->bbox.Right(), this->bbox.Top(), this->bbox.Front() };
        glm::vec3 right_bottom_front = { this->bbox.Right(), this->bbox.Bottom(), this->bbox.Front() };

        this->utils.PushQuadPrimitive(left_bottom_back, left_top_back, right_top_back, right_bottom_back, dark_blue);
        this->utils.PushQuadPrimitive(left_bottom_front, right_bottom_front, right_top_front, left_top_front, light_blue);
        this->utils.PushQuadPrimitive(left_top_back, left_top_front, right_top_front, right_top_back, dark_green);
        this->utils.PushQuadPrimitive(left_bottom_back, right_bottom_back, right_bottom_front, left_bottom_front, light_green);
        this->utils.PushQuadPrimitive(left_bottom_back, left_bottom_front, left_top_front, left_top_back, dark_red);
        this->utils.PushQuadPrimitive(right_bottom_back, right_top_back, right_top_front, right_bottom_front, light_red);

        // Render bounding box as replacement
        this->utils.DrawQuadPrimitives(mvp, glm::vec2(vp_fw, vp_fh));
    }
    else {

        auto cr3d_out = this->chainRenderSlot.CallAs<view::CallRender3D_2>();
        if (cr3d_out != nullptr) {
            *cr3d_out = call;
            return (*cr3d_out)(core::view::AbstractCallRender::FnRender);
        }
    }

    return true;
}
