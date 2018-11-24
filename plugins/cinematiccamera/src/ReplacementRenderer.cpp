/*
 * ReplacementRenderer.cpp
 *
 * Copyright (C) 2017 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"

#include "ReplacementRenderer.h"

#include "mmcore/param/BoolParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/param/IntParam.h"
#include "mmcore/param/ButtonParam.h"

#include "vislib/graphics/gl/IncludeAllGL.h"
#include "vislib/sys/Log.h"


using namespace megamol;
using namespace megamol::core;
using namespace megamol::cinematiccamera;
using namespace vislib;

/*
 * ReplacementRenderer::ReplacementRenderer (CTOR)
 */
ReplacementRenderer::ReplacementRenderer(void) : Renderer3DModule(),
    rendererCallerSlot("renderer", "outgoing renderer"),
    replacementRenderingParam(      "01_replacementRendering", "Show/hide replacement rendering for the model."),
    toggleReplacementRenderingParam("02_toggleReplacement", "Toggle replacement rendering."),
    replacementKeyParam(            "03_replacmentKeyAssign", "Assign a key to replacement rendering button."),
    alphaParam(                     "04_alpha", "The alpha value of the replacement rendering."),

    bbox(),
    toggleReplacementRendering(false)
{

    this->rendererCallerSlot.SetCompatibleCall<view::CallRender3DDescription>();
    this->MakeSlotAvailable(&this->rendererCallerSlot);

    this->replacementRenderingParam.SetParameter(new param::BoolParam(this->toggleReplacementRendering));
    this->MakeSlotAvailable(&this->replacementRenderingParam);

    alphaParam.SetParameter(new param::FloatParam(0.75f, 0.0f, 1.0f));
    this->MakeSlotAvailable(&alphaParam);

    param::EnumParam *tmpEnum = new param::EnumParam(static_cast<int>(keyAssignment::KEY_ASSIGN_NONE));
    tmpEnum->SetTypePair(keyAssignment::KEY_ASSIGN_NONE, "Choose key assignment for button.");
    tmpEnum->SetTypePair(keyAssignment::KEY_ASSIGN_O, "o");
    tmpEnum->SetTypePair(keyAssignment::KEY_ASSIGN_I, "i");
    tmpEnum->SetTypePair(keyAssignment::KEY_ASSIGN_J, "j");
    tmpEnum->SetTypePair(keyAssignment::KEY_ASSIGN_K, "k");
    tmpEnum->SetTypePair(keyAssignment::KEY_ASSIGN_X, "x");
    tmpEnum->SetTypePair(keyAssignment::KEY_ASSIGN_Y, "y");
    this->replacementKeyParam << tmpEnum;
    this->MakeSlotAvailable(&this->replacementKeyParam);
}

/*
 * ReplacementRenderer::~ReplacementRenderer (DTOR)
 */
ReplacementRenderer::~ReplacementRenderer(void) {
    this->Release();
}

/*
 * ReplacementRenderer::release
 */
void ReplacementRenderer::release(void) {

}

/*
 * ReplacementRenderer::create
 */
bool ReplacementRenderer::create(void) {
   
    return true;
}

/*
 * ReplacementRenderer::GetExtents
 */
bool ReplacementRenderer::GetExtents(megamol::core::view::CallRender3D& call) {

    view::CallRender3D *cr3d_in = dynamic_cast<view::CallRender3D*>(&call);
    if (cr3d_in == nullptr) return false;

    // Propagate changes made in GetExtents() from outgoing CallRender3D (cr3d_out) to incoming  CallRender3D (cr3d_in).
    view::CallRender3D *cr3d_out = this->rendererCallerSlot.CallAs<view::CallRender3D>();

    if ((cr3d_out != nullptr) && (*cr3d_out)(core::view::AbstractCallRender::FnGetExtents)) {
        unsigned int timeFramesCount = cr3d_out->TimeFramesCount();
        cr3d_in->SetTimeFramesCount((timeFramesCount > 0) ? (timeFramesCount) : (1));
        cr3d_in->SetTime(cr3d_out->Time());
        cr3d_in->AccessBoundingBoxes() = cr3d_out->AccessBoundingBoxes();

        this->bbox = cr3d_out->AccessBoundingBoxes().WorldSpaceBBox();
    }
    else {
        cr3d_in->SetTimeFramesCount(1);
        cr3d_in->SetTime(1.0f);

        this->bbox = vislib::math::Cuboid<float>(-1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f);
        cr3d_in->AccessBoundingBoxes().Clear();
        cr3d_in->AccessBoundingBoxes().SetWorldSpaceBBox(this->bbox);
    }

    return true;
}


/*
 * ReplacementRenderer::Render
 */
bool ReplacementRenderer::Render(megamol::core::view::CallRender3D& call) {

    view::CallRender3D *cr3d_in = dynamic_cast<view::CallRender3D*>(&call);
    if (cr3d_in == nullptr)  return false;

    // Update parameters
    if (this->replacementRenderingParam.IsDirty()) {
        this->replacementRenderingParam.ResetDirty();

        this->toggleReplacementRendering = this->replacementRenderingParam.Param<param::BoolParam>()->Value();
    }
    if (this->toggleReplacementRenderingParam.IsDirty()) {
        this->toggleReplacementRenderingParam.ResetDirty();

        this->toggleReplacementRendering = !this->toggleReplacementRendering;
        this->replacementRenderingParam.Param<param::BoolParam>()->SetValue(this->toggleReplacementRendering, false);
    }

    // This can only be done once ...
    if (this->replacementKeyParam.IsDirty()) {
        this->replacementKeyParam.ResetDirty();

        keyAssignment newKey = static_cast<keyAssignment>(this->replacementKeyParam.Param<param::EnumParam>()->Value());
        WORD newKeyWord = 0;
        switch (newKey) {
            case(keyAssignment::KEY_ASSIGN_O): newKeyWord = 'o'; break;
            case(keyAssignment::KEY_ASSIGN_I): newKeyWord = 'i'; break;
            case(keyAssignment::KEY_ASSIGN_J): newKeyWord = 'j'; break;
            case(keyAssignment::KEY_ASSIGN_K): newKeyWord = 'k'; break;
            case(keyAssignment::KEY_ASSIGN_X): newKeyWord = 'x'; break;
            case(keyAssignment::KEY_ASSIGN_Y): newKeyWord = 'y'; break;
            default: break;
        }

        // Make button param available ... 
        this->toggleReplacementRenderingParam.SetParameter(new param::ButtonParam(newKeyWord));
        this->MakeSlotAvailable(&this->toggleReplacementRenderingParam);
        // ... and set enum param unavailable.
        this->SetSlotUnavailable(static_cast<AbstractSlot*>(&this->replacementKeyParam));
    }

    // Render ...
    if (this->toggleReplacementRendering) {

        // Set opengl states
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
        glDisable(GL_LIGHTING);
        glEnable(GL_DEPTH_TEST);
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

        // Draw bounding box
        glEnable(GL_CULL_FACE);
        glCullFace(GL_FRONT);
        this->drawBoundingBox();
        glCullFace(GL_BACK);
        this->drawBoundingBox();
        glDisable(GL_CULL_FACE);

        // Reset opengl states
        glDisable(GL_BLEND);
    }
    else {

        // Call render function of slave renderer
        view::CallRender3D *cr3d_out = this->rendererCallerSlot.CallAs<view::CallRender3D>();
        if (cr3d_out != nullptr) {
            *cr3d_out = *cr3d_in;
            (*cr3d_out)(core::view::AbstractCallRender::FnRender);
        }
    }

    return true;
}


/*
* ReplacementRenderer::drawBoundingBox
*/
void ReplacementRenderer::drawBoundingBox() {

    float alpha = alphaParam.Param<param::FloatParam>()->Value();

    glBegin(GL_QUADS);

    glEdgeFlag(true);

    //glColor4f(0.5f, 0.5f, 0.5f, alpha);
    glColor4f(0.0f, 0.0f, 0.25f, alpha);
    glVertex3f(this->bbox.Left(), this->bbox.Bottom(), this->bbox.Back());
    glVertex3f(this->bbox.Left(), this->bbox.Top(), this->bbox.Back());
    glVertex3f(this->bbox.Right(), this->bbox.Top(), this->bbox.Back());
    glVertex3f(this->bbox.Right(), this->bbox.Bottom(), this->bbox.Back());

    //glColor4f(0.5f, 0.5f, 0.5f, alpha);
    glColor4f(0.0f, 0.0f, 0.75f, alpha);
    glVertex3f(this->bbox.Left(), this->bbox.Bottom(), this->bbox.Front());
    glVertex3f(this->bbox.Right(), this->bbox.Bottom(), this->bbox.Front());
    glVertex3f(this->bbox.Right(), this->bbox.Top(), this->bbox.Front());
    glVertex3f(this->bbox.Left(), this->bbox.Top(), this->bbox.Front());

    //glColor4f(0.75f, 0.75f, 0.75f, alpha);
    glColor4f(0.0f, 0.75f, 0.0f, alpha);
    glVertex3f(this->bbox.Left(), this->bbox.Top(), this->bbox.Back());
    glVertex3f(this->bbox.Left(), this->bbox.Top(), this->bbox.Front());
    glVertex3f(this->bbox.Right(), this->bbox.Top(), this->bbox.Front());
    glVertex3f(this->bbox.Right(), this->bbox.Top(), this->bbox.Back());

    //glColor4f(0.75f, 0.75f, 0.75f, alpha);
    glColor4f(0.0f, 0.25f, 0.0f, alpha);
    glVertex3f(this->bbox.Left(), this->bbox.Bottom(), this->bbox.Back());
    glVertex3f(this->bbox.Right(), this->bbox.Bottom(), this->bbox.Back());
    glVertex3f(this->bbox.Right(), this->bbox.Bottom(), this->bbox.Front());
    glVertex3f(this->bbox.Left(), this->bbox.Bottom(), this->bbox.Front());

    //glColor4f(0.25f, 0.25f, 0.25f, alpha);
    glColor4f(0.25f, 0.0f, 0.0f, alpha);
    glVertex3f(this->bbox.Left(), this->bbox.Bottom(), this->bbox.Back());
    glVertex3f(this->bbox.Left(), this->bbox.Bottom(), this->bbox.Front());
    glVertex3f(this->bbox.Left(), this->bbox.Top(), this->bbox.Front());
    glVertex3f(this->bbox.Left(), this->bbox.Top(), this->bbox.Back());

    //glColor4f(0.25f, 0.25f, 0.25f, alpha);
    glColor4f(0.75f, 0.0f, 0.0f, alpha);
    glVertex3f(this->bbox.Right(), this->bbox.Bottom(), this->bbox.Back());
    glVertex3f(this->bbox.Right(), this->bbox.Top(), this->bbox.Back());
    glVertex3f(this->bbox.Right(), this->bbox.Top(), this->bbox.Front());
    glVertex3f(this->bbox.Right(), this->bbox.Bottom(), this->bbox.Front());

    glEnd();

}