/*
 * ReplacementRenderer.cpp
 *
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
using namespace megamol::core::view;
using namespace megamol::cinematiccamera;
using namespace vislib;

/*
 * ReplacementRenderer::ReplacementRenderer (CTOR)
 */
ReplacementRenderer::ReplacementRenderer(void) : Renderer3DModule(),
    slaveRendererSlot("renderer", "outgoing renderer"), 
    replacementRenderingParam(      "01_replacementRendering", "Show/hide replacement rendering for the model."),
    toggleReplacementRenderingParam("02_toggleReplacement", "Toggle replacement rendering."),
    replacementKeyParam(            "03_replacmentKeyAssign", "Assign a key to replacement rendering button."),
    alphaParam(                     "04_alpha", "The alpha value of the replacement rendering.")
    {

    this->alpha = 0.75f;
    this->toggleReplacementRendering = false;
    this->bbox.SetNull();

    this->slaveRendererSlot.SetCompatibleCall<CallRender3DDescription>();
    this->MakeSlotAvailable(&this->slaveRendererSlot);

    this->replacementRenderingParam.SetParameter(new param::BoolParam(this->toggleReplacementRendering));
    this->MakeSlotAvailable(&this->replacementRenderingParam);

    //this->toggleReplacementRenderingParam.SetParameter(new param::ButtonParam('b'));
    //this->MakeSlotAvailable(&this->toggleReplacementRenderingParam);

    this->alphaParam.SetParameter(new param::FloatParam(this->alpha, 0.0f, 1.0f));
    this->MakeSlotAvailable(&this->alphaParam);

    param::EnumParam *tmpEnum = new param::EnumParam(static_cast<int>(keyAssignment::KEY_ASSIGN_NONE));
    tmpEnum->SetTypePair(keyAssignment::KEY_ASSIGN_NONE, "Choose key assignment for button.");
    tmpEnum->SetTypePair(keyAssignment::KEY_ASSIGN_O, "o");
    tmpEnum->SetTypePair(keyAssignment::KEY_ASSIGN_P, "p");
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
 * ReplacementRenderer::GetCapabilities
 */
bool ReplacementRenderer::GetCapabilities(Call& call) {

    view::CallRender3D *cr3d = dynamic_cast<view::CallRender3D*>(&call);
    if (cr3d == NULL) {
        vislib::sys::Log::DefaultLog.WriteError("[ReplacementRenderer] [GetCapabilities] Call is NULL.");
        return false;
    }

    CallRender3D *oc = this->slaveRendererSlot.CallAs<CallRender3D>();
    if (!(oc == NULL) || (!(*oc)(2))) {
        cr3d->AddCapability(oc->GetCapabilities());
    }
    cr3d->AddCapability(view::CallRender3D::CAP_RENDER);

    return true;
}

/*
 * ReplacementRenderer::GetExtents
 */
bool ReplacementRenderer::GetExtents(Call& call) {

    view::CallRender3D *cr3d = dynamic_cast<view::CallRender3D*>(&call);
    if (cr3d == NULL) {
        vislib::sys::Log::DefaultLog.WriteError("[ReplacementRenderer] [GetExtents] Call is NULL.");
        return false;
    }

    view::CallRender3D *oc = this->slaveRendererSlot.CallAs<CallRender3D>();
    if (oc == NULL) return false;

    // Get bounding box of renderer.
    if (!(*oc)(1)) return false;
    *cr3d = *oc;

    this->bbox = cr3d->AccessBoundingBoxes().WorldSpaceBBox();

    return true;
}


/*
 * ReplacementRenderer::Render
 */
bool ReplacementRenderer::Render(Call& call) {

    view::CallRender3D *cr3d = dynamic_cast<view::CallRender3D*>(&call);
    if (cr3d == NULL) {
        vislib::sys::Log::DefaultLog.WriteError("[ReplacementRenderer] [Render] CallRender3D is NULL.");
        return false;
    }   

    view::CallRender3D *oc = this->slaveRendererSlot.CallAs<CallRender3D>();
    if (oc == NULL) {
        vislib::sys::Log::DefaultLog.WriteError("[ReplacementRenderer] [Render] slaveRendererSlot-Call is NULL.");
        return false;
    }

    // Update parameters
    if (this->replacementRenderingParam.IsDirty()) {
        this->toggleReplacementRendering = this->replacementRenderingParam.Param<param::BoolParam>()->Value();
        this->replacementRenderingParam.ResetDirty();
    }
    if (this->toggleReplacementRenderingParam.IsDirty()) {
        this->toggleReplacementRendering = !this->toggleReplacementRendering;
        this->replacementRenderingParam.Param<param::BoolParam>()->SetValue(this->toggleReplacementRendering, false);
        this->toggleReplacementRenderingParam.ResetDirty();
    }

    // This can only be done once ...
    if (this->replacementKeyParam.IsDirty()) {
        keyAssignment newKey = static_cast<keyAssignment>(this->replacementKeyParam.Param<param::EnumParam>()->Value());
        WORD newKeyWord = 0;
        switch (newKey) {
            case(keyAssignment::KEY_ASSIGN_O): newKeyWord = 'o'; break;
            case(keyAssignment::KEY_ASSIGN_P): newKeyWord = 'p'; break;
            case(keyAssignment::KEY_ASSIGN_J): newKeyWord = 'j'; break;
            case(keyAssignment::KEY_ASSIGN_K): newKeyWord = 'k'; break;
            case(keyAssignment::KEY_ASSIGN_X): newKeyWord = 'x'; break;
            case(keyAssignment::KEY_ASSIGN_Y): newKeyWord = 'y'; break;
            default: break;
        }

        // Make button param available ...
        this->toggleReplacementRenderingParam.SetParameter(new param::ButtonParam(newKeyWord));
        this->MakeSlotAvailable(&this->toggleReplacementRenderingParam);
        // Remove Enum param ...
        this->replacementKeyParam.ResetDirty();
        this->replacementKeyParam.MakeUnavailable();
    }

    // Render ...
    if (this->toggleReplacementRendering) {

        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

        glDisable(GL_LIGHTING);
        glDisable(GL_TEXTURE_2D);
        glDisable(GL_TEXTURE_1D);
        glDisable(GL_LINE_SMOOTH);
        glDisable(GL_POLYGON_SMOOTH);

        glDepthFunc(GL_LEQUAL);
        glEnable(GL_DEPTH_TEST);

        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        glEnable(GL_BLEND);


        // Draw bounding box
        glEnable(GL_CULL_FACE);
        glCullFace(GL_FRONT);
        this->drawBoundingBox();
        glCullFace(GL_BACK);
        this->drawBoundingBox();

        glDisable(GL_CULL_FACE);
        glDisable(GL_BLEND);
    }
    else {
        *oc = *cr3d;
        // Call render function of slave renderer
        (*oc)(0);
    }

    return true;
}



/*
* ReplacementRenderer::drawBoundingBox
*/
void ReplacementRenderer::drawBoundingBox() {

    this->alpha = this->alphaParam.Param<param::FloatParam>()->Value();

    glBegin(GL_QUADS);

    glEdgeFlag(true);

    //glColor4f(0.5f, 0.5f, 0.5f, this->alpha);
    glColor4f(0.5f, 0.5f, 0.5f, this->alpha);
    glVertex3f(this->bbox.Left(), this->bbox.Bottom(), this->bbox.Back());
    glVertex3f(this->bbox.Left(), this->bbox.Top(), this->bbox.Back());
    glVertex3f(this->bbox.Right(), this->bbox.Top(), this->bbox.Back());
    glVertex3f(this->bbox.Right(), this->bbox.Bottom(), this->bbox.Back());

    //glColor4f(0.5f, 0.5f, 0.5f, this->alpha);
    glColor4f(0.0f, 0.0f, 0.75f, this->alpha);
    glVertex3f(this->bbox.Left(), this->bbox.Bottom(), this->bbox.Front());
    glVertex3f(this->bbox.Right(), this->bbox.Bottom(), this->bbox.Front());
    glVertex3f(this->bbox.Right(), this->bbox.Top(), this->bbox.Front());
    glVertex3f(this->bbox.Left(), this->bbox.Top(), this->bbox.Front());

    //glColor4f(0.75f, 0.75f, 0.75f, this->alpha);
    glColor4f(0.0f, 0.75f, 0.0f, this->alpha);
    glVertex3f(this->bbox.Left(), this->bbox.Top(), this->bbox.Back());
    glVertex3f(this->bbox.Left(), this->bbox.Top(), this->bbox.Front());
    glVertex3f(this->bbox.Right(), this->bbox.Top(), this->bbox.Front());
    glVertex3f(this->bbox.Right(), this->bbox.Top(), this->bbox.Back());

    //glColor4f(0.75f, 0.75f, 0.75f, this->alpha);
    glColor4f(0.75f, 0.75f, 0.75f, this->alpha);
    glVertex3f(this->bbox.Left(), this->bbox.Bottom(), this->bbox.Back());
    glVertex3f(this->bbox.Right(), this->bbox.Bottom(), this->bbox.Back());
    glVertex3f(this->bbox.Right(), this->bbox.Bottom(), this->bbox.Front());
    glVertex3f(this->bbox.Left(), this->bbox.Bottom(), this->bbox.Front());

    //glColor4f(0.25f, 0.25f, 0.25f, this->alpha);
    glColor4f(0.25f, 0.25f, 0.25f, this->alpha);
    glVertex3f(this->bbox.Left(), this->bbox.Bottom(), this->bbox.Back());
    glVertex3f(this->bbox.Left(), this->bbox.Bottom(), this->bbox.Front());
    glVertex3f(this->bbox.Left(), this->bbox.Top(), this->bbox.Front());
    glVertex3f(this->bbox.Left(), this->bbox.Top(), this->bbox.Back());

    //glColor4f(0.25f, 0.25f, 0.25f, this->alpha);
    glColor4f(0.75f, 0.0f, 0.0f, this->alpha);
    glVertex3f(this->bbox.Right(), this->bbox.Bottom(), this->bbox.Back());
    glVertex3f(this->bbox.Right(), this->bbox.Top(), this->bbox.Back());
    glVertex3f(this->bbox.Right(), this->bbox.Top(), this->bbox.Front());
    glVertex3f(this->bbox.Right(), this->bbox.Bottom(), this->bbox.Front());

    glEnd();

}