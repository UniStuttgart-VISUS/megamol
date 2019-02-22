/*
 * View2D.cpp
 *
 * Copyright (C) 2009 - 2010 by VISUS (Universitaet Stuttgart). 
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "vislib/graphics/gl/IncludeAllGL.h"
#include "mmcore/view/View2D.h"
#include "mmcore/CoreInstance.h"
#include "mmcore/view/CallRenderView.h"
#include "mmcore/view/CallRender2D.h"
#include "mmcore/param/BoolParam.h"
#include "mmcore/param/ButtonParam.h"
#include "mmcore/param/ColorParam.h"
#include "mmcore/param/StringParam.h"
#include "mmcore/utility/ColourParser.h"
#include "vislib/Trace.h"
#include "vislib/math/Matrix4.h"

using namespace megamol::core;


/*
 * view::View2D::View2D
 */
view::View2D::View2D(void) : view::AbstractRenderingView(),
        firstImg(false), height(1.0f),
        mouseMode(MouseMode::Propagate), mouseX(0.0f), mouseY(0.0f),
        rendererSlot("rendering", "Connects the view to a Renderer"),
        resetViewSlot("resetView", "Triggers the reset of the view"),
        showBBoxSlot("showBBox", "Shows/hides the bounding box"), 
		bboxCol{1.0f, 1.0f, 1.0f, 0.625f},
		bboxColSlot("bboxCol", "Sets the colour for the bounding box"),
        resetViewOnBBoxChangeSlot("resetViewOnBBoxChange", "whether to reset the view when the bounding boxes change"),
        viewX(0.0f), viewY(0.0f), viewZoom(1.0f), viewUpdateCnt(0),
        width(1.0f), incomingCall(NULL), overrideViewTile(NULL), timeCtrl() {

    this->rendererSlot.SetCompatibleCall<CallRender2DDescription>();
    this->MakeSlotAvailable(&this->rendererSlot);

    this->resetViewSlot << new param::ButtonParam(core::view::Key::KEY_HOME);
    this->resetViewSlot.SetUpdateCallback(&View2D::onResetView);
    this->MakeSlotAvailable(&this->resetViewSlot);

    this->showBBoxSlot << new param::BoolParam(true);
    this->MakeSlotAvailable(&this->showBBoxSlot);

    this->bboxColSlot << new param::ColorParam(this->bboxCol[0], this->bboxCol[1], this->bboxCol[2], this->bboxCol[3]);
    this->MakeSlotAvailable(&this->bboxColSlot);

    this->resetViewOnBBoxChangeSlot << new param::BoolParam(false);
    this->MakeSlotAvailable(&this->resetViewOnBBoxChangeSlot);

    for (unsigned int i = 0; this->timeCtrl.GetSlot(i) != NULL; i++) {
        this->MakeSlotAvailable(this->timeCtrl.GetSlot(i));
    }

    this->ResetView();
}


/*
 * view::View2D::~View2D
 */
view::View2D::~View2D(void) {
    this->Release();
    this->overrideViewTile = NULL;
}


/*
 * view::View2D::GetCameraSyncNumber
 */
unsigned int view::View2D::GetCameraSyncNumber(void) const {
    return this->viewUpdateCnt;
}


/*
 * view::View2D::SerialiseCamera
 */
void view::View2D::SerialiseCamera(vislib::Serialiser& serialiser) const {
    serialiser.Serialise(this->viewX, "viewX");
    serialiser.Serialise(this->viewY, "viewY");
    serialiser.Serialise(this->viewZoom, "viewZ");
}


/*
 * view::View2D::DeserialiseCamera
 */
void view::View2D::DeserialiseCamera(vislib::Serialiser& serialiser) {
    serialiser.Deserialise(this->viewX, "viewX");
    serialiser.Deserialise(this->viewY, "viewY");
    serialiser.Deserialise(this->viewZoom, "viewZ");
}


/*
 * view::View2D::Render
 */
void view::View2D::Render(const mmcRenderViewContext& context) {
    float time = static_cast<float>(context.Time);
    double instTime = context.InstanceTime;

    if (this->doHookCode()) {
        this->doBeforeRenderHook();
    }

    CallRender2D *cr2d = this->rendererSlot.CallAs<CallRender2D>();

    AbstractRenderingView::beginFrame();

    // clear viewport
    int vpx = 0, vpy = 0;
    float w = this->width;
    float h = this->height;
    if (this->overrideViewport != NULL) {
        if ((this->overrideViewport[0] >= 0) && (this->overrideViewport[1] >= 0)
            && (this->overrideViewport[2] > 0) && (this->overrideViewport[3] > 0)) {
            ::glViewport(
                vpx = this->overrideViewport[0], vpy = this->overrideViewport[1],
                this->overrideViewport[2], this->overrideViewport[3]);
            w = static_cast<float>(this->overrideViewport[2]);
            h = static_cast<float>(this->overrideViewport[3]);
        }
    } else {
        ::glViewport(0, 0,
            static_cast<GLsizei>(this->width),
            static_cast<GLsizei>(this->height));
    }

    const float *bkgndCol = (this->overrideBkgndCol != NULL)
        ? this->overrideBkgndCol : this->BkgndColour();
    ::glClearColor(bkgndCol[0], bkgndCol[1], bkgndCol[2], 0.0f);

    if (cr2d == NULL) {
        this->renderTitle(0.0f, 0.0f, this->width, this->height,
            this->width, this->height, false, false, instTime);
        AbstractRenderingView::endFrame(true);
        return;
    } else {
        this->removeTitleRenderer();
    }
    if (this->firstImg) {
        this->firstImg = false;
        this->ResetView();
    }

    if ((*cr2d)(AbstractCallRender::FnGetExtents)) {
        if (this->bbox != cr2d->GetBoundingBox()
            && resetViewOnBBoxChangeSlot.Param<param::BoolParam>()->Value()) {
            this->ResetView();
        }
        bbox = cr2d->GetBoundingBox();
        this->timeCtrl.SetTimeExtend(cr2d->TimeFramesCount(), cr2d->IsInSituTime());
        if (time > static_cast<float>(cr2d->TimeFramesCount())) {
            time = static_cast<float>(cr2d->TimeFramesCount());
        }
    }

    cr2d->SetTime(time);
    cr2d->SetInstanceTime(instTime);
    cr2d->SetGpuAffinity(context.GpuAffinity);
    cr2d->SetLastFrameTime(AbstractRenderingView::lastFrameTime());

    ::glMatrixMode(GL_PROJECTION);
    ::glLoadIdentity();
    float asp = h / w;
    if (this->overrideViewTile != NULL) {
        asp = this->overrideViewTile[3]
            / this->overrideViewTile[2];
    }
    //::glScalef(asp, 1.0f, 1.0f);
    //float aMatrix[16];
    vislib::math::Matrix<float, 4, vislib::math::MatrixLayout::COLUMN_MAJOR> m;
    //glGetFloatv(GL_PROJECTION_MATRIX, aMatrix);

    m.SetIdentity();
    m.SetAt(0,0,asp);
    glLoadMatrixf(m.PeekComponents());

    float vx = this->viewX;
    float vy = this->viewY;
    float vz = this->viewZoom;
    if (this->overrideViewTile != NULL) {
        float xo = (this->overrideViewTile[0] + 0.5f * this->overrideViewTile[2] - 0.5f * this->overrideViewTile[4])
            / this->overrideViewTile[4];
        float yo = (this->overrideViewTile[1] + 0.5f * this->overrideViewTile[3] - 0.5f * this->overrideViewTile[5])
            / this->overrideViewTile[5];
        float zf = this->overrideViewTile[5]
            / this->overrideViewTile[3];
        vx -= (xo / (0.5f * vz)) * this->overrideViewTile[4] / this->overrideViewTile[5];
        vy += yo / (0.5f * vz);
        vz *= zf;
    }

    ::glMatrixMode(GL_MODELVIEW);
    ::glLoadIdentity();
    m.SetIdentity();
    m.SetAt(0, 0, vz);
    m.SetAt(1, 1, vz);
    m.SetAt(0, 3, vx * vz);
    m.SetAt(1, 3, vy * vz);
    //::glScalef(vz, vz, 1.0f);
    //::glTranslatef(vx, vy, 0.0f);
    //glGetFloatv(GL_MODELVIEW_MATRIX, aMatrix);
    glLoadMatrixf(m.PeekComponents());

    cr2d->SetBackgroundColour(
        static_cast<unsigned char>(bkgndCol[0] * 255.0f),
        static_cast<unsigned char>(bkgndCol[1] * 255.0f),
        static_cast<unsigned char>(bkgndCol[2] * 255.0f));

    asp = 1.0f / asp;
    vislib::math::Rectangle<float> vr(
        (-asp / vz - vx),
        (-1.0f / vz - vy),
        (asp / vz - vx),
        (1.0f / vz - vy));
    cr2d->SetBoundingBox(vr);

    if (this->incomingCall == NULL) {
        cr2d->SetOutputBuffer(GL_BACK,
            vpx, vpy, static_cast<int>(w), static_cast<int>(h));
    } else {
        cr2d->SetOutputBuffer(*this->incomingCall);
    }
    ::glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // depth could be required even for 2d

	if (this->bboxColSlot.IsDirty()) {
        this->bboxColSlot.Param<param::ColorParam>()->Value(this->bboxCol[0], this->bboxCol[1], this->bboxCol[2], this->bboxCol[3]);
        this->bboxColSlot.ResetDirty();
    }

    if (this->showBBoxSlot.Param<param::BoolParam>()->Value()) {
        ::glEnable(GL_BLEND);
        ::glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        ::glEnable(GL_LINE_SMOOTH);
        ::glLineWidth(1.2f);
        ::glDisable(GL_LIGHTING);

        ::glColor4fv(this->bboxCol);
        ::glBegin(GL_LINE_LOOP);
        ::glVertex2f(bbox.Left(), bbox.Top());
        ::glVertex2f(bbox.Left(), bbox.Bottom());
        ::glVertex2f(bbox.Right(), bbox.Bottom());
        ::glVertex2f(bbox.Right(), bbox.Top());
        ::glEnd();

        ::glDisable(GL_LINE_SMOOTH);
        ::glDisable(GL_BLEND);
        ::glLineWidth(1.0f);
    }

    (*cr2d)(AbstractCallRender::FnRender);

    if (this->showSoftCursor()) {
        ::glMatrixMode(GL_PROJECTION);
        ::glLoadIdentity();
        ::glTranslatef(-1.0f, 1.0f, 0.0f);
        ::glScalef(2.0f / this->width, -2.0f / this->height, 1.0f);
        ::glMatrixMode(GL_MODELVIEW);
        ::glLoadIdentity();
        ::glBegin(GL_LINES);
        ::glColor4ub(255, 255, 0, 255);
        ::glVertex2f(0.0f, 0.0f);
        ::glVertex2f(this->mouseX, this->mouseY);
        ::glEnd();
    }

    AbstractRenderingView::endFrame();

}


/*
 * view::View2D::ResetView
 */
void view::View2D::ResetView(void) {
    // using namespace vislib::graphics;
    VLTRACE(VISLIB_TRCELVL_INFO, "View2D::ResetView\n");

    CallRender2D *cr2d = this->rendererSlot.CallAs<CallRender2D>();
    if ((cr2d != NULL) && ((*cr2d)(AbstractCallRender::FnGetExtents))) {
        this->viewX = -0.5f * (cr2d->GetBoundingBox().Left() + cr2d->GetBoundingBox().Right());
        this->viewY = -0.5f * (cr2d->GetBoundingBox().Bottom() + cr2d->GetBoundingBox().Top());
        if ((this->width / this->height) > static_cast<float>(cr2d->GetBoundingBox().AspectRatio())) {
            this->viewZoom = 2.0f / cr2d->GetBoundingBox().Height();
        } else {
            this->viewZoom = (2.0f * this->width) / (this->height * cr2d->GetBoundingBox().Width());
        }
        this->viewZoom *= 0.99f;

    } else {
        this->viewX = 0.0f;
        this->viewY = 0.0f;
        this->viewZoom = 1.0f;
    }
    this->viewUpdateCnt++;
}


/*
 * view::View2D::Resize
 */
void view::View2D::Resize(unsigned int width, unsigned int height) {
    this->width = static_cast<float>(width);
    this->height = static_cast<float>(height);
    // intentionally empty ATM
}


/*
 * view::View2D::OnRenderView
 */
bool view::View2D::OnRenderView(Call& call) {
    float overBC[3];
    int overVP[4] = {0, 0, 0, 0};
    float overTile[6];
    view::CallRenderView *crv = dynamic_cast<view::CallRenderView *>(&call);
    if (crv == NULL) return false;

    this->incomingCall = crv;
    this->overrideViewport = overVP; // never set window viewport
    if (crv->IsViewportSet()) {
        overVP[2] = crv->ViewportWidth();
        overVP[3] = crv->ViewportHeight();
    }
    if (crv->IsTileSet()) {
        overTile[0] = crv->TileX();
        overTile[1] = crv->TileY();
        overTile[2] = crv->TileWidth();
        overTile[3] = crv->TileHeight();
        overTile[4] = crv->VirtualWidth();
        overTile[5] = crv->VirtualHeight();
        this->overrideViewTile = overTile;
    }
    if (crv->IsBackgroundSet()) {
        overBC[0] = static_cast<float>(crv->BackgroundRed()) / 255.0f;
        overBC[1] = static_cast<float>(crv->BackgroundGreen()) / 255.0f;
        overBC[2] = static_cast<float>(crv->BackgroundBlue()) / 255.0f;
        this->overrideBkgndCol = overBC; // hurk
    }

    float time = crv->Time();
    if (time < 0.0f) time = this->DefaultTime(crv->InstanceTime());
    mmcRenderViewContext context;
    ::ZeroMemory(&context, sizeof(context));
    context.Time = time;
    context.InstanceTime = crv->InstanceTime();
    // TODO: Affinity
    this->Render(context);

    this->overrideBkgndCol = NULL;
    this->overrideViewport = NULL;
    this->overrideViewTile = NULL;
    this->incomingCall = NULL;

    return true;
}


/*
 * view::View2D::UpdateFreeze
 */
void view::View2D::UpdateFreeze(bool freeze) {
    // currently not supported
}


bool view::View2D::OnKey(Key key, KeyAction action, Modifiers mods) {
    auto* cr = this->rendererSlot.CallAs<view::CallRender2D>();
    if (cr == NULL) return false;

    InputEvent evt;
    evt.tag = InputEvent::Tag::Key;
    evt.keyData.key = key;
    evt.keyData.action = action;
    evt.keyData.mods = mods;
    cr->SetInputEvent(evt);
    if (!(*cr)(view::CallRender2D::FnOnKey)) return false;

    return true;
}


bool view::View2D::OnChar(unsigned int codePoint) {
    auto* cr = this->rendererSlot.CallAs<view::CallRender2D>();
    if (cr == NULL) return false;

    InputEvent evt;
    evt.tag = InputEvent::Tag::Char;
    evt.charData.codePoint = codePoint;
    cr->SetInputEvent(evt);
    if (!(*cr)(view::CallRender2D::FnOnChar)) return false;

    return true;
}


bool view::View2D::OnMouseButton(MouseButton button, MouseButtonAction action, Modifiers mods) {
	this->mouseMode = MouseMode::Propagate;

    auto* cr = this->rendererSlot.CallAs<view::CallRender2D>();
    if (cr) {
        InputEvent evt;
        evt.tag = InputEvent::Tag::MouseButton;
        evt.mouseButtonData.button = button;
        evt.mouseButtonData.action = action;
        evt.mouseButtonData.mods = mods;
        cr->SetInputEvent(evt);
        if ((*cr)(view::CallRender2D::FnOnMouseButton)) return true;
    }

    auto down = action == MouseButtonAction::PRESS;
    if (button == MouseButton::BUTTON_LEFT && down) {
        this->mouseMode = MouseMode::Pan;
    } else if (button == MouseButton::BUTTON_MIDDLE && down) {
        this->mouseMode = MouseMode::Zoom;
    }

    return true;
}


bool view::View2D::OnMouseMove(double x, double y) {
    if (this->mouseMode == MouseMode::Propagate) {
        float mx, my;
        mx = ((x * 2.0f / this->width) - 1.0f) * this->width / this->height;
        my = 1.0f - (y * 2.0f / this->height);
        mx /= this->viewZoom;
        my /= this->viewZoom;
        mx -= this->viewX;
        my -= this->viewY;

        auto* cr = this->rendererSlot.CallAs<view::CallRender2D>();
        if (cr) {
            InputEvent evt;
            evt.tag = InputEvent::Tag::MouseMove;
            evt.mouseMoveData.x = mx;
            evt.mouseMoveData.y = my;
            cr->SetInputEvent(evt);
            if ((*cr)(view::CallRender2D::FnOnMouseMove)) return true;
        }
    } else if (this->mouseMode == MouseMode::Pan) {
        float movSpeed = 2.0f / (this->viewZoom * this->height);
        this->viewX -= (this->mouseX - x) * movSpeed;
        this->viewY += (this->mouseY - y) * movSpeed;
        if (((this->mouseX - x) > 0.0f) || ((this->mouseY - y) > 0.0f)) {
            this->viewUpdateCnt++;
        }
    } else if (this->mouseMode == MouseMode::Zoom) {
        const double spd = 2.0;
        const double logSpd = log(spd);
        float base = 1.0f;

        CallRender2D* cr2d = this->rendererSlot.CallAs<CallRender2D>();
        if ((cr2d != NULL) && ((*cr2d)(AbstractCallRender::FnGetExtents))) {
            base = cr2d->GetBoundingBox().Height();
        }

        float newZoom =
            static_cast<float>(pow(spd, log(static_cast<double>(this->viewZoom / base)) / logSpd +
                                            static_cast<double>(((this->mouseY - y) * 1.0f / this->height)))) *
            base;

        if (!vislib::math::IsEqual(newZoom, this->viewZoom)) {
            this->viewUpdateCnt++;
        }
        this->viewZoom = newZoom;
    }

    this->mouseX = x;
    this->mouseY = y;

    return true;
}


bool view::View2D::OnMouseScroll(double dx, double dy) {
    auto* cr = this->rendererSlot.CallAs<view::CallRender2D>();
    if (cr == NULL) return false;

    InputEvent evt;
    evt.tag = InputEvent::Tag::MouseScroll;
    evt.mouseScrollData.dx = dx;
    evt.mouseScrollData.dy = dy;
    cr->SetInputEvent(evt);
    if (!(*cr)(view::CallRender2D::FnOnMouseScroll)) return false;

    return true;
}


/*
 * view::View2D::unpackMouseCoordinates
 */
void view::View2D::unpackMouseCoordinates(float &x, float &y) {
    x *= this->width;
    y *= this->height;
    y -= 1.0f;
}


/*
 * view::View2D::create
 */
bool view::View2D::create(void) {
 
    this->firstImg = true;

    return true;
}


/*
 * view::View2D::release
 */
void view::View2D::release(void) {
    this->removeTitleRenderer();
    // intentionally empty
}


/*
 * view::View2D::onResetView
 */
bool view::View2D::onResetView(param::ParamSlot& p) {
    this->ResetView();
    return true;
}
