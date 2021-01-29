/*
 * AbstractRenderingViewGL.cpp
 *
 * Copyright (C) 2010 by VISUS (Universitaet Stuttgart). 
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "vislib/graphics/gl/IncludeAllGL.h"
#include "mmcore/view/AbstractRenderingViewGL.h"
#include "mmcore/AbstractNamedObject.h"
#include "vislib/String.h"
#include "mmcore/param/BoolParam.h"
#include "mmcore/param/ColorParam.h"
#include "mmcore/param/StringParam.h"
#include "mmcore/utility/ColourParser.h"
#include "vislib/sys/AutoLock.h"
#include "vislib/sys/sysfunctions.h"
#include "mmcore/utility/sys/Thread.h"

using namespace megamol::core;


/*
 * view::AbstractRenderingViewGL::AbstractTitleRenderer::AbstractTitleRenderer
 */
view::AbstractRenderingViewGL::AbstractTitleRenderer::AbstractTitleRenderer(void) {
    // intentionally empty
}


/*
 * view::AbstractRenderingViewGL::AbstractTitleRenderer::~AbstractTitleRenderer
 */
view::AbstractRenderingViewGL::AbstractTitleRenderer::~AbstractTitleRenderer(void) {
    // intentionally empty
}


/*
 * view::AbstractRenderingViewGL::EmptyTitleRenderer::EmptyTitleRenderer
 */
view::AbstractRenderingViewGL::EmptyTitleRenderer::EmptyTitleRenderer(void) {
    // intentionally empty
}


/*
 * view::AbstractRenderingViewGL::EmptyTitleRenderer::~EmptyTitleRenderer
 */
view::AbstractRenderingViewGL::EmptyTitleRenderer::~EmptyTitleRenderer(void) {
    // intentionally empty
}


/*
 * view::AbstractRenderingViewGL::EmptyTitleRenderer::Create
 */
bool view::AbstractRenderingViewGL::EmptyTitleRenderer::Create(void) {
    // intentionally empty
    return true;
}


/*
 * view::AbstractRenderingViewGL::EmptyTitleRenderer::Render
 */
void view::AbstractRenderingViewGL::EmptyTitleRenderer::Render(
        float tileX, float tileY, float tileW, float tileH,
        float virtW, float virtH, bool stereo, bool leftEye, double instTime,
        class ::megamol::core::CoreInstance *core) {
    ::glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    ::glClear(GL_COLOR_BUFFER_BIT);
}


/*
 * view::AbstractRenderingViewGL::EmptyTitleRenderer::Release
 */
void view::AbstractRenderingViewGL::EmptyTitleRenderer::Release(void){
    // intentionally empty
}


/*
 * view::AbstractRenderingViewGL::AbstractRenderingViewGL
 */
view::AbstractRenderingViewGL::AbstractRenderingViewGL(void) : AbstractViewGL(),
        overrideBkgndCol(NULL), overrideViewport(NULL),
        bkgndColSlot("backCol", "The views background colour"),
        softCursor(false), softCursorSlot("softCursor", "Bool flag to activate software cursor rendering"),
        titleRenderer(NULL), fpsCounter(10), fpsThreadID(0), fpsOutputTimer(0) {

    this->bkgndCol[0] = 0.0f;
    this->bkgndCol[1] = 0.0f;
    this->bkgndCol[2] = 0.125f;

    this->bkgndColSlot << new param::ColorParam(this->bkgndCol[0], this->bkgndCol[1], this->bkgndCol[2], 1.0f);
    this->MakeSlotAvailable(&this->bkgndColSlot);

    this->softCursorSlot << new param::BoolParam(this->softCursor);
    this->MakeSlotAvailable(&this->softCursorSlot);

}


/*
 * view::AbstractRenderingViewGL::~AbstractRenderingViewGL
 */
view::AbstractRenderingViewGL::~AbstractRenderingViewGL(void) {
    this->removeTitleRenderer();
    this->overrideBkgndCol = NULL; // DO NOT DELETE
    this->overrideViewport = NULL; // DO NOT DELETE
}


/*
* view::AbstractRenderingViewGL::bkgndColour
*/
const float *view::AbstractRenderingViewGL::BkgndColour(void) const {
    if (this->bkgndColSlot.IsDirty()) {
        this->bkgndColSlot.ResetDirty();
        this->bkgndColSlot.Param<param::ColorParam>()->Value(this->bkgndCol[0], this->bkgndCol[1], this->bkgndCol[2]);
    }
    return this->bkgndCol;
}


/*
 * view::AbstractRenderingViewGL::beginFrame
 */
void view::AbstractRenderingViewGL::beginFrame(void) {
    vislib::sys::AutoLock(this->fpsLock);

    // The first thread that ever draws with this view will count the FPS
    if (this->fpsThreadID == 0) {
        this->fpsThreadID = vislib::sys::Thread::CurrentID();
    }

    if (this->fpsThreadID == vislib::sys::Thread::CurrentID()) {
        this->fpsCounter.FrameBegin();
    }

}


/*
 * view::AbstractRenderingViewGL::endFrame
 */
void view::AbstractRenderingViewGL::endFrame(bool abort) {
    vislib::sys::AutoLock(this->fpsLock);

    if (!abort) {
        unsigned int ticks = vislib::sys::GetTicksOfDay();
        if ((ticks < this->fpsOutputTimer) || (ticks >= this->fpsOutputTimer + 1000)) {
            this->fpsOutputTimer = ticks;
            //vislib::StringA name("UNKNOWN");
            //AbstractNamedObject *ano = dynamic_cast<AbstractNamedObject*>(this);
            //if (ano != NULL) {
            //    name = ano->FullName();
            //}
            // okey, does not make any sense when multiple windows are rendering, but better than nothing
            // printf("%s FPS: %f\n", name.PeekBuffer(), this->fpsCounter.FPS()); //Stop spamming the console
            // fflush(stdout); // grr
        }
    }

    if (this->fpsThreadID == vislib::sys::Thread::CurrentID()) {
        this->fpsCounter.FrameEnd();
    }

}


/*
 * view::AbstractRenderingViewGL::lastFrameTime
 */
double view::AbstractRenderingViewGL::lastFrameTime(void) const {
    return this->fpsCounter.LastFrameTime();
}


/*
 * view::AbstractRenderingViewGL::showSoftCursor
 */
bool view::AbstractRenderingViewGL::showSoftCursor(void) const {
    if (this->softCursorSlot.IsDirty()) {
        this->softCursorSlot.ResetDirty();
        this->softCursor = this->softCursorSlot.Param<param::BoolParam>()->Value();
    }
    return this->softCursor;
}


/*
 * view::AbstractRenderingViewGL::renderTitle
 */
void view::AbstractRenderingViewGL::renderTitle(
        float tileX, float tileY, float tileW, float tileH,
        float virtW, float virtH, bool stereo, bool leftEye, double instTime) const {
    //if (!this->titleRenderer) {
    //    this->titleRenderer = new special::TitleRenderer();
    //    if (!this->titleRenderer->Create()) {
    //        delete this->titleRenderer;
    //        this->titleRenderer = new EmptyTitleRenderer();
    //        ASSERT(this->titleRenderer->Create());
    //    }
    //}

    //this->titleRenderer->Render(tileX, tileY, tileW, tileH,
    //    virtW, virtH, stereo, leftEye, instTime, this->GetCoreInstance());

}


/*
 * view::AbstractRenderingViewGL::removeTitleRenderer
 */
void view::AbstractRenderingViewGL::removeTitleRenderer(void) const {
    if (this->titleRenderer) {
        this->titleRenderer->Release();
        delete this->titleRenderer;
        this->titleRenderer = NULL;
    }
}


/*
 * view::AbstractRenderingViewGL::toggleSoftCurse
 */
void view::AbstractRenderingViewGL::toggleSoftCurse(void) {
    this->softCursorSlot.Param<param::BoolParam>()->SetValue(!this->softCursor);
}
