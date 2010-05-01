/*
 * AbstractTileView.cpp
 *
 * Copyright (C) 2010 by VISUS (Universitaet Stuttgart). 
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#define _USE_MATH_DEFINES
#include "cluster/AbstractClusterView.h"
#include <cmath>
#include <GL/gl.h>
#include "vislib/mathfunctions.h"
#include "vislib/OutlineFont.h"
#include "vislib/sysfunctions.h"
#include "vislib/SystemInformation.h"
#include "HelvUC.inc"
#include "vislib/Verdana.inc"


using namespace megamol::core;


/*
 * cluster::AbstractClusterView::AbstractClusterView
 */
cluster::AbstractClusterView::AbstractClusterView(void) : view::AbstractTileView(),
        ClusterControllerClient() {

    // slot initialized in 'ClusterControllerClient::ctor'
    this->MakeSlotAvailable(&this->registerSlot);

    // TODO: Implement

}


/*
 * cluster::AbstractClusterView::~AbstractClusterView
 */
cluster::AbstractClusterView::~AbstractClusterView(void) {

    // TODO: Implement

}


/*
 * cluster::AbstractClusterView::ResetView
 */
void cluster::AbstractClusterView::ResetView(void) {
    // intentionally empty to disallow local user input
}


/*
 * cluster::AbstractClusterView::SetCursor2DButtonState
 */
void cluster::AbstractClusterView::SetCursor2DButtonState(unsigned int btn, bool down) {
    // intentionally empty to disallow local user input
}


/*
 * cluster::AbstractClusterView::SetCursor2DPosition
 */
void cluster::AbstractClusterView::SetCursor2DPosition(float x, float y) {
    // intentionally empty to disallow local user input
}


/*
 * cluster::AbstractClusterView::SetInputModifier
 */
void cluster::AbstractClusterView::SetInputModifier(mmcInputModifier mod, bool down) {
    // intentionally empty to disallow local user input
}


/*
 * cluster::AbstractClusterView::renderFallbackView
 */
void cluster::AbstractClusterView::renderFallbackView(void) {

    ::glViewport(0, 0, this->getViewportWidth(), this->getViewportHeight());
    ::glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    ::glClear(GL_COLOR_BUFFER_BIT);

    if ((this->getViewportHeight() <= 1) || (this->getViewportWidth() <= 1)) return;

    ::glMatrixMode(GL_PROJECTION);
    ::glLoadIdentity();
    float aspect = static_cast<float>(this->getViewportWidth())
        / static_cast<float>(this->getViewportHeight());
    if ((this->getProjType() == vislib::graphics::CameraParameters::MONO_PERSPECTIVE)
            || (this->getProjType() == vislib::graphics::CameraParameters::MONO_ORTHOGRAPHIC)) {
        if (aspect > 1.0f) {
            ::glScalef(2.0f / aspect, -2.0f, 1.0f);
        } else {
            ::glScalef(2.0f, -2.0f * aspect, 1.0f);
        }
        ::glTranslatef(-0.5f, -0.5f, 0.0f);
    } else {
        if (this->getEye() == vislib::graphics::CameraParameters::RIGHT_EYE) {
            ::glTranslatef(0.5f, 0.0f, 0.0f);
        } else {
            ::glTranslatef(-0.5f, 0.0f, 0.0f);
        }
        if (aspect > 2.0f) {
            ::glScalef(2.0f / aspect, -2.0f, 1.0f);
        } else {
            ::glScalef(1.0f, -1.0f * aspect, 1.0f);
        }
        ::glTranslatef(-0.5f, -0.5f, 0.0f);
    }
    const float border = 0.05f;
    ::glTranslatef(border, border, 0.0f);
    ::glScalef(1.0f - 2.0f * border, 1.0f - 2.0f * border, 0.0f);

    ::glMatrixMode(GL_MODELVIEW);
    ::glLoadIdentity();

    IconState icon;
    vislib::TString message;



    // TODO: Build real message and icon state
    icon = static_cast<IconState>(vislib::sys::GetTicksOfDay() % 40000 / 10000);
    switch (icon) {
        case ICONSTATE_ERROR:
            message = _T("DEBUG: I don't feel so good");
            break;
        case ICONSTATE_OK:
            message = _T("DEBUG: Everything looks fine from here");
            break;
        case ICONSTATE_WAIT:
            message = _T("DEBUG: Give me some time");
            break;
        case ICONSTATE_WORK:
            message = _T("DEBUG: Working for you ...");
            break;
    }



    unsigned char r, g, b;
    switch (icon) {
        case ICONSTATE_ERROR: r = 255; g = 0; b = 0; break;
        case ICONSTATE_OK: r = 40; g = 255; b = 40; break;
        default: r = 210; g = 212; b = 217; break;
    }
    renderInfoIconBorder(r, g, b);
    switch (icon) {
        case ICONSTATE_ERROR:
            renderErrorInfoIcon(r, g, b, message);
            break;
        case ICONSTATE_OK:
            renderOKInfoIcon(r, g, b, message);
            break;
        case ICONSTATE_WAIT:
            renderWaitInfoIcon(r, g, b, message);
            break;
        case ICONSTATE_WORK:
            renderWorkingInfoIcon(r, g, b, message);
            break;
        default:
            renderErrorInfoIcon(r, g, b,
                vislib::TString("ERROR: Unable to query internal state"));
            break;
    }
}


/*
 * cluster::AbstractClusterView::cornerPtCnt
 */
const int cluster::AbstractClusterView::cornerPtCnt = 24;


/*
 * cluster::AbstractClusterView::cornerBigRad
 */
const float cluster::AbstractClusterView::cornerBigRad = 0.08f;


/*
 * cluster::AbstractClusterView::cornerMidRad
 */
const float cluster::AbstractClusterView::cornerMidRad = 0.04f;


/*
 * cluster::AbstractClusterView::cornerSmlRad
 */
const float cluster::AbstractClusterView::cornerSmlRad = 0.03f;


/*
 * cluster::AbstractClusterView::borderSize
 */
const float cluster::AbstractClusterView::borderSize = 0.06f;


/*
 * cluster::AbstractClusterView::infoFont
 */
const vislib::graphics::AbstractFont& cluster::AbstractClusterView::infoFont(void) {
    // TODO: Change to a global font object
    const static vislib::graphics::gl::OutlineFont font(
        vislib::graphics::gl::FontInfo_Verdana,
        vislib::graphics::gl::OutlineFont::RENDERTYPE_FILL_AND_OUTLINE);
    return font;
}


/*
 * cluster::AbstractClusterView::infoIconBorderCaption
 */
vislib::TString cluster::AbstractClusterView::infoIconBorderCaption(void) {
    vislib::TString rv;
    vislib::TString t;
    vislib::sys::SystemInformation::ComputerName(t);
    if (t.IsEmpty()) {
        t = _T("Unnamed Computer");
    }
    rv.Format(_T(" %s "), t.PeekBuffer());
    return rv;
}


/*
 * cluster::AbstractClusterView::renderErrorInfoIcon
 */
void cluster::AbstractClusterView::renderErrorInfoIcon(unsigned char colR,
        unsigned char colG, unsigned char colB,
        const vislib::TString& message) {
    const float border = 2.0f * borderSize;

    setupRendering();
    ::glColor3ub(colR, colG, colB);

    if ((vislib::sys::GetTicksOfDay() % 1500) < 750) {
        // outline
        ::glBegin(GL_LINE_LOOP);
        for (int i = 0; i < cornerPtCnt; i++) {
            float a = static_cast<float>(M_PI) * 0.5f
                * static_cast<float>(i) / static_cast<float>(cornerPtCnt - 1);
            ::glVertex2f(border + cornerSmlRad * (1.0f - sin(a)),
                border + cornerSmlRad * (1.0f - cos(a)));
        }
        for (int i = 0; i < cornerPtCnt; i++) {
            float a = static_cast<float>(M_PI) * 0.5f
                * static_cast<float>(i) / static_cast<float>(cornerPtCnt - 1);
            ::glVertex2f(border + cornerSmlRad * (1.0f - cos(a)),
                1.0f - (border + cornerSmlRad * (1.0f - sin(a))));
        }
        for (int i = 0; i < cornerPtCnt; i++) {
            float a = static_cast<float>(M_PI) * 0.5f
                * static_cast<float>(i) / static_cast<float>(cornerPtCnt - 1);
            ::glVertex2f(1.0f - (border + cornerSmlRad * (1.0f - sin(a))),
                1.0f - (border + cornerSmlRad * (1.0f - cos(a))));
        }
        for (int i = 0; i < cornerPtCnt; i++) {
            float a = static_cast<float>(M_PI) * 0.5f
                * static_cast<float>(i) / static_cast<float>(cornerPtCnt - 1);
            ::glVertex2f(1.0f - (border + cornerSmlRad * (1.0f - cos(a))),
                border + cornerSmlRad * (1.0f - sin(a)));
        }
        ::glEnd();

        // filled
        ::glBegin(GL_TRIANGLE_FAN);
        for (int i = 0; i < cornerPtCnt; i++) {
            float a = static_cast<float>(M_PI) * 0.5f
                * static_cast<float>(i) / static_cast<float>(cornerPtCnt - 1);
            ::glVertex2f(border + cornerSmlRad * (1.0f - sin(a)),
                border + cornerSmlRad * (1.0f - cos(a)));
        }
        for (int i = 0; i < cornerPtCnt; i++) {
            float a = static_cast<float>(M_PI) * 0.5f
                * static_cast<float>(i) / static_cast<float>(cornerPtCnt - 1);
            ::glVertex2f(border + cornerSmlRad * (1.0f - cos(a)),
                1.0f - (border + cornerSmlRad * (1.0f - sin(a))));
        }
        for (int i = 0; i < cornerPtCnt; i++) {
            float a = static_cast<float>(M_PI) * 0.5f
                * static_cast<float>(i) / static_cast<float>(cornerPtCnt - 1);
            ::glVertex2f(1.0f - (border + cornerSmlRad * (1.0f - sin(a))),
                1.0f - (border + cornerSmlRad * (1.0f - cos(a))));
        }
        for (int i = 0; i < cornerPtCnt; i++) {
            float a = static_cast<float>(M_PI) * 0.5f
                * static_cast<float>(i) / static_cast<float>(cornerPtCnt - 1);
            ::glVertex2f(1.0f - (border + cornerSmlRad * (1.0f - cos(a))),
                border + cornerSmlRad * (1.0f - sin(a)));
        }
        ::glEnd();

        ::glColor3ub(0, 0, 0);
    }

    ::glPushMatrix();
    ::glTranslatef(0.5f, message.IsEmpty() ? 0.5f : 0.425f, 0.0f);
    ::glScalef(0.4f, 0.4f, 1.0f);

    // outline
    ::glBegin(GL_LINE_LOOP);
    ::glVertex2d(-0.6, -0.4);
    ::glVertex2d(-0.4, -0.6);
    ::glVertex2d( 0.0, -0.2);
    ::glVertex2d( 0.4, -0.6);
    ::glVertex2d( 0.6, -0.4);
    ::glVertex2d( 0.2,  0.0);
    ::glVertex2d( 0.6,  0.4);
    ::glVertex2d( 0.4,  0.6);
    ::glVertex2d( 0.0,  0.2);
    ::glVertex2d(-0.4,  0.6);
    ::glVertex2d(-0.6,  0.4);
    ::glVertex2d(-0.2,  0.0);
    ::glEnd();

    // filled
    ::glBegin(GL_TRIANGLE_FAN);
    ::glVertex2d(0.0, 0.0);
    ::glVertex2d(-0.6, -0.4);
    ::glVertex2d(-0.4, -0.6);
    ::glVertex2d( 0.0, -0.2);
    ::glVertex2d( 0.4, -0.6);
    ::glVertex2d( 0.6, -0.4);
    ::glVertex2d( 0.2,  0.0);
    ::glVertex2d( 0.6,  0.4);
    ::glVertex2d( 0.4,  0.6);
    ::glVertex2d( 0.0,  0.2);
    ::glVertex2d(-0.4,  0.6);
    ::glVertex2d(-0.6,  0.4);
    ::glVertex2d(-0.2,  0.0);
    ::glVertex2d(-0.6, -0.4);
    ::glEnd();

    ::glPopMatrix();

    // message
    infoFont().DrawString(3.0f * borderSize, 3.0f * borderSize,
        1.0f - 6.0f * borderSize, 1.0f - 6.0f * borderSize, 0.75f * borderSize,
        message.PeekBuffer(), vislib::graphics::AbstractFont::ALIGN_CENTER_BOTTOM);

}


/*
 * cluster::AbstractClusterView::renderInfoIconBorder
 */
void cluster::AbstractClusterView::renderInfoIconBorder(unsigned char colR,
        unsigned char colG, unsigned char colB) {
    const static float borderStubSize = 0.04f;
    const static vislib::StringW compName = infoIconBorderCaption();
    const static vislib::graphics::gl::OutlineFont font(
        vislib::graphics::gl::FontInfo_Helvetica_UltraCompressed,
        vislib::graphics::gl::OutlineFont::RENDERTYPE_FILL_AND_OUTLINE);
    const static float fontSize = borderSize
        / vislib::graphics::gl::FontInfo_Helvetica_UltraCompressed.charHeight;
    const static float fontTopLine = fontSize
        * (vislib::graphics::gl::FontInfo_Helvetica_UltraCompressed.charHeight
        - vislib::graphics::gl::FontInfo_Helvetica_UltraCompressed.baseline);
    const static float captionMaxWidth = 1.0f
        - 2.0f * (cornerBigRad + borderStubSize);
    const static float captionRealWidth = font.LineWidth(fontSize, compName);
    const static float captionWidth
        = vislib::math::Min(captionMaxWidth, captionRealWidth);

    setupRendering();
    ::glColor3ub(colR, colG, colB);

    // outline
    ::glBegin(GL_LINE_LOOP);
    ::glVertex2f(cornerBigRad + borderStubSize, 0.0f);
    for (int i = 0; i < cornerPtCnt; i++) {
        float a = static_cast<float>(M_PI) * 0.5f
            * static_cast<float>(i) / static_cast<float>(cornerPtCnt - 1);
        ::glVertex2f(cornerBigRad * (1.0f - sin(a)),
            cornerBigRad * (1.0f - cos(a)));
    }
    for (int i = 0; i < cornerPtCnt; i++) {
        float a = static_cast<float>(M_PI) * 0.5f
            * static_cast<float>(i) / static_cast<float>(cornerPtCnt - 1);
        ::glVertex2f(cornerBigRad * (1.0f - cos(a)),
            1.0f - cornerBigRad * (1.0f - sin(a)));
    }
    for (int i = 0; i < cornerPtCnt; i++) {
        float a = static_cast<float>(M_PI) * 0.5f
            * static_cast<float>(i) / static_cast<float>(cornerPtCnt - 1);
        ::glVertex2f(1.0f - cornerBigRad * (1.0f - sin(a)),
            1.0f - cornerBigRad * (1.0f - cos(a)));
    }
    for (int i = 0; i < cornerPtCnt; i++) {
        float a = static_cast<float>(M_PI) * 0.5f
            * static_cast<float>(i) / static_cast<float>(cornerPtCnt - 1);
        ::glVertex2f(1.0f - cornerBigRad * (1.0f - cos(a)),
            cornerBigRad * (1.0f - sin(a)));
    }
    ::glVertex2f(cornerBigRad + borderStubSize + captionWidth, 0.0f);
    ::glVertex2f(cornerBigRad + borderStubSize + captionWidth, borderSize);
    for (int i = 0; i < cornerPtCnt; i++) {
        float a = static_cast<float>(M_PI) * 0.5f
            * static_cast<float>(i) / static_cast<float>(cornerPtCnt - 1);
        ::glVertex2f(1.0f - cornerMidRad - borderSize + cornerMidRad * sin(a),
            cornerMidRad + borderSize - cornerMidRad * cos(a));
    }
    for (int i = 0; i < cornerPtCnt; i++) {
        float a = static_cast<float>(M_PI) * 0.5f
            * static_cast<float>(i) / static_cast<float>(cornerPtCnt - 1);
        ::glVertex2f(1.0f - cornerMidRad - borderSize + cornerMidRad * cos(a),
            1.0f - cornerMidRad - borderSize + cornerMidRad * sin(a));
    }
    for (int i = 0; i < cornerPtCnt; i++) {
        float a = static_cast<float>(M_PI) * 0.5f
            * static_cast<float>(i) / static_cast<float>(cornerPtCnt - 1);
        ::glVertex2f(cornerMidRad + borderSize - cornerMidRad * sin(a),
            1.0f - cornerMidRad - borderSize + cornerMidRad * cos(a));
    }
    for (int i = 0; i < cornerPtCnt; i++) {
        float a = static_cast<float>(M_PI) * 0.5f
            * static_cast<float>(i) / static_cast<float>(cornerPtCnt - 1);
        ::glVertex2f(cornerMidRad + borderSize - cornerMidRad * cos(a),
            cornerMidRad + borderSize - cornerMidRad * sin(a));
    }
    ::glVertex2f(cornerBigRad + borderStubSize, borderSize);
    ::glEnd();

    // filling
    ::glBegin(GL_QUAD_STRIP);
    ::glVertex2f(cornerBigRad + borderStubSize, 0.0f);
    ::glVertex2f(cornerBigRad + borderStubSize, borderSize);
    for (int i = 0; i < cornerPtCnt; i++) {
        float a = static_cast<float>(M_PI) * 0.5f
            * static_cast<float>(i) / static_cast<float>(cornerPtCnt - 1);
        ::glVertex2f(cornerBigRad * (1.0f - sin(a)),
            cornerBigRad * (1.0f - cos(a)));
        ::glVertex2f(borderSize + cornerMidRad * (1.0f - sin(a)),
            borderSize + cornerMidRad * (1.0f - cos(a)));
    }
    for (int i = 0; i < cornerPtCnt; i++) {
        float a = static_cast<float>(M_PI) * 0.5f
            * static_cast<float>(i) / static_cast<float>(cornerPtCnt - 1);
        ::glVertex2f(cornerBigRad * (1.0f - cos(a)),
            1.0f - cornerBigRad * (1.0f - sin(a)));
        ::glVertex2f(borderSize + cornerMidRad * (1.0f - cos(a)),
            1.0f - borderSize - cornerMidRad * (1.0f - sin(a)));
    }
    for (int i = 0; i < cornerPtCnt; i++) {
        float a = static_cast<float>(M_PI) * 0.5f
            * static_cast<float>(i) / static_cast<float>(cornerPtCnt - 1);
        ::glVertex2f(1.0f - cornerBigRad * (1.0f - sin(a)),
            1.0f - cornerBigRad * (1.0f - cos(a)));
        ::glVertex2f(1.0f - borderSize - cornerMidRad * (1.0f - sin(a)),
            1.0f - borderSize - cornerMidRad * (1.0f - cos(a)));
    }
    for (int i = 0; i < cornerPtCnt; i++) {
        float a = static_cast<float>(M_PI) * 0.5f
            * static_cast<float>(i) / static_cast<float>(cornerPtCnt - 1);
        ::glVertex2f(1.0f - cornerBigRad * (1.0f - cos(a)),
            cornerBigRad * (1.0f - sin(a)));
        ::glVertex2f(1.0f - borderSize - cornerMidRad * (1.0f - cos(a)),
            borderSize + cornerMidRad * (1.0f - sin(a)));
    }
    ::glVertex2f(cornerBigRad + borderStubSize + captionWidth, 0.0f);
    ::glVertex2f(cornerBigRad + borderStubSize + captionWidth, borderSize);
    ::glEnd();

    // caption
    if (captionRealWidth > captionMaxWidth) {
        ::glPushMatrix();
        ::glTranslatef(cornerBigRad + borderStubSize, fontTopLine, 0.0f);
        ::glScalef(captionMaxWidth / captionRealWidth, 1.0f, 1.0f);
        font.DrawString(0.0f, 0.0f, fontSize, false, compName);
        ::glPopMatrix();
    } else {
        font.DrawString(cornerBigRad + borderStubSize, fontTopLine, fontSize,
            false, compName);
    }

}


/*
 * cluster::AbstractClusterView::renderOKInfoIcon
 */
void cluster::AbstractClusterView::renderOKInfoIcon(unsigned char colR,
        unsigned char colG, unsigned char colB,
        const vislib::TString& message) {
    const float border = 2.0f * borderSize;

    setupRendering();
    ::glColor3ub(colR, colG, colB);

    if ((vislib::sys::GetTicksOfDay() % 1500) < 750) {
        // outline
        ::glBegin(GL_LINE_LOOP);
        for (int i = 0; i < cornerPtCnt; i++) {
            float a = static_cast<float>(M_PI) * 0.5f
                * static_cast<float>(i) / static_cast<float>(cornerPtCnt - 1);
            ::glVertex2f(border + cornerSmlRad * (1.0f - sin(a)),
                border + cornerSmlRad * (1.0f - cos(a)));
        }
        for (int i = 0; i < cornerPtCnt; i++) {
            float a = static_cast<float>(M_PI) * 0.5f
                * static_cast<float>(i) / static_cast<float>(cornerPtCnt - 1);
            ::glVertex2f(border + cornerSmlRad * (1.0f - cos(a)),
                1.0f - (border + cornerSmlRad * (1.0f - sin(a))));
        }
        for (int i = 0; i < cornerPtCnt; i++) {
            float a = static_cast<float>(M_PI) * 0.5f
                * static_cast<float>(i) / static_cast<float>(cornerPtCnt - 1);
            ::glVertex2f(1.0f - (border + cornerSmlRad * (1.0f - sin(a))),
                1.0f - (border + cornerSmlRad * (1.0f - cos(a))));
        }
        for (int i = 0; i < cornerPtCnt; i++) {
            float a = static_cast<float>(M_PI) * 0.5f
                * static_cast<float>(i) / static_cast<float>(cornerPtCnt - 1);
            ::glVertex2f(1.0f - (border + cornerSmlRad * (1.0f - cos(a))),
                border + cornerSmlRad * (1.0f - sin(a)));
        }
        ::glEnd();

        // filled
        ::glBegin(GL_TRIANGLE_FAN);
        for (int i = 0; i < cornerPtCnt; i++) {
            float a = static_cast<float>(M_PI) * 0.5f
                * static_cast<float>(i) / static_cast<float>(cornerPtCnt - 1);
            ::glVertex2f(border + cornerSmlRad * (1.0f - sin(a)),
                border + cornerSmlRad * (1.0f - cos(a)));
        }
        for (int i = 0; i < cornerPtCnt; i++) {
            float a = static_cast<float>(M_PI) * 0.5f
                * static_cast<float>(i) / static_cast<float>(cornerPtCnt - 1);
            ::glVertex2f(border + cornerSmlRad * (1.0f - cos(a)),
                1.0f - (border + cornerSmlRad * (1.0f - sin(a))));
        }
        for (int i = 0; i < cornerPtCnt; i++) {
            float a = static_cast<float>(M_PI) * 0.5f
                * static_cast<float>(i) / static_cast<float>(cornerPtCnt - 1);
            ::glVertex2f(1.0f - (border + cornerSmlRad * (1.0f - sin(a))),
                1.0f - (border + cornerSmlRad * (1.0f - cos(a))));
        }
        for (int i = 0; i < cornerPtCnt; i++) {
            float a = static_cast<float>(M_PI) * 0.5f
                * static_cast<float>(i) / static_cast<float>(cornerPtCnt - 1);
            ::glVertex2f(1.0f - (border + cornerSmlRad * (1.0f - cos(a))),
                border + cornerSmlRad * (1.0f - sin(a)));
        }
        ::glEnd();

        ::glColor3ub(0, 0, 0);
    }

    ::glPushMatrix();
    ::glTranslatef(0.5f, message.IsEmpty() ? 0.5f : 0.425f, 0.0f);
    ::glScalef(0.45f, 0.45f, 1.0f);

    // outline
    ::glBegin(GL_LINE_LOOP);
    ::glVertex2d(-0.2,  0.1);
    ::glVertex2d(-0.4, -0.1);
    ::glVertex2d(-0.6,  0.1);
    ::glVertex2d(-0.2,  0.5);
    ::glVertex2d( 0.6, -0.3);
    ::glVertex2d( 0.4, -0.5);
    ::glEnd();

    // filled
    ::glBegin(GL_TRIANGLE_FAN);
    ::glVertex2d(-0.2,  0.1);
    ::glVertex2d(-0.4, -0.1);
    ::glVertex2d(-0.6,  0.1);
    ::glVertex2d(-0.2,  0.5);
    ::glVertex2d( 0.6, -0.3);
    ::glVertex2d( 0.4, -0.5);
    ::glEnd();

    ::glPopMatrix();

    // message
    infoFont().DrawString(3.0f * borderSize, 3.0f * borderSize,
        1.0f - 6.0f * borderSize, 1.0f - 6.0f * borderSize, 0.75f * borderSize,
        message.PeekBuffer(), vislib::graphics::AbstractFont::ALIGN_CENTER_BOTTOM);

}


/*
 * cluster::AbstractClusterView::renderWaitInfoIcon
 */
void cluster::AbstractClusterView::renderWaitInfoIcon(unsigned char colR,
        unsigned char colG, unsigned char colB,
        const vislib::TString& message) {
    const float scale = (1.0f - 8.0f * borderSize) * 0.5f;
    const int circlePtCnt = 24 * cornerPtCnt;
    const int tipPtCnt = cornerPtCnt;
    const float rad = 0.15f;

    setupRendering();
    float r = static_cast<float>(colR) / 255.0f;
    float g = static_cast<float>(colG) / 255.0f;
    float b = static_cast<float>(colB) / 255.0f;

    ::glPushMatrix();
    ::glTranslatef(0.5f, message.IsEmpty() ? 0.5f : 0.425f, 0.0f);
    ::glScalef(scale, scale, 1.0f);
    ::glRotatef(static_cast<float>(vislib::sys::GetTicksOfDay() % 1500) / 1500.0f * 360.0f, 0.0f, 0.0f, 1.0f);

    // outline
    ::glBegin(GL_LINE_LOOP);
    for (int i = 0; i < circlePtCnt; i++) {
        float a = 1.0f - static_cast<float>(i) / static_cast<float>(circlePtCnt - 1);
        float ang = a * 2.0f * static_cast<float>(M_PI);
        float rd = rad * ((i < tipPtCnt)
            ? vislib::math::Sqrt(1.0f - vislib::math::Sqr(1.0f - static_cast<float>(i) / static_cast<float>(tipPtCnt - 1)))
            : (1.0f - static_cast<float>(i - tipPtCnt)
                / static_cast<float>(circlePtCnt - tipPtCnt - 1)));
        float rr = 1.0f - rad;
        ::glColor3f(r * a, g * a, b * a);
        ::glVertex2f((rr + rd) * cos(ang), (rr + rd) * sin(ang));
    }
    for (int i = circlePtCnt - 2; i > 0; i--) {
        float a = 1.0f - static_cast<float>(i) / static_cast<float>(circlePtCnt - 1);
        float ang = a * 2.0f * static_cast<float>(M_PI);
        float rd = rad * ((i < tipPtCnt)
            ? vislib::math::Sqrt(1.0f - vislib::math::Sqr(1.0f - static_cast<float>(i) / static_cast<float>(tipPtCnt - 1)))
            : (1.0f - static_cast<float>(i - tipPtCnt)
                / static_cast<float>(circlePtCnt - tipPtCnt - 1)));
        float rr = 1.0f - rad;
        ::glColor3f(r * a, g * a, b * a);
        ::glVertex2f((rr - rd) * cos(ang), (rr - rd) * sin(ang));
    }
    ::glEnd();

    // filled
    ::glBegin(GL_QUAD_STRIP);
    for (int i = 0; i < circlePtCnt; i++) {
        float a = 1.0f - static_cast<float>(i) / static_cast<float>(circlePtCnt - 1);
        float ang = a * 2.0f * static_cast<float>(M_PI);
        float rd = rad * ((i < tipPtCnt)
            ? vislib::math::Sqrt(1.0f - vislib::math::Sqr(1.0f - static_cast<float>(i) / static_cast<float>(tipPtCnt - 1)))
            : (1.0f - static_cast<float>(i - tipPtCnt)
                / static_cast<float>(circlePtCnt - tipPtCnt - 1)));
        float rr = 1.0f - rad;
        ::glColor3f(r * a, g * a, b * a);
        ::glVertex2f((rr + rd) * cos(ang), (rr + rd) * sin(ang));
        ::glVertex2f((rr - rd) * cos(ang), (rr - rd) * sin(ang));
    }
    ::glEnd();

    ::glPopMatrix();

    // message
    ::glColor3ub(colR, colG, colB);
    infoFont().DrawString(2.5f * borderSize, 2.5f * borderSize,
        1.0f - 5.0f * borderSize, 1.0f - 5.0f * borderSize, 0.75f * borderSize,
        message.PeekBuffer(), vislib::graphics::AbstractFont::ALIGN_CENTER_BOTTOM);

}


/*
 * cluster::AbstractClusterView::renderWorkingInfoIcon
 */
void cluster::AbstractClusterView::renderWorkingInfoIcon(unsigned char colR,
        unsigned char colG, unsigned char colB,
        const vislib::TString& message) {
    const float scale = 1.0f - 4.0f * borderSize;
    const int ticks = 6;
    const int tickTime = 150;
    const float tickSize = 1.0f / static_cast<float>(2 * ticks - 1);

    setupRendering();
    ::glColor3ub(colR, colG, colB);

    ::glPushMatrix();
    ::glTranslatef(2.0f * borderSize, 2.0f * borderSize, 0.0f);
    ::glScalef(scale, scale, 1.0f);
    ::glTranslatef(0.0f, (1.0f - tickSize) * 0.5f, 0.0f);

    int tick = (vislib::sys::GetTicksOfDay() % (ticks * tickTime)) / tickTime;
    float ftick = static_cast<float>(tick);

    for (int i = 0; i < 2; i++) {
        // outline
        ::glBegin(GL_LINE_LOOP);
        ::glVertex2f(ftick * tickSize,  0.0f);
        ::glVertex2f((1.0f + ftick) * tickSize,  0.0f);
        ::glVertex2f((1.0f + ftick) * tickSize,  tickSize);
        ::glVertex2f(ftick * tickSize,  tickSize);
        ::glEnd();
        // fill
        ::glBegin(GL_QUADS);
        ::glVertex2f(ftick * tickSize,  0.0f);
        ::glVertex2f((1.0f + ftick) * tickSize,  0.0f);
        ::glVertex2f((1.0f + ftick) * tickSize,  tickSize);
        ::glVertex2f(ftick * tickSize,  tickSize);
        ::glEnd();
        if (tick == ticks - 1) break;
        ftick = static_cast<float>(2 * ticks - 2 - tick);
    }

    ::glPopMatrix();

    // message
    infoFont().DrawString(2.5f * borderSize, 2.5f * borderSize,
        1.0f - 5.0f * borderSize, 1.0f - 5.0f * borderSize, 0.75f * borderSize,
        message.PeekBuffer(), vislib::graphics::AbstractFont::ALIGN_CENTER_BOTTOM);

}


/*
 * cluster::AbstractClusterView::setupRendering
 */
void cluster::AbstractClusterView::setupRendering(void) {
    ::glDisable(GL_TEXTURE);
    ::glDisable(GL_LIGHTING);
    ::glEnable(GL_LINE_SMOOTH);
    ::glEnable(GL_BLEND);
    ::glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    ::glLineWidth(1.1f);
}
