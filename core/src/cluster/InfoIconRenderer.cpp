/*
 * InfoIconRenderer.h
 *
 * Copyright (C) 2010 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#define _USE_MATH_DEFINES
#include "vislib/graphics/gl/IncludeAllGL.h"
#include "mmcore/cluster/InfoIconRenderer.h"
#include "mmcore/view/graphicsresources.h"
#include <cmath>
#include "vislib/math/mathfunctions.h"
#include "vislib/graphics/gl/OutlineFont.h"
#include "vislib/sys/sysfunctions.h"
#include "vislib/sys/SystemInformation.h"
#include "vislib/UnsupportedOperationException.h"
#include "HelvUC.inc"

using namespace megamol::core;


/*
 * cluster::InfoIconRenderer::RenderErrorInfoIcon
 */
void cluster::InfoIconRenderer::RenderErrorInfoIcon(unsigned char colR,
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
    view::GetGlobalFont(view::FONTPURPISE_OPENGL_INFO_HQ).DrawString(
        3.0f * borderSize, 3.0f * borderSize,
        1.0f - 6.0f * borderSize, 1.0f - 6.0f * borderSize,
        0.75f * borderSize, message.PeekBuffer(),
        vislib::graphics::AbstractFont::ALIGN_CENTER_BOTTOM);

}


/*
 * cluster::InfoIconRenderer::RenderInfoIcon
 */
void cluster::InfoIconRenderer::RenderInfoIcon(IconState icon,
        const vislib::TString& message) {
    unsigned char r, g, b;
    switch (icon) {
        case ICONSTATE_UNKNOWN: r = 0xe6; g = 0xcb; b = 0xc0; break;
        case ICONSTATE_ERROR: r = 255; g = 0; b = 0; break;
        case ICONSTATE_OK: r = 40; g = 255; b = 40; break;
        case ICONSTATE_WAIT: r = 210; g = 212; b = 217; break;
        case ICONSTATE_WORK: r = 210; g = 212; b = 217; break;
        default: {
            vislib::TString msg(_T("Error: Illegal Icon State specified"));
            if (!message.IsEmpty()) {
                vislib::TString m(msg);
                msg.Format(_T("%s\nMessage: %s"), m.PeekBuffer(), message.PeekBuffer());
            }
            RenderInfoIcon(ICONSTATE_ERROR, 255, 0, 0, msg);
            return;
        }
    }
    RenderInfoIcon(icon, r, g, b, message);
}


/*
 * cluster::InfoIconRenderer::RenderInfoIcon
 */
void cluster::InfoIconRenderer::RenderInfoIcon(IconState icon,
        unsigned char colR, unsigned char colG, unsigned char colB,
        const vislib::TString& message) {
    RenderInfoIconBorder(colR, colG, colB);
    switch (icon) {
        case ICONSTATE_UNKNOWN:
            RenderUnknownStateInfoIcon(colR, colG, colB, message);
            break;
        case ICONSTATE_ERROR:
            RenderErrorInfoIcon(colR, colG, colB, message);
            break;
        case ICONSTATE_OK:
            RenderOKInfoIcon(colR, colG, colB, message);
            break;
        case ICONSTATE_WAIT:
            RenderWaitInfoIcon(colR, colG, colB, message);
            break;
        case ICONSTATE_WORK:
            RenderWorkingInfoIcon(colR, colG, colB, message);
            break;
        default: {
            vislib::TString msg(_T("Error: Illegal Icon State specified"));
            if (!message.IsEmpty()) {
                vislib::TString m(msg);
                msg.Format(_T("%s\nMessage: %s"), m.PeekBuffer(), message.PeekBuffer());
            }
            RenderErrorInfoIcon(255, 0, 0, msg);
        } break;
    }
}


/*
 * cluster::InfoIconRenderer::RenderInfoIconBorder
 */
void cluster::InfoIconRenderer::RenderInfoIconBorder(unsigned char colR,
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
 * cluster::InfoIconRenderer::RenderOKInfoIcon
 */
void cluster::InfoIconRenderer::RenderOKInfoIcon(unsigned char colR,
        unsigned char colG, unsigned char colB,
        const vislib::TString& message) {

    setupRendering();
    ::glColor3ub(colR, colG, colB);

    ::glPushMatrix();
    ::glTranslatef(0.5f, message.IsEmpty() ? 0.5f : 0.425f, 0.0f);
    ::glScalef(0.45f, 0.45f, 1.0f);

    if ((vislib::sys::GetTicksOfDay() % 1000) < 500) {
        // outline
        ::glBegin(GL_LINE_LOOP);
        ::glVertex2d(-0.2,  0.0);
        ::glVertex2d(-0.4, -0.2);
        ::glVertex2d(-0.7,  0.1);
        ::glVertex2d(-0.2,  0.6);
        ::glVertex2d( 0.7, -0.3);
        ::glVertex2d( 0.4, -0.6);
        ::glEnd();

        // filled
        ::glBegin(GL_TRIANGLE_FAN);
        ::glVertex2d(-0.2,  0.0);
        ::glVertex2d(-0.4, -0.2);
        ::glVertex2d(-0.7,  0.1);
        ::glVertex2d(-0.2,  0.6);
        ::glVertex2d( 0.7, -0.3);
        ::glVertex2d( 0.4, -0.6);
        ::glEnd();

        ::glColor3ub(0, 0, 0);

        // outline
        ::glBegin(GL_LINE_LOOP);
        ::glVertex2d(-0.2,  0.2);
        ::glVertex2d(-0.4,  0.0);
        ::glVertex2d(-0.5,  0.1);
        ::glVertex2d(-0.2,  0.4);
        ::glVertex2d( 0.5, -0.3);
        ::glVertex2d( 0.4, -0.4);
        ::glEnd();

        // filled
        ::glBegin(GL_TRIANGLE_FAN);
        ::glVertex2d(-0.2,  0.2);
        ::glVertex2d(-0.4,  0.0);
        ::glVertex2d(-0.5,  0.1);
        ::glVertex2d(-0.2,  0.4);
        ::glVertex2d( 0.5, -0.3);
        ::glVertex2d( 0.4, -0.4);
        ::glEnd();

        ::glColor3ub(colR, colG, colB);

    } else {
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
    }

    ::glPopMatrix();

    // message
    view::GetGlobalFont(view::FONTPURPISE_OPENGL_INFO_HQ).DrawString(
        2.5f * borderSize, 2.5f * borderSize,
        1.0f - 5.0f * borderSize, 1.0f - 5.0f * borderSize,
        0.75f * borderSize, message.PeekBuffer(),
        vislib::graphics::AbstractFont::ALIGN_CENTER_BOTTOM);

}


/*
 * cluster::InfoIconRenderer::RenderUnknownStateInfoIcon
 */
void cluster::InfoIconRenderer::RenderUnknownStateInfoIcon(unsigned char colR,
        unsigned char colG, unsigned char colB,
        const vislib::TString& message) {

    setupRendering();
    ::glColor3ub(colR, colG, colB);

    ::glPushMatrix();
    ::glTranslatef(0.5f, message.IsEmpty() ? 0.5f : 0.425f, 0.0f);
    ::glScalef(0.45f, 0.45f, 1.0f);

    float a = static_cast<float>(vislib::sys::GetTicksOfDay() % 1500) / 1500.0f;
    a *= static_cast<float>(M_PI) * 2.0f;
    a = sin(a) * 20.0f;
    ::glTranslatef(0.0f, 1.0f, 0.0f);
    const vislib::graphics::AbstractFont &font = view::GetGlobalFont(view::FONTPURPISE_OPENGL_INFO_HQ);
    ::glTranslatef(-0.04f, -0.4f, 0.0f);
    ::glRotatef(a, 0.0f, 0.0f, 1.0f);
    ::glTranslatef(0.04f, 0.4f, 0.0f);
    font.DrawString(0.0f, 0.0f, 2.0f, "?", vislib::graphics::AbstractFont::ALIGN_CENTER_BOTTOM);

    ::glPopMatrix();

    // message
    view::GetGlobalFont(view::FONTPURPISE_OPENGL_INFO_HQ).DrawString(
        2.5f * borderSize, 2.5f * borderSize,
        1.0f - 5.0f * borderSize, 1.0f - 5.0f * borderSize,
        0.75f * borderSize, message.PeekBuffer(),
        vislib::graphics::AbstractFont::ALIGN_CENTER_BOTTOM);
}


/*
 * cluster::InfoIconRenderer::RenderWaitInfoIcon
 */
void cluster::InfoIconRenderer::RenderWaitInfoIcon(unsigned char colR,
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
    view::GetGlobalFont(view::FONTPURPISE_OPENGL_INFO_HQ).DrawString(
        2.5f * borderSize, 2.5f * borderSize,
        1.0f - 5.0f * borderSize, 1.0f - 5.0f * borderSize,
        0.75f * borderSize, message.PeekBuffer(),
        vislib::graphics::AbstractFont::ALIGN_CENTER_BOTTOM);

}


/*
 * cluster::InfoIconRenderer::RenderWorkingInfoIcon
 */
void cluster::InfoIconRenderer::RenderWorkingInfoIcon(unsigned char colR,
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
    view::GetGlobalFont(view::FONTPURPISE_OPENGL_INFO_HQ).DrawString(
        2.5f * borderSize, 2.5f * borderSize,
        1.0f - 5.0f * borderSize, 1.0f - 5.0f * borderSize,
        0.75f * borderSize, message.PeekBuffer(),
        vislib::graphics::AbstractFont::ALIGN_CENTER_BOTTOM);

}


/*
 * cluster::InfoIconRenderer::cornerPtCnt
 */
const int cluster::InfoIconRenderer::cornerPtCnt = 24;


/*
 * cluster::InfoIconRenderer::cornerBigRad
 */
const float cluster::InfoIconRenderer::cornerBigRad = 0.08f;


/*
 * cluster::InfoIconRenderer::cornerMidRad
 */
const float cluster::InfoIconRenderer::cornerMidRad = 0.04f;


/*
 * cluster::InfoIconRenderer::cornerSmlRad
 */
const float cluster::InfoIconRenderer::cornerSmlRad = 0.03f;


/*
 * cluster::InfoIconRenderer::borderSize
 */
const float cluster::InfoIconRenderer::borderSize = 0.06f;



/*
 * cluster::InfoIconRenderer::infoIconBorderCaption
 */
vislib::TString cluster::InfoIconRenderer::infoIconBorderCaption(void) {
    vislib::TString rv;
    vislib::TString t;
    vislib::sys::SystemInformation::ComputerName(t);
    if (t.IsEmpty()) {
        t = _T("Unnamed Computer");
    }
    rv.Format(_T(" %s "), t.PeekBuffer());
    rv.ToUpperCase();
    return rv;
}


/*
 * cluster::InfoIconRenderer::setupRendering
 */
void cluster::InfoIconRenderer::setupRendering(void) {
    ::glDisable(GL_TEXTURE);
    ::glDisable(GL_LIGHTING);
    ::glEnable(GL_LINE_SMOOTH);
    ::glEnable(GL_BLEND);
    ::glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    ::glLineWidth(1.1f);
}


/*
 * cluster::InfoIconRenderer::InfoIconRenderer
 */
cluster::InfoIconRenderer::InfoIconRenderer(void) {
    throw vislib::UnsupportedOperationException("InfoIconRenderer::ctor", __FILE__, __LINE__);
}
