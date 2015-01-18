/*
 * SimpleFontTest.h
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#ifndef VISLIBTEST_SIMPLEFONTTEST_H_INCLUDED
#define VISLIBTEST_SIMPLEFONTTEST_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "AbstractGlutApp.h"
#include "vislib/AbstractFont.h"
#include "vislib/CameraOpenGL.h"
#include "vislib/CameraRotate2DLookAt.h"
#include "vislib/Cursor2D.h"
#include "vislib/InputModifiers.h"
#include "vislib/mathfunctions.h"
#include "vislib/PtrArray.h"
#include "vislib/types.h"


/*
 * Test for vislib::graphics::gl::SimpleFont
 */
class SimpleFontTest: public AbstractGlutApp {
public:
    SimpleFontTest(void);
    virtual ~SimpleFontTest(void);

    virtual int GLInit(void);
    virtual void GLDeinit(void);

    virtual void OnResize(unsigned int w, unsigned int h);
    virtual void Render(void);
    virtual bool OnKeyPress(unsigned char key, int x, int y);
    virtual void OnMouseEvent(int button, int state, int x, int y);
    virtual void OnMouseMove(int x, int y);
    virtual void OnSpecialKey(int key, int x, int y);

private:

    class AbstractTest {
    protected:
        float x, y, z;
        unsigned char col[3];
        vislib::graphics::AbstractFont *& font;
        float size;
        bool flipY;
        vislib::graphics::AbstractFont::Alignment align;
        AbstractTest(float x, float y, float z, unsigned char colr, unsigned char colg, unsigned char colb,
                vislib::graphics::AbstractFont *& font, float size, bool flipY, vislib::graphics::AbstractFont::Alignment align)
                : x(x), y(y), z(z), font(font), size(size), flipY(flipY), align(align), active(true) {
            this->col[0] = colr;
            this->col[1] = colg;
            this->col[2] = colb;
        }
    public:
        static int Compare(AbstractTest* const & lhs, AbstractTest* const & rhs) {
            float v = rhs->z - lhs->z;
            if (vislib::math::IsEqual(0.0f, v)) {
                return 0;
            } else if (v > 0.0f) {
                return 1;
            } else {
                return -1;
            }
        }

        virtual ~AbstractTest(void) {
            this->font = NULL; // DO NOT DELETE
        }
        inline float Z(void) const {
            return this->z;
        }
        virtual void Draw(void) const = 0;
        bool active;
    };

    template<class T> class BoxTest : public AbstractTest {
    protected:
        float w, h;
        vislib::String<vislib::CharTraits<T> > txt;
    public:
        BoxTest(float x, float y, float w, float h, float z, unsigned char colr, unsigned char colg, unsigned char colb,
                vislib::graphics::AbstractFont *& font, float size, bool flipY, vislib::graphics::AbstractFont::Alignment align,
                const vislib::String<vislib::CharTraits<T> >& txt)
                : AbstractTest(x, y, z, colr, colg, colb, font, size, flipY, align), w(w), h(h), txt(txt) {
        }
        virtual ~BoxTest(void) {
        }
        virtual void Draw(void) const;
    };

    template<class T> class LineTest : public AbstractTest {
    protected:
        vislib::String<vislib::CharTraits<T> > txt;
    public:
        LineTest(float x, float y, float z, unsigned char colr, unsigned char colg, unsigned char colb,
                vislib::graphics::AbstractFont *& font, float size, bool flipY, vislib::graphics::AbstractFont::Alignment align,
                const vislib::String<vislib::CharTraits<T> >& txt)
                : AbstractTest(x, y, z, colr, colg, colb, font, size, flipY, align), txt(txt) {
        }
        virtual ~LineTest(void) {
        }
        virtual void Draw(void) const;
    };

    vislib::graphics::gl::CameraOpenGL camera;
    vislib::graphics::InputModifiers modkeys;
    vislib::graphics::Cursor2D cursor;
    vislib::graphics::CameraRotate2DLookAt rotator;

    vislib::PtrArray<AbstractTest> tests;

    vislib::graphics::AbstractFont *font1;
    bool rot;

};

#endif /* VISLIBTEST_SIMPLEFONTTEST_H_INCLUDED */
