/*
 * glfunctions.cpp
 *
 * Copyright (C) 2007 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#include "vislib_gl/graphics/gl/glfunctions.h"
#include "vislib_gl/graphics/gl/IncludeAllGL.h"

#include <stdlib.h>

//#ifdef _WIN32
//#include "GL/wglext.h"
//#else /* _WIN32 */
//// HAHA!
//#endif /* _WIN32 */

/*
 * vislib_gl::graphics::gl::DrawCuboidLines
 */
void vislib_gl::graphics::gl::DrawCuboidLines(const vislib::math::Cuboid<int>& box) {
    ::glBegin(GL_LINES);
    ::glVertex3i(box.Left(), box.Bottom(), box.Back());
    ::glVertex3i(box.Left(), box.Bottom(), box.Front());
    ::glVertex3i(box.Left(), box.Top(), box.Back());
    ::glVertex3i(box.Left(), box.Top(), box.Front());
    ::glVertex3i(box.Right(), box.Bottom(), box.Back());
    ::glVertex3i(box.Right(), box.Bottom(), box.Front());
    ::glVertex3i(box.Right(), box.Top(), box.Back());
    ::glVertex3i(box.Right(), box.Top(), box.Front());

    ::glVertex3i(box.Left(), box.Bottom(), box.Back());
    ::glVertex3i(box.Left(), box.Top(), box.Back());
    ::glVertex3i(box.Left(), box.Bottom(), box.Front());
    ::glVertex3i(box.Left(), box.Top(), box.Front());
    ::glVertex3i(box.Right(), box.Bottom(), box.Back());
    ::glVertex3i(box.Right(), box.Top(), box.Back());
    ::glVertex3i(box.Right(), box.Bottom(), box.Front());
    ::glVertex3i(box.Right(), box.Top(), box.Front());

    ::glVertex3i(box.Left(), box.Bottom(), box.Front());
    ::glVertex3i(box.Right(), box.Bottom(), box.Front());
    ::glVertex3i(box.Left(), box.Top(), box.Front());
    ::glVertex3i(box.Right(), box.Top(), box.Front());
    ::glVertex3i(box.Left(), box.Bottom(), box.Back());
    ::glVertex3i(box.Right(), box.Bottom(), box.Back());
    ::glVertex3i(box.Left(), box.Top(), box.Back());
    ::glVertex3i(box.Right(), box.Top(), box.Back());
    ::glEnd();
}


/*
 * vislib_gl::graphics::gl::DrawCuboidLines
 */
void vislib_gl::graphics::gl::DrawCuboidLines(const vislib::math::Cuboid<float>& box) {
    ::glBegin(GL_LINES);
    ::glVertex3f(box.Left(), box.Bottom(), box.Back());
    ::glVertex3f(box.Left(), box.Bottom(), box.Front());
    ::glVertex3f(box.Left(), box.Top(), box.Back());
    ::glVertex3f(box.Left(), box.Top(), box.Front());
    ::glVertex3f(box.Right(), box.Bottom(), box.Back());
    ::glVertex3f(box.Right(), box.Bottom(), box.Front());
    ::glVertex3f(box.Right(), box.Top(), box.Back());
    ::glVertex3f(box.Right(), box.Top(), box.Front());

    ::glVertex3f(box.Left(), box.Bottom(), box.Back());
    ::glVertex3f(box.Left(), box.Top(), box.Back());
    ::glVertex3f(box.Left(), box.Bottom(), box.Front());
    ::glVertex3f(box.Left(), box.Top(), box.Front());
    ::glVertex3f(box.Right(), box.Bottom(), box.Back());
    ::glVertex3f(box.Right(), box.Top(), box.Back());
    ::glVertex3f(box.Right(), box.Bottom(), box.Front());
    ::glVertex3f(box.Right(), box.Top(), box.Front());

    ::glVertex3f(box.Left(), box.Bottom(), box.Front());
    ::glVertex3f(box.Right(), box.Bottom(), box.Front());
    ::glVertex3f(box.Left(), box.Top(), box.Front());
    ::glVertex3f(box.Right(), box.Top(), box.Front());
    ::glVertex3f(box.Left(), box.Bottom(), box.Back());
    ::glVertex3f(box.Right(), box.Bottom(), box.Back());
    ::glVertex3f(box.Left(), box.Top(), box.Back());
    ::glVertex3f(box.Right(), box.Top(), box.Back());
    ::glEnd();
}


/*
 * vislib_gl::graphics::gl::DrawCuboidLines
 */
void vislib_gl::graphics::gl::DrawCuboidLines(const vislib::math::Cuboid<double>& box) {
    ::glBegin(GL_LINES);
    ::glVertex3d(box.Left(), box.Bottom(), box.Back());
    ::glVertex3d(box.Left(), box.Bottom(), box.Front());
    ::glVertex3d(box.Left(), box.Top(), box.Back());
    ::glVertex3d(box.Left(), box.Top(), box.Front());
    ::glVertex3d(box.Right(), box.Bottom(), box.Back());
    ::glVertex3d(box.Right(), box.Bottom(), box.Front());
    ::glVertex3d(box.Right(), box.Top(), box.Back());
    ::glVertex3d(box.Right(), box.Top(), box.Front());

    ::glVertex3d(box.Left(), box.Bottom(), box.Back());
    ::glVertex3d(box.Left(), box.Top(), box.Back());
    ::glVertex3d(box.Left(), box.Bottom(), box.Front());
    ::glVertex3d(box.Left(), box.Top(), box.Front());
    ::glVertex3d(box.Right(), box.Bottom(), box.Back());
    ::glVertex3d(box.Right(), box.Top(), box.Back());
    ::glVertex3d(box.Right(), box.Bottom(), box.Front());
    ::glVertex3d(box.Right(), box.Top(), box.Front());

    ::glVertex3d(box.Left(), box.Bottom(), box.Front());
    ::glVertex3d(box.Right(), box.Bottom(), box.Front());
    ::glVertex3d(box.Left(), box.Top(), box.Front());
    ::glVertex3d(box.Right(), box.Top(), box.Front());
    ::glVertex3d(box.Left(), box.Bottom(), box.Back());
    ::glVertex3d(box.Right(), box.Bottom(), box.Back());
    ::glVertex3d(box.Left(), box.Top(), box.Back());
    ::glVertex3d(box.Right(), box.Top(), box.Back());
    ::glEnd();
}
