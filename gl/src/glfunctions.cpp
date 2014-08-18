/*
 * glfunctions.cpp
 *
 * Copyright (C) 2007 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#include "vislib/glfunctions.h"
#include "glh/glh_genext.h"

#include <stdlib.h>

#ifdef _WIN32
#include "GL/wglext.h"
#else /* _WIN32 */
// HAHA!
#endif /* _WIN32 */

#ifdef _WIN32
/**
 * Answers whether or not a given wgl extension is supported.
 *
 * @param extensionName The name of the extension to search for.
 *
 * @return 0 if the extension is not supported,
 *         1 if the extension is supported, or
 *         -1 if there was an error while asking for the extension strings.
 */
int WGLExtensionSupported(const char *extensionName) {
    // this is pointer to function which returns pointer to string with list of all wgl extensions
    PFNWGLGETEXTENSIONSSTRINGEXTPROC wglGetExtensionsStringEXT = NULL;

    // determine pointer to wglGetExtensionsStringEXT function
    wglGetExtensionsStringEXT = (PFNWGLGETEXTENSIONSSTRINGEXTPROC) wglGetProcAddress("wglGetExtensionsStringEXT");

    if (wglGetExtensionsStringEXT == NULL) {
        return -1;
    }

    if (strstr(wglGetExtensionsStringEXT(), extensionName) == NULL) {
        // string was not found
        return 0;
    }

    // extension is supported
    return 1;
}
#else /* _WIN32 */
// HAHA!
#endif /* _WIN32 */


/*
 * vislib::graphics::gl::DrawCuboidLines
 */
void vislib::graphics::gl::DrawCuboidLines(
        const vislib::math::Cuboid<int>& box) {
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
 * vislib::graphics::gl::DrawCuboidLines
 */
void vislib::graphics::gl::DrawCuboidLines(
        const vislib::math::Cuboid<float>& box) {
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
 * vislib::graphics::gl::DrawCuboidLines
 */
void vislib::graphics::gl::DrawCuboidLines(
        const vislib::math::Cuboid<double>& box) {
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


/*
 * vislib::graphics::gl::EnableVSync
 */
bool vislib::graphics::gl::EnableVSync(bool enable) {
#ifdef _WIN32
    static PFNWGLSWAPINTERVALEXTPROC wglSwapIntervalEXT = NULL;
    static bool initialised = false;
    if (!initialised) {
        initialised = true;
        int support = WGLExtensionSupported("WGL_EXT_swap_control");
        if (support == 1) {
            wglSwapIntervalEXT = (PFNWGLSWAPINTERVALEXTPROC)
                wglGetProcAddress("wglSwapIntervalEXT");
        }
    }
    if (wglSwapIntervalEXT != NULL) {
        wglSwapIntervalEXT(enable ? 1 : 0);
        bool rvError = false;
        bool rvI = IsVSyncEnabled(&rvError);
        if (!rvError) return (rvI == enable);
    }
    return false;
#else /* _WIN32 */

    // LINUX DOES NOT WORK AT ALL!!!
    //  GLX_SGI_swap_control cannot be disabled!
    //  __GL_SYNC_TO_VBLANK cannot be changed on runtime (at least not while context exists)
    //  GLX_MESA_fuckup is no longer supported
    //  nvidia-settings does not seem to have any effect

    return false;
#endif /* _WIN32 */
}


/*
 * vislib::graphics::gl::GLVersion
 */
const vislib::VersionNumber& vislib::graphics::gl::GLVersion(void) {
    static VersionNumber number(0, 0, 0, 0);
    if (number.GetMajorVersionNumber() == 0) {
        // fetch version string
        vislib::StringA verStr(reinterpret_cast<const char*>(
            glGetString(GL_VERSION)));
        verStr.TrimSpaces();
        int major = 1, minor = 0, release = 0;

        // truncate vendor information
        int pos = verStr.Find(' ');
        if (pos > 0) {
            verStr.Truncate(pos);
        }

        // parse major version
        pos = verStr.Find('.');
        if (pos > 0) {
            major = CharTraitsA::ParseInt(verStr.Substring(0, pos));
            verStr = verStr.Substring(pos + 1);
        } else {
            // error fallback
            number.Set(1, 0, 0, 0);
            return number;
        }

        // parse minor version
        pos = verStr.Find('.');
        if (pos > 0) {
            minor = CharTraitsA::ParseInt(verStr.Substring(0, pos));
            verStr = verStr.Substring(pos + 1);

            // parse release number
            release = CharTraitsA::ParseInt(verStr);

        } else {
            minor = CharTraitsA::ParseInt(verStr);
        }

        number.Set(major, minor, release);

    }
    return number;
}


/*
 * vislib::graphics::gl::IsVSyncEnabled
 */
bool vislib::graphics::gl::IsVSyncEnabled(bool *error) {
#ifdef _WIN32
    static PFNWGLGETSWAPINTERVALEXTPROC wglGetSwapIntervalEXT = NULL;
    if (wglGetSwapIntervalEXT == NULL) {
        if (WGLExtensionSupported("WGL_EXT_swap_control") == 1) {
            wglGetSwapIntervalEXT = (PFNWGLGETSWAPINTERVALEXTPROC)
                wglGetProcAddress("wglGetSwapIntervalEXT");
        }
        if (wglGetSwapIntervalEXT == NULL) {
            if (error != NULL) *error = true;
            return false;
        }
    }
    if (error != NULL) *error = false;
    return (wglGetSwapIntervalEXT() == 1);
#else /* _WIN32 */
    // LINUX DOES NOT WORK AT ALL!!!
    //  GLX_SGI_swap_control cannot be disabled!
    //  __GL_SYNC_TO_VBLANK cannot be changed on runtime (at least not while context exists)
    //  GLX_MESA_fuckup is no longer supported
    //  nvidia-settings does not seem to have any effect

    if (error != NULL) *error = true;
    return false;
#endif /* _WIN32 */
}
