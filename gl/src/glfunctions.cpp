/*
 * glfunctions.cpp
 *
 * Copyright (C) 2007 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#include "vislib/glfunctions.h"
#include <GL/gl.h>

#ifdef _WIN32
// TODO: change or check implementation
#include "GL/wglext.h"

bool WGLExtensionSupported(const char *extension_name) {
    // this is pointer to function which returns pointer to string with list of all wgl extensions
    PFNWGLGETEXTENSIONSSTRINGEXTPROC _wglGetExtensionsStringEXT = NULL;

    // determine pointer to wglGetExtensionsStringEXT function
    _wglGetExtensionsStringEXT = (PFNWGLGETEXTENSIONSSTRINGEXTPROC) wglGetProcAddress("wglGetExtensionsStringEXT");

    if (strstr(_wglGetExtensionsStringEXT(), extension_name) == NULL) {
        // string was not found
        return false;
    }

    // extension is supported
    return true;
}
#endif


/*
 * vislib::graphics::gl::EnableVSync
 */
void vislib::graphics::gl::EnableVSync(bool enable) {
#ifdef _WIN32
    static PFNWGLSWAPINTERVALEXTPROC wglSwapIntervalEXT = NULL;
    static bool initialised = false;
    if (!initialised) {
        initialised = true;
        if (WGLExtensionSupported("WGL_EXT_swap_control")) {
            // this is another function from WGL_EXT_swap_control extension
            wglSwapIntervalEXT = (PFNWGLSWAPINTERVALEXTPROC)wglGetProcAddress("wglSwapIntervalEXT");
        }
    }
    if (wglSwapIntervalEXT != NULL) {
        wglSwapIntervalEXT(enable ? 1 : 0);
    }
#else /* _WIN32 */
    // TODO: Implement
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
bool vislib::graphics::gl::IsVSyncEnabled(void) {
#ifdef _WIN32
    static PFNWGLGETSWAPINTERVALEXTPROC wglGetSwapIntervalEXT = NULL;
    static bool initialised = false;
    if (!initialised) {
        initialised = true;
        if (WGLExtensionSupported("WGL_EXT_swap_control")) {
            // this is another function from WGL_EXT_swap_control extension
            wglGetSwapIntervalEXT = (PFNWGLGETSWAPINTERVALEXTPROC)wglGetProcAddress("wglGetSwapIntervalEXT");
        }
    }
    return (wglGetSwapIntervalEXT != NULL) ? (wglGetSwapIntervalEXT() == 1) : false;
#else /* _WIN32 */
    // TODO: Implement
    return false;
#endif /* _WIN32 */
}
