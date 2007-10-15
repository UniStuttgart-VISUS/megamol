/*
 * glfunctions.cpp
 *
 * Copyright (C) 2007 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#include "vislib/glfunctions.h"
#include <GL/gl.h>


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
