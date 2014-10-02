#ifndef INCLUDEALLGL_H_INCLUDED
#define INCLUDEALLGL_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */

#include "glload/include/glload/visgl_load.h"

#include "glload/include/glload/gl_all.h"
#include "glload/include/glload/gl_load.h"
#ifdef _WIN32
#include <Windows.h>
#include "glload/include/glload/wgl_all.h"
#include "glload/include/glload/wgl_load.h"
#else
#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include "glload/include/glload/glx_all.h"
#include "glload/include/glload/glx_load.h"
#endif

//#include <atomic>

namespace vislib {
namespace graphics {
namespace gl {

inline void LoadAllGL() {
    //static std::atomic<bool> alreadyLoaded(false);
    static bool alreadyLoaded = true;
    //bool expected = false;
    //if (alreadyLoaded.compare_exchange_strong(expected, true)) {
    if (alreadyLoaded == true) {
#ifdef _WIN32
        wgl_LoadFunctions(wglGetCurrentDC());
#else
        Display *display = XOpenDisplay(NULL);
        glx_LoadFunctions(display, DefaultScreen(display));
        XCloseDisplay(display);
#endif
        ogl_LoadFunctions();
    }
}

} /* end namespace gl */
} /* end namespace graphics */
} /* end namespace vislib */


#endif /* INCLUDEALLGL_H_INCLUDED */
