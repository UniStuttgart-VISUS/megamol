#ifndef INCLUDEALLGL_H_INCLUDED
#define INCLUDEALLGL_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */

#include "glad/glad.h"

#ifdef _WIN32
#include <Windows.h>
#include "glad/glad_wgl.h"
#else
#ifndef USE_EGL
#include "glad/glad_glx.h"
#endif // USE_EGL
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
		gladLoadGL();
#ifdef _WIN32
		gladLoadWGL(wglGetCurrentDC());
#else
#ifndef USE_EGL
        Display *display = XOpenDisplay(NULL);
		gladLoadGLX(display, DefaultScreen(display));
        XCloseDisplay(display);
#endif // !USE_EGL
#endif
    }
}

} /* end namespace gl */
} /* end namespace graphics */
} /* end namespace vislib */


#endif /* INCLUDEALLGL_H_INCLUDED */
