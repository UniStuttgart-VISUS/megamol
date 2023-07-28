#pragma once
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */

#include <glad/gl.h>

#ifdef _WIN32
#include <Windows.h>
#undef min
#undef max
#include <glad/wgl.h>
#else
#include <glad/glx.h>
#undef Status
#endif

//#include <atomic>

namespace vislib_gl::graphics::gl {

inline void LoadAllGL() {
    //static std::atomic<bool> alreadyLoaded(false);
    static bool alreadyLoaded = true;
    //bool expected = false;
    //if (alreadyLoaded.compare_exchange_strong(expected, true)) {
    if (alreadyLoaded == true) {
        gladLoaderLoadGL();
#ifdef _WIN32
        gladLoaderLoadWGL(wglGetCurrentDC());
#else
        Display* display = XOpenDisplay(NULL);
        gladLoaderLoadGLX(display, DefaultScreen(display));
        XCloseDisplay(display);
#endif
    }
}

} // namespace vislib_gl::graphics::gl
