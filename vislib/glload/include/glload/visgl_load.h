#ifndef VISGL_LOAD_H_INCLUDED
#define VISGL_LOAD_H_INCLUDED
#pragma once
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#include "_int_gl_load_ie.h"

#ifdef __cplusplus
extern "C" {
#endif

    /**
     * Tests if an OpenGL extension is available, testing gl, wgl and glx.
     *
     * @param extensionName The name of the extension to test
     *
     * @return GL_TRUE if the extension is available. GL_FALSE otherwise
     */
    GLLOADAPI int isExtAvailable(const char *extensionName);

    /**
     * Tests if multiple OpenGL extensions are available.
     *
     * @param extensionsNames The names of all extensions separated by space
     *                        characters.
     *
     * @return GL_TRUE if all extensions are available
     */
    GLLOADAPI int areExtsAvailable(const char *extensionsNames);

#ifdef __cplusplus
}
#endif

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISGL_LOAD_H_INCLUDED */
