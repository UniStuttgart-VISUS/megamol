#include "glload/visgl_load.h"

#include "glload/gl_all.h"
#include "glload/gl_load.h"
#ifdef _WIN32
#include <Windows.h>
#include "glload/wgl_all.h"
#include "glload/wgl_load.h"
#else
#include "glload/glx_all.h"
#include "glload/glx_load.h"
#endif
#include <cstring>
#include <cassert>


GLLOADAPI int isExtAvailable(const char *extensionName) {
    // Using extension strings of the form "GL_VERSION_x_y" is no longer supported!
    // Use 'ogl_IsVersionGEQ(x, y)' instead
    assert(::memcmp(extensionName, "GL_VERSION_", 11) != 0);

    int res = FindGLExt(extensionName);
    if (res == ogl_LOAD_SUCCEEDED) {
        return GL_TRUE;
    } else {
#ifdef _WIN32
        res = FindWGLExt(extensionName);
        if (res == wgl_LOAD_SUCCEEDED) {
            return GL_TRUE;
        }
#else
        res = FindGLXExt(extensionName);
        if (res == glx_LOAD_SUCCEEDED) {
            return GL_TRUE;
        }
#endif
    }
    return GL_FALSE;
}


GLLOADAPI int areExtsAvailable(const char *extensionsNames) {
    size_t iExtListLen = ::strlen(extensionsNames);
    const char *strExtListEnd = extensionsNames + iExtListLen;
    const char *strCurrPos = extensionsNames;
    const int strWorkBuffLen = 256;
    char strWorkBuff[strWorkBuffLen];

    while(*strCurrPos) {
        /*Get the extension at our position.*/
        int iStrLen = 0;
        const char *strEndStr = strchr(strCurrPos, ' ');
        int iStop = 0;
        if (strEndStr == NULL) {
            strEndStr = strExtListEnd;
            iStop = 1;
        }

        iStrLen = (int)((ptrdiff_t)strEndStr - (ptrdiff_t)strCurrPos);

        if (iStrLen > strWorkBuffLen - 1) return GL_FALSE;

        strncpy(strWorkBuff, strCurrPos, iStrLen);
        strWorkBuff[iStrLen] = '\0';

        if (isExtAvailable(strWorkBuff) == GL_FALSE) return GL_FALSE;

        strCurrPos = strEndStr + 1;
        if (iStop) break;
    }

    return GL_TRUE;
}
