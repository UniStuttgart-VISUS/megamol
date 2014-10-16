/*
 * MegaMolViewerStatic.h
 *
 * Copyright (C) 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLVIEWER_MEGALMOLVIEWERSTATIC_H_INCLUDED
#define MEGAMOLVIEWER_MEGALMOLVIEWERSTATIC_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#ifndef MEGAMOLVIEWER_MEGALMOLVIEWER_H_INCLUDED
#error You must not include MegaMolViewerStatic.h directly. \
Always include MegaMolViewer.h
#endif /* !MEGAMOLVIEWER_MEGALMOLVIEWER_H_INCLUDED */

#ifdef __cplusplus
extern "C" {
#endif

#ifndef MEGAMOLVIEWER_MEGALMOLVIEWERDYNAMIC_H_INCLUDED
#   ifdef _WIN32
/* defines to controll the export and import of functions */
#       ifdef MEGAMOLVIEWER_EXPORTS
#           define MEGAMOLVIEWER_API __declspec(dllexport)
#       else /* MEGAMOLVIEWER_EXPORTS */
#           define MEGAMOLVIEWER_API __declspec(dllimport)
#       endif /* MEGAMOLVIEWER_EXPORTS */
#       define MEGAMOLVIEWER_CALL(F) __cdecl F
#   else /* _WIN32 */
#       define MEGAMOLVIEWER_API
#       define MEGAMOLVIEWER_CALL(F) F
#   endif /* _WIN32 */
#endif /* !MEGAMOLVIEWER_MEGALMOLVIEWERDYNAMIC_H_INCLUDED */

/** Possible operating systems */
#ifndef MMCOSYSENUM_DEFINED
#define MMCOSYSENUM_DEFINED 1
typedef enum mmcOSysEnum {
    MMC_OSYSTEM_WINDOWS,
    MMC_OSYSTEM_LINUX,
    MMC_OSYSTEM_UNKNOWN
} mmcOSys;
#endif /* MMCOSYSENUM_DEFINED */

/** Possible hardware architectures */
#ifndef MMCHARCHENUM_DEFINED
#define MMCHARCHENUM_DEFINED 1
typedef enum mmcHArchEnum {
    MMC_HARCH_I86,
    MMC_HARCH_X64,
    MMC_HARCH_UNKNOWN
} mmcHArch;
#endif /* MMCHARCHENUM_DEFINED */

/** Possible view handle types */
typedef enum mmvHandleTypeEnum {
    MMV_HTYPE_INVALID,  // The handle is invalid or no handle at all.
    MMV_HTYPE_VIEWCOREINSTANCE, // View core Instance handle.
    MMV_HTYPE_GLWINDOWINSTANCE, // OpenGL window Instance handle.
    MMV_HTYPE_UNKNOWN   // The handle is a valid handle, but it is unknown of
} mmvHandleType;

/** callback parameter struct for the key callback */
typedef struct mmvKeyParamsStruct {
    unsigned short keycode; // uses a vislib key code
    bool modShift:1;
    bool modCtrl:1;
    bool modAlt:1;
    int mouseX;
    int mouseY;
} mmvKeyParams;

/** callback parameter struct for the mouse button callback */
typedef struct mmvMouseButtonParamsStruct {
    unsigned char button;
    bool buttonDown:1;
    bool modShift:1;
    bool modCtrl:1;
    bool modAlt:1;
    int mouseX;
    int mouseY;
} mmvMouseButtonParams;

/** callback parameter struct for the mouse move callback */
typedef struct mmvMouseMoveParamsStruct {
    bool modShift:1;
    bool modCtrl:1;
    bool modAlt:1;
    int mouseX;
    int mouseY;
} mmvMouseMoveParams;

/** Flags for view hints */
#define MMV_VIEWHINT_NONE           0x00000000
#define MMV_VIEWHINT_QUADBUFFER     0x00000001
#define MMV_VIEWHINT_ALPHABUFFER    0x00000002

/** Flags for window hints */
#define MMV_WINHINT_NONE            0x00000000
#define MMV_WINHINT_NODECORATIONS   0x00000001
#define MMV_WINHINT_HIDECURSOR      0x00000002
#define MMV_WINHINT_STAYONTOP       0x00000004
#define MMV_WINHINT_PRESENTATION    0x00000008
#define MMV_WINHINT_VSYNC           0x00000010
#define MMV_WINHINT_PARAMGUI        0x00000020

/**
 * the callback function syntax
 *
 * @param userData The userData associated with the object who called this
 *                 callback.
 * @param params Pointer to the memory holding the parameters if available,
 *               otherwise 'NULL'.
 */
typedef void (*mmvCallback)(void *userData, void *params);

/** Possible window callback slots */
typedef enum mmvWindowCallbacksEnum {
    MMV_WINCB_RENDER,       // param: bool (receiving whether or not to do continuous redrawing)
    MMV_WINCB_RESIZE,       // param: 2x unsigned int (new window size)
    MMV_WINCB_KEY,          // param: mmvKeyParamsStruct
    MMV_WINCB_MOUSEBUTTON,  // param: mmvMouseButtonParams
    MMV_WINCB_MOUSEMOVE,    // param: mmvMouseButtonParams
    MMV_WINCB_CLOSE,        // param: NULL
    MMV_WINCB_COMMAND,      // param: int (the command id)
    MMV_WINCB_APPEXIT,      // param: NULL
    MMV_WINCB_UPDATEFREEZE  // param: int (0 unfreeze, 1 freeze/update)
} mmvWindowCallbacks;

/** Library building flags */
#define MMV_BFLAG_DEBUG     0x00000001  // debug build
#define MMV_BFLAG_DIRTY     0x00000002  // dirty build (DO NOT RELEASE!)

/**
 * Returns the version of the MegaMolCore.
 *
 * Applications should check this version directly after startup to ensure
 * compatibility with this core.
 *
 * @param outMajorVersion Pointer to a short word receiving the major version
 * @param outMinorVersion Pointer to a short word receiving the minor version
 * @param outMajorRevision Pointer to a short word receiving the major revision
 * @param outMinorRevision Pointer to a short word receiving the minor revision
 * @param outSys Pointer to a mmcOSys variable receiving the system type of 
 *               the core.
 * @param outArch Pointer to a mmcHArchEnum variable receiving the 
 *                architecture of the core.
 * @param outFlags Pointer to a unsigned int receiving build flags MMV_BFLAG_*
 * @param outNameStr Pointer to a ANSI-character buffer receiving the name of
 *                   the library. Copies the whole name, but at least
 *                   'inOutNameSize' bytes and returns the number of copied
 *                   bytes in 'inOutNameSize'. 'inOutNameSize' must not be
 *                   NULL if 'outNameStr' is not NULL.
 * @param inOutNameSize Pointer to an integer defining the size of the buffer
 *                      'outNameStr' points to. If 'outNameStr' is NULL and
 *                      'inOutNameSize' is not NULL, 'inOutNameSize' receives
 *                      the number of characters (excluding the terminating
 *                      zero) required to store the name of the library.
 * @param outCommentStr Pointer to a ANSI-character buffer receiving the name
 *                      of the library. Copies the whole name, but at least
 *                      'inOutCommentSize' bytes and returns the number of
 *                      copied bytes in 'inOutCommentSize'. 'inOutCommentSize'
 *                      must not be NULL if 'outCommentStr' is not NULL.
 * @param inOutCommentSize Pointer to an integer defining the size of the
 *                         buffer 'outCommentStr' points to. If
 *                         'outCommentStr' is NULL and 'inOutCommentSize' is
 *                         not NULL, 'inOutCommentSize' receives the number of
 *                         characters (excluding the terminating zero)
 *                         required to store the name of the library.
 */
MEGAMOLVIEWER_API void MEGAMOLVIEWER_CALL(mmvGetVersionInfo)(
    unsigned short *outMajorVersion, unsigned short *outMinorVersion,
    unsigned short *outMajorRevision, unsigned short *outMinorRevision,
    mmcOSys *outSys, mmcHArch *outArch, unsigned int *outFlags,
    char *outNameStr, unsigned int *inOutNameSize,
    char *outCommentStr, unsigned int *inOutCommentSize);

/**
 * Returns the size needed to store a viewer handle.
 *
 * @return the size in bytes for the a viewer handle.
 */
MEGAMOLVIEWER_API unsigned int MEGAMOLVIEWER_CALL(mmvGetHandleSize)(void);

/**
 * Disposes a viewer handle. The handle will be invalid after this call.
 * Since this method cannot fail, it is save to call it with an invalid handle.
 * In this case no operation is performed.
 *
 * Note that handles to dependen object become implicitly invalid. Using such
 * an handle will result in undefined behaviour.
 *
 * @param hndl The viewer handle to be disposed.
 */
MEGAMOLVIEWER_API void MEGAMOLVIEWER_CALL(mmvDisposeHandle)(void *hndl);


/**
 * Checks wether the supplied handle is a valid mega mol viewer handle or not.
 * If you encounter problems with a handle and this functions returns true,
 * the handle may still be from a wrong handle type.
 *
 * @param hndl The handle to be tested.
 *
 * @return 'true' if the handle is a valid mega mol viewer handle,
 *         otherwise 'false'.
 */
MEGAMOLVIEWER_API bool MEGAMOLVIEWER_CALL(mmvIsHandleValid)(void *hndl);

/**
 * Answers the type of the supplied handle.
 *
 * @param hndl The handle to be tested.
 *
 * @return The type of the specified handle.
 */
MEGAMOLVIEWER_API mmvHandleType MEGAMOLVIEWER_CALL(mmvGetHandleType)(
    void *hndl);

/**
 * Creates a viewer instance and places the handle into the specified memory.
 *
 * The caller is responsible that 'hView' points to an allocated memory block
 * of sufficient size. The size must be determined by calling 
 * 'mmvGetHandleSize'. The caller remains owner of the memory and must ensure
 * that the memory is not freed or moved until 'mmvDisposeViewerHandle' has 
 * returned.The first byte of the memory block speified by 'hView' must be 
 * set to zero. If 'hView' point to a memory block already holding a valid 
 * view handle the method fails.
 *
 * Warning: DO NOT CHANGE the data 'hView' points to, as long as a valid core
 * handle is placed there.
 *
 * @param hView Points to the memory receiving the view instance handle.
 * @param hints The viewer initialization hints
 *
 * @return 'true' on success, 'false' otherwise.
 */
MEGAMOLVIEWER_API bool MEGAMOLVIEWER_CALL(mmvCreateViewerHandle)(void *hView,
    unsigned int hints);

/**
 * Initializes the vislib stack trace.
 *
 * @param stm Pointer to the vislib stack trace manager
 */
MEGAMOLVIEWER_API void MEGAMOLVIEWER_CALL(mmvInitStackTracer)(void *stm);

/**
 * This method must be called periodically. It triggeres the event processing
 * of the window message loop.
 *
 * @param hView The view instance handle.
 *
 * @return 'true' if the message loop is still active, 'false' if the loop
 *         terminated.
 */
MEGAMOLVIEWER_API bool MEGAMOLVIEWER_CALL(mmvProcessEvents)(void *hView);

/**
 * Creates a new OpenGL rendering window and a new OpenGL context. This also
 * activates the new context.
 *
 * @param hView The view instance handle.
 * @param hWnd Points to the memory receiving the window handle.
 *
 * @return 'true' on success, 'false' otherwise.
 */
MEGAMOLVIEWER_API bool MEGAMOLVIEWER_CALL(mmvCreateWindow)(void *hView,
    void *hWnd);

/**
 * Sets the user data pointer of the object of the handle 'hndl' to
 * 'userData'. The previous pointer will simply be overwritten. No further
 * action (e.g. 'delete' or 'free') is performed. Note that user data is
 * associated with objects and not with handles!
 *
 * @param hndl The handle.
 * @param userDate The new user data pointer.
 */
MEGAMOLVIEWER_API void MEGAMOLVIEWER_CALL(mmvSetUserData)(void *hndl,
    void *userData);

/**
 * Gets the user data pointer of the objec of the handle 'hndl'.
 *
 * @param hndl The handle.
 *
 * @return The user data pointer.
 */
MEGAMOLVIEWER_API void * MEGAMOLVIEWER_CALL(mmvGetUserData)(void *hndl);

/**
 * Registers a callback function for the given callback slot of the specified
 * window.
 *
 * @param hWnd The window to register the callback on.
 * @param slot The callback slot to register the callback on.
 * @param function The callback function to be registered.
 */
MEGAMOLVIEWER_API void MEGAMOLVIEWER_CALL(mmvRegisterWindowCallback)(
    void *hWnd, mmvWindowCallbacks slot, mmvCallback function);

/**
 * Unregisters a callback function from the given callback slot of the
 * specified window.
 *
 * @param hWnd The window to register the callback on.
 * @param slot The callback slot to register the callback on.
 * @param function The callback function to be registered.
 */
MEGAMOLVIEWER_API void MEGAMOLVIEWER_CALL(mmvUnregisterWindowCallback)(
    void *hWnd, mmvWindowCallbacks slot, mmvCallback function);

/**
 * Answer if this window supports a context menu
 *
 * @param hWnd The window handle
 *
 * @return 'true' if a context menu is supported, 'false' otherwise
 */
MEGAMOLVIEWER_API bool MEGAMOLVIEWER_CALL(mmvSupportContextMenu)(void *hWnd);

/**
 * Installs a context menu on the given window.
 *
 * @param hWnd The window handle.
 *
 * @return 'true' on success, 'false' on failure.
 */
MEGAMOLVIEWER_API bool MEGAMOLVIEWER_CALL(mmvInstallContextMenu)(void *hWnd);

/**
 * Installs a command to the context menu. This function fails silently if
 * the context menu has not been installed.
 *
 * @param hWnd The window handle.
 * @param caption The caption for the command menu item.
 * @param value The value of the command.
 */
#if defined(UNICODE) || defined(_UNICODE)
#define mmvInstallContextMenuCommand mmvInstallContextMenuCommandW
#else /* defined(UNICODE) || defined(_UNICODE) */
#define mmvInstallContextMenuCommand mmvInstallContextMenuCommandA
#endif /* defined(UNICODE) || defined(_UNICODE) */

/*
 * ANSI implementation of mmvInstallContextMenuCommand
 * @see mmvInstallContextMenuCommand
 */
MEGAMOLVIEWER_API void MEGAMOLVIEWER_CALL(mmvInstallContextMenuCommandA)(
    void *hWnd, const char *caption, int value);

/*
 * Unicode implementation of mmvInstallContextMenuCommand
 * @see mmvInstallContextMenuCommand
 */
MEGAMOLVIEWER_API void MEGAMOLVIEWER_CALL(mmvInstallContextMenuCommandW)(
    void *hWnd, const wchar_t *caption, int value);

/**
 * Answer if this window supports a parameter gui
 *
 * @param hWnd The window handle
 *
 * @return 'true' if a parameter gui is supported, 'false' otherwise
 */
MEGAMOLVIEWER_API bool MEGAMOLVIEWER_CALL(mmvSupportParameterGUI)(void *hWnd);

/**
 * Installs a parameter gui on the given window.
 *
 * @param hWnd The window handle
 *
 * @return 'true' on success, 'false' on failure.
 */
MEGAMOLVIEWER_API bool MEGAMOLVIEWER_CALL(mmvInstallParameterGUI)(void *hWnd);

/**
 * Installs a parameter into the parameter gui of the given window.
 * 'mmvInstallParameterGUI' must have been called successfully before.
 *
 * @param hWnd The window handle
 * @param paramID A void pointer with a unique value identifying the
 *                parameter. The pointers value will only be used for
 *                identification and will never be dereferenced.
 * @param desc A MegaMol parameter description
 * @param len The length of the MegaMol parameter description in bytes
 *
 * @return 'true' on success, 'false' on failure.
 */
MEGAMOLVIEWER_API bool MEGAMOLVIEWER_CALL(mmvInstallParameter)(void *hWnd,
    void *paramID, const unsigned char *desc, unsigned int len);

/**
 * Removes a parameter from the parameter gui of the given window. Fails
 * silently on any error.
 *
 * @param hWnd The window handle
 * @param paramID The parameter id void pointer
 */
MEGAMOLVIEWER_API void MEGAMOLVIEWER_CALL(mmvRemoveParameter)(void *hWnd,
    void *paramID);

/**
 * Sets the value of a parameter in the parameter gui. The parameter must have
 * been successfully installed before.
 *
 * @param hWnd The window handle
 * @param paramID The parameter id void pointer
 * @param value A zero-terminated string representation of the parameter
 *
 * @return 'true' on success, 'false' on failure.
 */
#if defined(UNICODE) || defined(_UNICODE)
#define mmvSetParameterValue mmvSetParameterValueW
#else /* defined(UNICODE) || defined(_UNICODE) */
#define mmvSetParameterValue mmvSetParameterValueA
#endif /* defined(UNICODE) || defined(_UNICODE) */

/*
 * ANSI implementation of mmvSetParameterValue
 * @see mmvSetParameterValue
 */
MEGAMOLVIEWER_API bool MEGAMOLVIEWER_CALL(mmvSetParameterValueA)(void *hWnd,
    void *paramID, const char *value);

/*
 * Unicode implementation of mmvSetParameterValue
 * @see mmvSetParameterValue
 */
MEGAMOLVIEWER_API bool MEGAMOLVIEWER_CALL(mmvSetParameterValueW)(void *hWnd,
    void *paramID, const wchar_t *value);

/**
 * Registers a callback function for the given parameter of the parameter gui
 * of the specified window. The callback will be called with 'paramID' in
 * 'userData' and an ANSI string representation of the new parameter value
 * in 'param' whenever the value of the parameter is changed in the gui.
 *
 * @param hWnd The window to register the callback on.
 * @param paramID The parameter id void pointer
 * @param function The callback function to be registered.
 *
 * @return 'true' on success, 'false' on failure.
 */
MEGAMOLVIEWER_API bool MEGAMOLVIEWER_CALL(mmvRegisterParameterCallback)(
    void *hWnd, void *paramID, mmvCallback function);

/**
 * Unregisters a callback function from the given parameter of the parameter
 * gui of the specified window.
 *
 * @param hWnd The window handle
 * @param paramID The parameter id void pointer
 */
MEGAMOLVIEWER_API void MEGAMOLVIEWER_CALL(mmvUnregisterParameterCallback)(
    void *hWnd, void *paramID);

/**
 * Sets the size of the given window.
 *
 * @param hWnd The window handle.
 * @param width The new width.
 * @param height The new height.
 */
MEGAMOLVIEWER_API void MEGAMOLVIEWER_CALL(mmvSetWindowSize)(void *hWnd,
    unsigned int width, unsigned int height);

/**
 * Sets the title of a view window.
 *
 * @param hWnd The window handle.
 * @param title The new window title.
 */
#if defined(UNICODE) || defined(_UNICODE)
#define mmvSetWindowTitle mmvSetWindowTitleW
#else /* defined(UNICODE) || defined(_UNICODE) */
#define mmvSetWindowTitle mmvSetWindowTitleA
#endif /* defined(UNICODE) || defined(_UNICODE) */

/*
 * ANSI implementation of mmvSetWindowTitle
 * @see mmvSetWindowTitle
 */
MEGAMOLVIEWER_API void MEGAMOLVIEWER_CALL(mmvSetWindowTitleA)(
    void *hWnd, const char *title);

/*
 * Unicode implementation of mmvSetWindowTitle
 * @see mmvSetWindowTitle
 */
MEGAMOLVIEWER_API void MEGAMOLVIEWER_CALL(mmvSetWindowTitleW)(
    void *hWnd, const wchar_t *title);

/**
 * Sets the position of the given window.
 *
 * @param hWnd The window handle.
 * @param x The new x coordinate.
 * @param y The new y coordinate.
 */
MEGAMOLVIEWER_API void MEGAMOLVIEWER_CALL(mmvSetWindowPosition)(void *hWnd,
    int x, int y);

/**
 * Switches the given window to full screen mode.
 *
 * @param hWnd The window handle.
 */
MEGAMOLVIEWER_API void MEGAMOLVIEWER_CALL(mmvSetWindowFullscreen)(void *hWnd);

/**
 * Sets window hints.
 * A flag set in the mask will set a new value based on the value of hints
 *
 * @param hWnd The window handle
 * @param mask The set hints mask
 * @param hints The hints
 */
MEGAMOLVIEWER_API void MEGAMOLVIEWER_CALL(mmvSetWindowHints)(void *hWnd,
    unsigned int mask, unsigned int hints);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* MEGAMOLVIEWER_MEGALMOLVIEWERSTATIC_H_INCLUDED */
