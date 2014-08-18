/*
 * MegaMolCore.h
 *
 * Copyright (C) 2006 - 2010 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_MEGALMOLCORE_H_INCLUDED
#define MEGAMOLCORE_MEGALMOLCORE_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

/*
 * The MegaMolCore API is exported using ansi-c functions only, to allow the
 * usage of the core with non c/c++ frontends.
 *
 * This header specifies the API functions and can be used as reference for 
 * other headers, importing the functions by dynamic loading the library or
 * for headers of other programming languages.
 *
 * The naming convention is that all MegaMolCore function names start with
 * "mmc" (e.g. mmcInit).
 *
 * This file must not do anything but declaring the functions exported by the
 * core. No implementations (inline functions) must be placed here.
 */

#ifdef __cplusplus
extern "C" {
#endif


/*
 * MegaMolCore.std.h
 *
 * Copyright (C) 2009 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_STD_H_INCLUDED
#define MEGAMOLCORE_STD_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

//
// CONTENT OF THIS FILE IS GENERATED
// DO NOT EDIT
//


/**
 * Integer value of the core revision used for compatiblity test between core
 * and plugins. Last digit shows the dirty flag and should be zero for all
 * production releases.
 */
#define MEGAMOL_CORE_COMP_REV 2441


/////**
//// * Struct to be returned by a plugin api holding compatibility informations
//// */
////typedef struct _mmplg_compatibilityvalues_t {
////    SIZE_T size; // number of bytes of this struct
////    SIZE_T mmcoreRev; // MegaMol™ core revision
////    SIZE_T vislibRev; // VISlib revision or zero if no vislib is used
////} mmplgCompatibilityValues;


/**
 * defines to control the export and import of functions
 */
#ifdef _WIN32
#   ifdef MEGAMOLCORE_EXPORTS
#       define MEGAMOLCORE_API __declspec(dllexport)
#   else /* MEGAMOLCORE_EXPORTS */
#       define MEGAMOLCORE_API __declspec(dllimport)
#   endif /* MEGAMOLCORE_EXPORTS */
#   define MEGAMOLCORE_CALLBACK __stdcall
#   define MEGAMOLCORE_CALL(R, F) MEGAMOLCORE_API R MEGAMOLCORE_CALLBACK F
#else /* _WIN32 */
#   define MEGAMOLCORE_API
#   define MEGAMOLCORE_CALL(R, F)
#   define MEGAMOLCORE_CALLBACK
#endif /* _WIN32 */


#endif /* MEGAMOLCORE_STD_H_INCLUDED */



/*****************************************************************************/
/** TYPES */

/** Possible operating systems */
#ifndef MMCOSYSENUM_DEFINED
#define MMCOSYSENUM_DEFINED 1
typedef enum _mmcOSysEnum {
    MMC_OSYSTEM_WINDOWS,
    MMC_OSYSTEM_LINUX,
    MMC_OSYSTEM_UNKNOWN
} mmcOSys;
#endif /* MMCOSYSENUM_DEFINED */

/** Possible hardware architectures */
#ifndef MMCHARCHENUM_DEFINED
#define MMCHARCHENUM_DEFINED 1
typedef enum _mmcHArchEnum {
    MMC_HARCH_I86,
    MMC_HARCH_X64,
    MMC_HARCH_UNKNOWN
} mmcHArch;
#endif /* MMCHARCHENUM_DEFINED */

/** Possible handle types */
typedef enum _mmcHandleTypeEnum {
    MMC_HTYPE_INVALID, // The handle is invalid or no handle at all.
                       // which type.
    MMC_HTYPE_COREINSTANCE, // Core Instance handle.
    MMC_HTYPE_NAMESPACE, // module namespace
    MMC_HTYPE_RENDERINGOUTPUTMODULE, // a rendering output window module
    MMC_HTYPE_THREADMODULE, // a job thread module
    MMC_HTYPE_MODULE, // a module
    MMC_HTYPE_CALL, // a call
    MMC_HTYPE_CALLER, // a call
    MMC_HTYPE_CALLEE, // a call
    MMC_HTYPE_PARAMETER, // a parameter
    MMC_HTYPE_VALUE, // a value object
    MMC_HTYPE_UNKNOWN // The handle is a valid handle, but it is unknown of
} mmcHandleType;

/** Possible error codes */
typedef enum _mmcErrorCodeEnum {
    MMC_ERR_NO_ERROR = 0, // No Error. This denotes success.
    MMC_ERR_MEMORY, // Generic memory error.
    MMC_ERR_HANDLE, // Generic handle error.
    MMC_ERR_INVALID_HANDLE, // The handle specified was invalid.
    MMC_ERR_STATE, // The object was in a incompatible state.
    MMC_ERR_TYPE, // Generic type error (normally incompatible type or cast 
                  // failed).
    MMC_ERR_NOT_IMPLEMENTED, // Function not implemented.
    MMC_ERR_LICENSING, // Requested action not possible due to licensing
    MMC_ERR_PARAM, // A function parameter was illegal
    MMC_ERR_NOT_FOUND, // The element was not found
    MMC_ERR_HANDLE_TYPE, // A specified handle was of a wrong type
    MMC_ERR_UNKNOWN // Unknown error.
} mmcErrorCode;

/** Possible value types. */
typedef enum _mmcValueTypeEnum {
    MMC_TYPE_INT32, // 32 bit signed integer
    MMC_TYPE_UINT32, // 32 bit unsigned integer
    MMC_TYPE_INT64, // 64 bit signed integer
    MMC_TYPE_UINT64, // 64 bit unsigned integer
    MMC_TYPE_BYTE, // 8 bit unsigned integer
    MMC_TYPE_BOOL, // bool (platform specific integer size)
    MMC_TYPE_FLOAT, // 32 bit float
    MMC_TYPE_DOUBLE, // 64 bit float
    MMC_TYPE_CSTR, // Ansi string (Pointer or Array of ansi characters).
    MMC_TYPE_WSTR, // Unicode string (Pointer or Array of wide characters).
#if defined(UNICODE) || defined(_UNICODE)
#define MMC_TYPE_TSTR MMC_TYPE_WSTR
#else /* defined(UNICODE) || defined(_UNICODE) */
#define MMC_TYPE_TSTR MMC_TYPE_CSTR
#endif /* defined(UNICODE) || defined(_UNICODE) */
    MMC_TYPE_VOIDP, // Manuel type convertion. Use with care!
    MMC_TYPE_INT32ARRAY, // array of 32 bit signed integer
    MMC_TYPE_UINT32ARRAY, // array of 32 bit unsigned integer
    MMC_TYPE_INT64ARRAY, // array of 64 bit signed integer
    MMC_TYPE_UINT64ARRAY, // array of 64 bit unsigned integer
    MMC_TYPE_BYTEARRAY, // array of 8 bit unsigned integer
    MMC_TYPE_BOOLARRAY, // array of bool (platform specific integer size)
    MMC_TYPE_FLOATARRAY, // array of 32 bit float
    MMC_TYPE_DOUBLEARRAY, // array of 64 bit float
    MMC_TYPE_CSTRARRAY, // array of Ansi string (Pointer of Pointer or Array of ansi characters).
    MMC_TYPE_WSTRARRAY, // array of Unicode string (Pointer of Pointer or Array of wide characters).
#if defined(UNICODE) || defined(_UNICODE)
#define MMC_TYPE_TSTRARRAY MMC_TYPE_WSTRARRAY
#else /* defined(UNICODE) || defined(_UNICODE) */
#define MMC_TYPE_TSTRARRAY MMC_TYPE_CSTRARRAY
#endif /* defined(UNICODE) || defined(_UNICODE) */
    MMC_TYPE_INVALID // value has invalid content
} mmcValueType;

/** Library building flags */
#define MMC_BFLAG_DEBUG     0x00000001  // debug build
#define MMC_BFLAG_DIRTY     0x00000002  // dirty build (DO NOT RELEASE!)

/**
 * Function pointer type for callbacks
 *
 * @param hndl The handle of the calling object. The callback function should
 *             release the handle as soon as it is no longer used.
 * @param context The user specified pointer
 * @param reason The reason why the callback was called
 * @param data The data for the callback
 */
typedef void (MEGAMOLCORE_CALLBACK *mmcCoreCallback)(void *hndl, void *context, unsigned int reason, void *data);

/*****************************************************************************/
/** STRUCTS */

/** Data struct used when requesting an OpenGL window from the front end */
typedef struct _mmc_feres_openglwindowdata_t {

    /** Handle of the module requesting the resource */
    const void *modHandle;

    /** Will be set by the frontend to 'true' if the resource was allocated */
    bool success;

    /** Requests/acknowledges quad-buffer rendering context */
    bool quadBuffer;

    /** Flag if 'winX' and 'winY' are valid */
    bool winPosValid;

    /** Flag if 'winWidth' and 'winHeight' are valid */
    bool winSizeValid;

    /** Flag if 'winDec' is valid */
    bool winDecValid;

    /** The left position for the window */
    int winX;

    /** The top position for the window */
    int winY;

    /** The width of the client area of the window */
    int winWidth;

    /** The height of the client area of the window */
    int winHeight;

    /** Shows(true)/Hides(false) the window decorations */
    bool winDec;

    /** Caption for the window or NULL */
    const char *caption;

    /** Flag whether or not to start in fullscreen */
    bool winFullscreen;

} MMCFEResOpenGLWindowData;

/*****************************************************************************/
/** DEFINES */

/** Possible front end resource request reasons */
#define MMC_FERES_OPENGLWINDOW  0x00000001  // OpenGL rendering windows

/** Utility macro for verifying success of core functions */
#define MMC_USING_VERIFY mmcErrorCode __mmc_verify_error__;

/** Utility macro for verifying success of core functions */
#define MMC_VERIFY_THROW(call) \
    if ((__mmc_verify_error__ = call) != MMC_ERR_NO_ERROR) {\
        vislib::StringA str; \
        str.Format("MegaMolCore Error %d", __mmc_verify_error__); \
        throw vislib::Exception(str, __FILE__, __LINE__);\
    }


/*****************************************************************************/


#if defined(MEGAMOL_USE_STATIC_CORE) || defined(MEGAMOLCORE_EXPORTS)
/*
 * MegaMolCoreStatic.h
 *
 * Copyright (C) 2010 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_MEGALMOLCORESTATIC_H_INCLUDED
#define MEGAMOLCORE_MEGALMOLCORESTATIC_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */


/*
 * Include Check
 */
#ifndef MEGAMOLCORE_MEGALMOLCORE_H_INCLUDED
#error DO NOT Include "MegaMolCoreStatic.h" directly! Include "MegaMolCore.h" instead.
#endif /* MEGAMOLCORE_MEGALMOLCORE_H_INCLUDED */


#ifdef __cplusplus
extern "C" {
#endif

/*****************************************************************************/
/** FUNCTIONS */

/**
 * Adds a callback to an object
 *
 * @param hndl Handle to the object
 * @param name The function name to add the callback for
 * @param callback The callback pointer
 * @param context The user defined context pointer, which will be passed on to
 *                The callback function
 *
 * @return The error code of the method
 */
MEGAMOLCORE_CALL(mmcErrorCode, mmcAddCallbackA)(void *hndl, const char *name, mmcCoreCallback callback, void *context);

/**
 * Adds a callback to an object
 *
 * @param hndl Handle to the object
 * @param name The function name to add the callback for
 * @param callback The callback pointer
 * @param context The user defined context pointer, which will be passed on to
 *                The callback function
 *
 * @return The error code of the method
 */
MEGAMOLCORE_CALL(mmcErrorCode, mmcAddCallbackW)(void *hndl, const wchar_t *name, mmcCoreCallback callback, void *context);

/**
 * Calls an object
 *
 * @param hndl The handle to the object to be called
 * @param callID The identification number of the call
 * @param param1 The first parameter
 * @param param2 The second parameter
 *
 * @return The error code of the method
 */
MEGAMOLCORE_CALL(mmcErrorCode, mmcCall)(void *hndl, unsigned int callID, void *param1, void *param2);

/**
 * removes all modules and calls from the core instance
 *
 * @param hCore The core instance handle
 *
 * @return The error code of the method
 */
MEGAMOLCORE_CALL(mmcErrorCode, mmcClear)(void *hCore);

/**
 * Tests if two handle point to the same object.
 *
 * @param hndl1 The first handle
 * @param hndl2 The second handle
 *
 * @return True if both handles are valid and reference the same object, false
 *         otherwise.
 */
MEGAMOLCORE_CALL(bool, mmcCompareHandles)(void *hndl1, void *hndl2);

/**
 * Connects static vislib objects from front-end to core
 *
 * @param id The id of the object to connect
 * @param obj The front-end object to connect
 */
MEGAMOLCORE_CALL(void, mmcConnectVLStatics)(int id, void *obj);

/**
 * Converts the type of a value type object
 *
 * @param hndl The handle to the value object to convert it's type
 * @param newType The new type for the value object
 * @param dryryn If set to true the object of 'hndl' will not be altered, but
 *               the return value of the function will be calculated correctly
 *
 * @return A value indicating the success of the conversion
 *         0 Conversion was not possible (You cannot convert array types to
 *           non-array types, nor can you convert form or to
 *           'MMC_TYPE_INVALID')
 *         1 Conversion was successful without altering the value, i.e. a
 *           conversion back to the original type will result in exactly the
 *           same value.
 *         2 Conversion was done but was possible altering the value, i.e.
 *           truncating or clipping to a value representable in the new type
 *           (e.g. rounding a double to float or truncating an int32 to int64)
 *         3 Conversion was done but has altered the value for sure. The new
 *           value can be considdered the value representable in the new type
 *           closest to the original value.
 */
MEGAMOLCORE_CALL(int, mmcConvertValueType)(void *hndl, mmcValueType newType, bool dryrun);

/**
 * Creates a new call object
 *
 * @param hCode Handle to the core instance
 * @param hndl If not NULL, points to the memory to receive a handle to the
 *             newly created call object
 * @param type The type name for the new call object
 * @param hFrom Handle to the caller slot to be connected
 * @param hTo Handle to the callee slot to be connected
 *
 * @return The error code of the method
 */
MEGAMOLCORE_CALL(mmcErrorCode, mmcCreateCallA)(void *hCore, void *hndl, const char *type, void *hFrom, void *hTo);

/**
 * Creates a new call object
 *
 * @param hCode Handle to the core instance
 * @param hndl If not NULL, points to the memory to receive a handle to the
 *             newly created call object
 * @param type The type name for the new call object
 * @param hFrom Handle to the caller slot to be connected
 * @param hTo Handle to the callee slot to be connected
 *
 * @return The error code of the method
 */
MEGAMOLCORE_CALL(mmcErrorCode, mmcCreateCallW)(void *hCore, void *hndl, const wchar_t *type, void *hFrom, void *hTo);

/**
 * Creates a core instance and places the handle into the specified memory.
 *
 * The caller is responsible that hCore points to an allocated memory block
 * of sufficient size. The size must be determined by calling 
 * 'mmcGetHandleSize'. The caller remains owner of the memory and must ensure
 * that the memory is not freed or moved until 'mmcDisposeHandle' has returned.
 * The first byte of the memory block speified by hCore must be set to zero. 
 * If hCore point to a memory block already holding a valid core handle the 
 * method fails.
 *
 * Warning: DO NOT CHANGE the data hCore points to, as long as a valid core
 * handle is placed there.
 *
 * @param hCore Points to the memory receiving the core instance handle.
 *
 * @return 'MMC_ERR_NO_ERROR' on success or an nonzero error code if the 
 *         function fails.
 */
MEGAMOLCORE_CALL(mmcErrorCode, mmcCreateCore)(void *hCore);

/**
 * Creates a new module object
 *
 * @param hParent Handle to the namespace or core instance object which
 *                will be the parent of the new object.
 * @param hndl If not NULL, points to the memory to receive a handle to the
 *             newly created module object
 * @param type The type name for the new module object
 * @param name The name for the new module object
 *
 * @return The error code of the method
 */
MEGAMOLCORE_CALL(mmcErrorCode, mmcCreateModuleA)(void *hParent, void *hndl, const char *type, const char *name);

/**
 * Creates a new module object
 *
 * @param hParent Handle to the namespace or core instance object which
 *                will be the parent of the new object.
 * @param hndl If not NULL, points to the memory to receive a handle to the
 *             newly created module object
 * @param type The type name for the new module object
 * @param name The name for the new module object
 *
 * @return The error code of the method
 */
MEGAMOLCORE_CALL(mmcErrorCode, mmcCreateModuleW)(void *hParent, void *hndl, const wchar_t *type, const wchar_t *name);

/**
 * Creates a new namespace object
 *
 * @param hParent Handle to the namespace, module or core instance object which
 *                will be the parent of the new object.
 * @param hndl If not NULL, points to the memory to receive a handle to the
 *             newly created namespace object
 * @param name The name for the new namespace object
 *
 * @return The error code of the method
 */
MEGAMOLCORE_CALL(mmcErrorCode, mmcCreateNamespaceA)(void *hParent, void *hndl, const char *name);

/**
 * Creates a new namespace object
 *
 * @param hParent Handle to the namespace, module or core instance object which
 *                will be the parent of the new object.
 * @param hndl If not NULL, points to the memory to receive a handle to the
 *             newly created namespace object
 * @param name The name for the new namespace object
 *
 * @return The error code of the method
 */
MEGAMOLCORE_CALL(mmcErrorCode, mmcCreateNamespaceW)(void *hParent, void *hndl, const wchar_t *name);

/**
 * Creates an new value object
 *
 * @param hValue Memory to recieve the handle to the new value object
 *
 * @return The error code of the method
 */
MEGAMOLCORE_CALL(mmcErrorCode, mmcCreateValueObject)(void *hValue);

/**
 * Deletes an object and optionally all subobjects
 *
 * @param hndl The object to be deleted
 * @param recursive If set to true, all child objects will also be deleted, if
 *                  set to false and not all child object could be relocated
 *                  or otherwise handled, the function will return an error
 *
 * @return The error code of the method
 */
MEGAMOLCORE_CALL(mmcErrorCode, mmcDeleteObject)(void *hndl, bool recursive);

/**
 * Disposes a core handle. The handle will be invalid after this call.
 * Since this method cannot fail, it is save to call it with an invalid handle.
 * In this case no operation is performed.
 *
 * Note that handles to dependent object become implicitly invalid. Using such
 * an handle will result in undefined behaviour.
 *
 * @param hndl The core handle to be disposed.
 */
MEGAMOLCORE_CALL(void, mmcDisposeHandle)(void *hndl);

/**
 * Duplicates a core handle.
 *
 * @param tarHndl The targeted memory to receive the copy of the handle. The
 *                caller is responsible for the memory 'tarHndl' points to. It
 *                must be at least of 'mmcGetHandleSize()' bytes in size and
 *                must remain valid as long as a valid handle is placed inside.
 * @param srcHndl The handle to be duplicated. If srcHndl does not hold a valid
 *                handle, the function will not change the data 'tarHndl'
 *                points to
 *
 * @return True if the handle was successfully duplicated into 'tarHndl',
 *         otherwise,
 */
MEGAMOLCORE_CALL(bool, mmcDuplicateHandle)(void *tarHndl, void *srcHndl);

/**
 * Gets the handle to the call object connected to the specified caller slot
 *
 * @param hSlot The caller slot
 * @param hndl Points to the memory to receive the handle of the requested call
 *
 * @return The error code of the method
 */
MEGAMOLCORE_CALL(mmcErrorCode, mmcGetCallHandle)(void *hSlot, void *hndl);

/**
 * Answer all call types available for instantiation
 *
 * @param hCode Handle to the core instance
 * @param hNames Handle to the value object to recieve the array of names of
 *               the available call types. If it is not a valid handle, a new
 *               value object will be created at the memory pointed to
 * @param hDescs Handle to the value object to receive the array of human
 *               -readable descriptions of the available call types. If it
 *               is not a valid handle, a new value object will be created at
 *               the memory pointed to
 *
 * @return The error code of the method
 */
MEGAMOLCORE_CALL(mmcErrorCode, mmcGetCallTypes)(void *hCore, void *hNames, void *hDescs);

/**
 * Gets the children names of an object
 *
 * @param hParent The object to query
 * @param hValue The value object to receive the array of the names of the
 *               child objects. If hValue is not an value object a new value
 *               object is created at the memory hValue points to
 *
 * @return The error code of the method
 */
MEGAMOLCORE_CALL(mmcErrorCode, mmcGetChildren)(void *hParent, void *hValue);

/**
 * Gets the core handle of the core of the specified handle
 *
 * @param hndl The handle of the object to receive it's core
 * @param hCore  Points to the memory to receive the requested core handle
 *
 * @return The error code of the method
 */
MEGAMOLCORE_CALL(mmcErrorCode, mmcGetCoreHandle)(void *hndl, void *hCore);

/**
 * Returns the size needed to store a handle. All handles used by the 
 * MegaMolCore API have the same size.
 *
 * @return the size in bytes for the a handle
 */
MEGAMOLCORE_CALL(unsigned int, mmcGetHandleSize)(void);

/**
 * Answers the type of the supplied handle.
 *
 * @param hndl The handle to be tested.
 *
 * @return The type of the specified handle.
 */
MEGAMOLCORE_CALL(mmcHandleType, mmcGetHandleType)(void *hndl);

/**
 * Answer all module types available for instantiation
 *
 * @param hCode Handle to the core instance
 * @param hNames Handle to the value object to recieve the array of names of
 *               the available module types. If it is not a valid handle, a new
 *               value object will be created at the memory pointed to
 * @param hDescs Handle to the value object to receive the array of human
 *               -readable descriptions of the available module types. If it
 *               is not a valid handle, a new value object will be created at
 *               the memory pointed to
 *
 * @return The error code of the method
 */
MEGAMOLCORE_CALL(mmcErrorCode, mmcGetModuleTypes)(void *hCore, void *hNames, void *hDescs);

/**
 * Gets the name of an object
 *
 * @param hndl The object to query
 * @param hValue The value object to receive the name of the object. If hValue 
 *               is not an value object a new value object is created at the
 *               memory hValue points to
 *
 * @return The error code of the method
 */
MEGAMOLCORE_CALL(mmcErrorCode, mmcGetName)(void *hndl, void *hValue);

/**
 * Answer the handle to the object with the specified name
 *
 * @param hCode Handle to the core instance
 * @param hndl Points to the memory to receive the handle of the object with
 *             the requested name
 * @param name The name of the object to search for
 *
 * @return The error code of the method
 */
MEGAMOLCORE_CALL(mmcErrorCode, mmcGetObjectHandleA)(void *hCore, void *hndl, const char *name);

/**
 * Answer the handle to the object with the specified name
 *
 * @param hCode Handle to the core instance
 * @param hndl Points to the memory to receive the handle of the object with
 *             the requested name
 * @param name The name of the object to search for
 *
 * @return The error code of the method
 */
MEGAMOLCORE_CALL(mmcErrorCode, mmcGetObjectHandleW)(void *hCore, void *hndl, const wchar_t *name);

/**
 * Gets a value from an object. This can be a description-, meta-, parameter-,
 * or configuration-value
 *
 * @param hndl The handle to the object
 * @param name The name of the value
 * @param hValue The value object to receive the name of the object. If hValue 
 *               is not an value object a new value object is created at the
 *               memory hValue points to
 *
 * @return The error code of the method
 */
MEGAMOLCORE_CALL(mmcErrorCode, mmcGetObjectValue)(void *hndl, const char *name, void *hValue);

/**
 * Gets the value of a value object
 *
 * @param hValue The value object handle
 * @param index The index in the value object array (ignored if not an array)
 *
 * @return Pointer to the data stored
 */
MEGAMOLCORE_CALL(const void *, mmcGetValue)(void *hValue, SIZE_T index);

/**
 * Gets the size of the array of a value object
 *
 * @param hValue The value object handle
 *
 * @return The size of the array of the value object
 */
MEGAMOLCORE_CALL(SIZE_T, mmcGetValueArraySize)(void *hValue);

/**
 * Gets the size of a value of a value object
 *
 * @param hValue The value object handle
 * @param index The index in the value object array (ignored if not an array)
 *
 * @return The size of the data in bytes
 */
MEGAMOLCORE_CALL(SIZE_T, mmcGetValueSize)(void *hValue, SIZE_T index);

/**
 * Answer the value type stored in a value object
 *
 * @param hValue The value object handle
 *
 * @return The type of data stored in the value object
 */
MEGAMOLCORE_CALL(mmcValueType, mmcGetValueType)(void *hValue);

/**
 * Returns the version of the MegaMolCore.
 *
 * Applications should check this version directly after startup to ensure
 * compatibility with this core.
 *
 * @param outVersion Pointer to four short word receiving the version number
 * @param outSys Pointer to a mmcOSys variable receiving the system type of 
 *               the core.
 * @param outArch Pointer to a mmcHArchEnum variable receiving the 
 *                architecture of the core.
 * @param outFlags Pointer to an unsigned int receiving build flags MMV_BFLAG_*
 * @param outVislibVersion Pointer to four short word receiving the version of
 *                         the vislib used and linked to.
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
 * @param outCopyrightStr Pointer to a ANSI-character buffer receiving the
 *                        name of the library. Copies the whole name, but at
 *                        least 'inOutCopyrightSize' bytes and returns the
 *                        number of copied bytes in 'inOutCopyrightSize'.
 *                        'inOutCopyrightSize' must not be NULL if
 *                        'outCopyrightStr' is not NULL.
 * @param inOutCopyrightSize Pointer to an integer defining the size of the
 *                           buffer 'outCopyrightStr' points to. If
 *                           'outCopyrightStr' is NULL and
 *                           'inOutCopyrightSize' is not NULL,
 *                           'inOutCopyrightSize' receives the number of
 *                           characters (excluding the terminating zero)
 *                           required to store the name of the library.
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
MEGAMOLCORE_CALL(void, mmcGetVersionInfo)(
    unsigned short *outVersion,
    mmcOSys *outSys, mmcHArch *outArch, unsigned int *outFlags,
    unsigned short *outVislibVersion,
    char *outNameStr, unsigned int *inOutNameSize,
    char *outCopyrightStr, unsigned int *inOutCopyrightSize,
    char *outCommentStr, unsigned int *inOutCommentSize);

/**
 * Checks wether the supplied handle is a valid mega mol handle or not. If you
 * encounter problems with a handle and this functions returns true, the handle
 * may still be from a wrong handle type (e.g. providing an entry handle where
 * a view handle is wanted).
 *
 * @param hndl The handle to be tested.
 *
 * @return 'true' if the handle is a valid mega mol handle, otherwise 'false'.
 */
MEGAMOLCORE_CALL(bool, mmcIsHandleValid)(const void *hndl);

/**
 * Loads a configuation file. If 'configFilePath' is NULL, the default config.
 * file search is performed.
 *
 * @param hCode Handle to the core instance
 * @param configFilePath Path to the config file to load
 *
 * @return The error code of the method
 */
MEGAMOLCORE_CALL(mmcErrorCode, mmcLoadConfigurationA)(void *hCore, const char *configFilePath);

/**
 * Loads a configuation file. If 'configFilePath' is NULL, the default config.
 * file search is performed.
 *
 * @param hCode Handle to the core instance
 * @param configFilePath Path to the config file to load
 *
 * @return The error code of the method
 */
MEGAMOLCORE_CALL(mmcErrorCode, mmcLoadConfigurationW)(void *hCore, const wchar_t *configFilePath);

/**
 * Loads a project file into the core
 *
 * @param hCore The core instance handle
 * @param projFileName The file name path to the project file to load
 *
 * @return The error code of the method
 */
MEGAMOLCORE_CALL(mmcErrorCode, mmcLoadProjectA)(void *hCore, const char *projFileName);

/**
 * Loads a project file into the core
 *
 * @param hCore The core instance handle
 * @param projFileName The file name path to the project file to load
 *
 * @return The error code of the method
 */
MEGAMOLCORE_CALL(mmcErrorCode, mmcLoadProjectW)(void *hCore, const wchar_t *projFileName);

/**
 * Performs a quickstart of a data file
 *
 * @param hCore The core instance handle
 * @param filename The path to the data file to quickstart
 *
 * @return The error code of the method
 */
MEGAMOLCORE_CALL(mmcErrorCode, mmcQuickstartA)(void *hCore, const char *filename);

/**
 * Registers data file types for quickstart (if supported by the OS)
 *
 * @param hCore The core instance handle
 * @param frontend Path to the front end to be called
 * @param feparams The parameter string to be used when calling the frontend.
 *                 use '$(FILENAME)' to specify the position of the data file name.
 * @param filetype Semicolor separated list of file type extensions to register
 *                 or "*" if all known file type extensions should be used
 * @param unreg If true, the file types will be removed from the quickstart registry instead of added
 * @param overwrite If true, any previous registration will be overwritten.
 *                  If false, previous registrations will be placed as alternative start commands.
 *                  When unregistering and true, all registrations will be removed,
 *                  if false only registrations to this binary will be removed.
 *
 * @return The error code of the method
 */
MEGAMOLCORE_CALL(mmcErrorCode, mmcQuickstartRegistryA)(void *hCore,
    const char *frontend, const char *feparams,
    const char *filetype, bool unreg, bool overwrite);

/**
 * Registers data file types for quickstart (if supported by the OS)
 *
 * @param hCore The core instance handle
 * @param frontend Path to the front end to be called
 * @param feparams The parameter string to be used when calling the frontend.
 *                 use '$(FILENAME)' to specify the position of the data file name.
 * @param filetype Semicolor separated list of file type extensions to register
 *                 or "*" if all known file type extensions should be used
 * @param unreg If true, the file types will be removed from the quickstart registry instead of added
 * @param overwrite If true, any previous registration will be overwritten.
 *                  If false, previous registrations will be placed as alternative start commands.
 *                  When unregistering and true, all registrations will be removed,
 *                  if false only registrations to this binary will be removed.
 *
 * @return The error code of the method
 */
MEGAMOLCORE_CALL(mmcErrorCode, mmcQuickstartRegistryW)(void *hCore,
    const wchar_t *frontend, const wchar_t *feparams,
    const wchar_t *filetype, bool unreg, bool overwrite);

/**
 * Performs a quickstart of a data file
 *
 * @param hCore The core instance handle
 * @param filename The path to the data file to quickstart
 *
 * @return The error code of the method
 */
MEGAMOLCORE_CALL(mmcErrorCode, mmcQuickstartW)(void *hCore, const wchar_t *filename);

/**
 * Removes a callback from an object. The callback is identified by its call-
 * back function pointer and context pointer.
 *
 * @param hndl Handle to the object
 * @param name The function name to remove the callback from
 * @param callback The callback pointer
 * @param context The user defined context pointer
 *
 * @return The error code of the method
 */
MEGAMOLCORE_CALL(mmcErrorCode, mmcRemoveCallbackA)(void *hndl, const char *name, mmcCoreCallback callback, void *context);

/**
 * Removes a callback from an object. The callback is identified by its call-
 * back function pointer and context pointer.
 *
 * @param hndl Handle to the object
 * @param name The function name to remove the callback from
 * @param callback The callback pointer
 * @param context The user defined context pointer
 *
 * @return The error code of the method
 */
MEGAMOLCORE_CALL(mmcErrorCode, mmcRemoveCallbackW)(void *hndl, const wchar_t *name, mmcCoreCallback callback, void *context);

/**
 * Saves the current core instance into an project file
 *
 * @param hCore The core instance handle
 * @param projFileName The file name path to the project file to write
 * @param withParams If set to true all parameter values will also be written
 *                   to the file
 *
 * @return The error code of the method
 */
MEGAMOLCORE_CALL(mmcErrorCode, mmcSaveProjectA)(void *hCore, const char *projFileName, bool withParams);

/**
 * Saves the current core instance into an project file
 *
 * @param hCore The core instance handle
 * @param projFileName The file name path to the project file to write
 * @param withParams If set to true all parameter values will also be written
 *                   to the file
 *
 * @return The error code of the method
 */
MEGAMOLCORE_CALL(mmcErrorCode, mmcSaveProjectW)(void *hCore, const wchar_t *projFileName, bool withParams);

/**
 * Activates a configuration set
 *
 * @param hCore Handle to the core instance
 * @param cfgSetName The name of the config set to be activated
 *
 * @return The error code of the method
 */
MEGAMOLCORE_CALL(mmcErrorCode, mmcSelectConfigSetA)(void *hCore, const char *cfgSetName);

/**
 * Activates a configuration set
 *
 * @param hCore Handle to the core instance
 * @param cfgSetName The name of the config set to be activated
 *
 * @return The error code of the method
 */
MEGAMOLCORE_CALL(mmcErrorCode, mmcSelectConfigSetW)(void *hCore, const wchar_t *cfgSetName);

/**
 * Sets a value for an object. This can be a description-, meta-, parameter-,
 * or configuration-value, if setable.
 *
 * @param hndl The handle to the object
 * @param name The name of the value
 * @param hValue The value object holding the new value
 *
 * @return The error code of the method
 */
MEGAMOLCORE_CALL(mmcErrorCode, mmcSetObjectValue)(void *hndl, const char *name, void *hValue);

/**
 * Sets the value of a value object. If you want to set one value of an array,
 * you must specify the array type, not the scalar type. If you're not setting
 * a value of an array.
 *
 * @param hValue The value object handle
 * @param index The index in the value object array (ignored if not an array)
 * @param type The type of the value data
 * @param dat Pointer to the value data
 * @param size The size of the value data in bytes
 *
 * @return The error code of the method
 */
MEGAMOLCORE_CALL(mmcErrorCode, mmcSetValue)(void *hValue, SIZE_T index, mmcValueType type, const void *dat, SIZE_T size);

/**
 * Sets the array size of the value object
 *
 * @param hValue The value object handle
 * @param type The type of the value data (must be an array type)
 * @param size The size of the value data in elements
 *
 * @return The error code of the method
 */
MEGAMOLCORE_CALL(mmcErrorCode, mmcSetValueArraySize)(void *hValue, mmcValueType type, SIZE_T size);

#if defined(UNICODE) || defined(_UNICODE)
#define mmcAddCallback mmcAddCallbackW
#define mmcCreateCall mmcCreateCallW
#define mmcCreateModule mmcCreateModuleW
#define mmcCreateNamespace mmcCreateNamespaceW
#define mmcGetObjectHandle mmcGetObjectHandleW
#define mmcLoadConfiguration mmcLoadConfigurationW
#define mmcLoadProject mmcLoadProjectW
#define mmcQuickstart mmcQuickstartW
#define mmcQuickstartRegistry mmcQuickstartRegistryW
#define mmcRemoveCallback mmcRemoveCallbackW
#define mmcSaveProject mmcSaveProjectW
#define mmcSelectConfigSet mmcSelectConfigSetW

#else /* defined(UNICODE) || defined(_UNICODE) */
#define mmcAddCallback mmcAddCallbackA
#define mmcCreateCall mmcCreateCallA
#define mmcCreateModule mmcCreateModuleA
#define mmcCreateNamespace mmcCreateNamespaceA
#define mmcGetObjectHandle mmcGetObjectHandleA
#define mmcLoadConfiguration mmcLoadConfigurationA
#define mmcLoadProject mmcLoadProjectA
#define mmcQuickstart mmcQuickstartA
#define mmcQuickstartRegistry mmcQuickstartRegistryA
#define mmcRemoveCallback mmcRemoveCallbackA
#define mmcSaveProject mmcSaveProjectA
#define mmcSelectConfigSet mmcSelectConfigSetA

#endif /* defined(UNICODE) || defined(_UNICODE) */


#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* MEGAMOLCORE_MEGALMOLCORESTATIC_H_INCLUDED */

#else /* MEGAMOLVIEWER_USESTATIC */
/*
 * MegaMolCore.std.h
 *
 * Copyright (C) 2010 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCOREDYNAMIC_H_INCLUDED
#define MEGAMOLCOREDYNAMIC_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

//
// CONTENT OF THIS FILE IS GENERATED
// DO NOT EDIT
//


/*
 * Use 'MEGAMOLCOREAPI_SINGLEFILE' to create function symbols in exactly one
 * object file.
 */
 

/*
 * Include Check
 */
#ifndef MEGAMOLCORE_MEGALMOLCORE_H_INCLUDED
#error DO NOT Include "MegaMolCoreDynamic.h" directly! Include "MegaMolCore.h" instead.
#endif /* MEGAMOLCORE_MEGALMOLCORE_H_INCLUDED */


#undef MEGAMOLCORE_CALL
#ifdef _WIN32
#   define MEGAMOLCORE_CALL(R, F) typedef R (__cdecl * F##_FNPTRT)
#else /* _WIN32 */
#   define MEGAMOLCORE_CALL(R, F) typedef R (* F##_FNPTRT)
#endif /* _WIN32 */

/* Include static header for type definitions */
#include "MegaMolCoreStatic.h"

#ifdef MEGAMOLCOREAPI_SINGLEFILE
#   define MEGAMOLCORE_PTR(F) F##_FNPTRT F = 0;
#else /* MEGAMOLCOREAPI_SINGLEFILE */
#   define MEGAMOLCORE_PTR(F) extern F##_FNPTRT F;
#endif /* MEGAMOLCOREAPI_SINGLEFILE */

/*****************************************************************************/
/** FUNCTION POINTERS */

MEGAMOLCORE_PTR(mmcAddCallbackA)
MEGAMOLCORE_PTR(mmcAddCallbackW)
MEGAMOLCORE_PTR(mmcCall)
MEGAMOLCORE_PTR(mmcClear)
MEGAMOLCORE_PTR(mmcCompareHandles)
MEGAMOLCORE_PTR(mmcConnectVLStatics)
MEGAMOLCORE_PTR(mmcConvertValueType)
MEGAMOLCORE_PTR(mmcCreateCallA)
MEGAMOLCORE_PTR(mmcCreateCallW)
MEGAMOLCORE_PTR(mmcCreateCore)
MEGAMOLCORE_PTR(mmcCreateModuleA)
MEGAMOLCORE_PTR(mmcCreateModuleW)
MEGAMOLCORE_PTR(mmcCreateNamespaceA)
MEGAMOLCORE_PTR(mmcCreateNamespaceW)
MEGAMOLCORE_PTR(mmcCreateValueObject)
MEGAMOLCORE_PTR(mmcDeleteObject)
MEGAMOLCORE_PTR(mmcDisposeHandle)
MEGAMOLCORE_PTR(mmcDuplicateHandle)
MEGAMOLCORE_PTR(mmcGetCallHandle)
MEGAMOLCORE_PTR(mmcGetCallTypes)
MEGAMOLCORE_PTR(mmcGetChildren)
MEGAMOLCORE_PTR(mmcGetCoreHandle)
MEGAMOLCORE_PTR(mmcGetHandleSize)
MEGAMOLCORE_PTR(mmcGetHandleType)
MEGAMOLCORE_PTR(mmcGetModuleTypes)
MEGAMOLCORE_PTR(mmcGetName)
MEGAMOLCORE_PTR(mmcGetObjectHandleA)
MEGAMOLCORE_PTR(mmcGetObjectHandleW)
MEGAMOLCORE_PTR(mmcGetObjectValue)
MEGAMOLCORE_PTR(mmcGetValue)
MEGAMOLCORE_PTR(mmcGetValueArraySize)
MEGAMOLCORE_PTR(mmcGetValueSize)
MEGAMOLCORE_PTR(mmcGetValueType)
MEGAMOLCORE_PTR(mmcGetVersionInfo)
MEGAMOLCORE_PTR(mmcIsHandleValid)
MEGAMOLCORE_PTR(mmcLoadConfigurationA)
MEGAMOLCORE_PTR(mmcLoadConfigurationW)
MEGAMOLCORE_PTR(mmcLoadProjectA)
MEGAMOLCORE_PTR(mmcLoadProjectW)
MEGAMOLCORE_PTR(mmcQuickstartA)
MEGAMOLCORE_PTR(mmcQuickstartRegistryA)
MEGAMOLCORE_PTR(mmcQuickstartRegistryW)
MEGAMOLCORE_PTR(mmcQuickstartW)
MEGAMOLCORE_PTR(mmcRemoveCallbackA)
MEGAMOLCORE_PTR(mmcRemoveCallbackW)
MEGAMOLCORE_PTR(mmcSaveProjectA)
MEGAMOLCORE_PTR(mmcSaveProjectW)
MEGAMOLCORE_PTR(mmcSelectConfigSetA)
MEGAMOLCORE_PTR(mmcSelectConfigSetW)
MEGAMOLCORE_PTR(mmcSetObjectValue)
MEGAMOLCORE_PTR(mmcSetValue)
MEGAMOLCORE_PTR(mmcSetValueArraySize)


/*****************************************************************************/
/** LOADING FUNCTIONS */

#ifndef MEGAMOLCOREAPI_SINGLEFILE
namespace megamol {
namespace core {

    /**
     * Answer whether or not the library is loaded.
     *
     * @return 'true' if the library is loaded, 'false' otherwise.
     */
    extern bool mmcapiIsLibraryLoaded(void);

    /**
     * Load the MegaMol™ Core library with the specified name.
     *
     * @param libfilename The name of the library to load.
     *
     * @return 'true' if the library was successfully loaded, 'false' otherwise.
     */
    extern bool mmcapiLoadLibraryA(const char* filename);

    /**
     * Load the MegaMol™ Core library with the specified name.
     *
     * @param libfilename The name of the library to load.
     *
     * @return 'true' if the library was successfully loaded, 'false' otherwise.
     */
    extern bool mmcapiLoadLibraryW(const wchar_t* filename);

    /**
     * Unload the MegaMol™ Core library. This method should only be used if
     * it is really necessary. Usually it is not, because the library will be
     * released on application exit.
     */
    extern void mmcapiUnloadLibrary(void);
    
} /* end namespace core */
} /* end namespace megamol */
#endif /* !MEGAMOLCOREAPI_SINGLEFILE */

#if defined(UNICODE) || defined(_UNICODE)
#define mmcapiLoadLibrary mmcapiLoadLibraryW
#else /* defined(UNICODE) || defined(_UNICODE) */
#define mmcapiLoadLibrary mmcapiLoadLibraryA
#endif /* defined(UNICODE) || defined(_UNICODE) */

#ifdef MEGAMOLCOREAPI_SINGLEFILE

#include "vislib/DynamicLinkLibrary.h"
#include "vislib/functioncast.h"

namespace megamol {
namespace core {
	
    static vislib::sys::DynamicLinkLibrary __mmcLib;

    /** forward declaration */
    void mmcapiUnloadLibrary(void);


    /**
     * Answer whether or not the library is loaded.
     *
     * @return 'true' if the library is loaded, 'false' otherwise.
     */
    bool mmcapiIsLibraryLoaded(void) {
        return (__mmcLib.IsLoaded()
            && (mmcAddCallbackA != NULL)
            && (mmcAddCallbackW != NULL)
            && (mmcCall != NULL)
            && (mmcClear != NULL)
            && (mmcCompareHandles != NULL)
            && (mmcConnectVLStatics != NULL)
            && (mmcConvertValueType != NULL)
            && (mmcCreateCallA != NULL)
            && (mmcCreateCallW != NULL)
            && (mmcCreateCore != NULL)
            && (mmcCreateModuleA != NULL)
            && (mmcCreateModuleW != NULL)
            && (mmcCreateNamespaceA != NULL)
            && (mmcCreateNamespaceW != NULL)
            && (mmcCreateValueObject != NULL)
            && (mmcDeleteObject != NULL)
            && (mmcDisposeHandle != NULL)
            && (mmcDuplicateHandle != NULL)
            && (mmcGetCallHandle != NULL)
            && (mmcGetCallTypes != NULL)
            && (mmcGetChildren != NULL)
            && (mmcGetCoreHandle != NULL)
            && (mmcGetHandleSize != NULL)
            && (mmcGetHandleType != NULL)
            && (mmcGetModuleTypes != NULL)
            && (mmcGetName != NULL)
            && (mmcGetObjectHandleA != NULL)
            && (mmcGetObjectHandleW != NULL)
            && (mmcGetObjectValue != NULL)
            && (mmcGetValue != NULL)
            && (mmcGetValueArraySize != NULL)
            && (mmcGetValueSize != NULL)
            && (mmcGetValueType != NULL)
            && (mmcGetVersionInfo != NULL)
            && (mmcIsHandleValid != NULL)
            && (mmcLoadConfigurationA != NULL)
            && (mmcLoadConfigurationW != NULL)
            && (mmcLoadProjectA != NULL)
            && (mmcLoadProjectW != NULL)
            && (mmcQuickstartA != NULL)
            && (mmcQuickstartRegistryA != NULL)
            && (mmcQuickstartRegistryW != NULL)
            && (mmcQuickstartW != NULL)
            && (mmcRemoveCallbackA != NULL)
            && (mmcRemoveCallbackW != NULL)
            && (mmcSaveProjectA != NULL)
            && (mmcSaveProjectW != NULL)
            && (mmcSelectConfigSetA != NULL)
            && (mmcSelectConfigSetW != NULL)
            && (mmcSetObjectValue != NULL)
            && (mmcSetValue != NULL)
            && (mmcSetValueArraySize != NULL));
    }


    /**
     * Load the MegaMol™ Core library with the specified name.
     *
     * @param libfilename The name of the library to load.
     *
     * @return 'true' if the library was successfully loaded, 'false' otherwise.
     */
    bool mmcapiLoadLibraryA(const char* filename) {
        try {
            if (__mmcLib.IsLoaded()) {
                __mmcLib.Free();
            }
            if (!__mmcLib.Load(filename)) {
                throw 0;
            }
            mmcAddCallbackA = function_cast<mmcAddCallbackA_FNPTRT>(__mmcLib.GetProcAddress("mmcAddCallbackA"));
            mmcAddCallbackW = function_cast<mmcAddCallbackW_FNPTRT>(__mmcLib.GetProcAddress("mmcAddCallbackW"));
            mmcCall = function_cast<mmcCall_FNPTRT>(__mmcLib.GetProcAddress("mmcCall"));
            mmcClear = function_cast<mmcClear_FNPTRT>(__mmcLib.GetProcAddress("mmcClear"));
            mmcCompareHandles = function_cast<mmcCompareHandles_FNPTRT>(__mmcLib.GetProcAddress("mmcCompareHandles"));
            mmcConnectVLStatics = function_cast<mmcConnectVLStatics_FNPTRT>(__mmcLib.GetProcAddress("mmcConnectVLStatics"));
            mmcConvertValueType = function_cast<mmcConvertValueType_FNPTRT>(__mmcLib.GetProcAddress("mmcConvertValueType"));
            mmcCreateCallA = function_cast<mmcCreateCallA_FNPTRT>(__mmcLib.GetProcAddress("mmcCreateCallA"));
            mmcCreateCallW = function_cast<mmcCreateCallW_FNPTRT>(__mmcLib.GetProcAddress("mmcCreateCallW"));
            mmcCreateCore = function_cast<mmcCreateCore_FNPTRT>(__mmcLib.GetProcAddress("mmcCreateCore"));
            mmcCreateModuleA = function_cast<mmcCreateModuleA_FNPTRT>(__mmcLib.GetProcAddress("mmcCreateModuleA"));
            mmcCreateModuleW = function_cast<mmcCreateModuleW_FNPTRT>(__mmcLib.GetProcAddress("mmcCreateModuleW"));
            mmcCreateNamespaceA = function_cast<mmcCreateNamespaceA_FNPTRT>(__mmcLib.GetProcAddress("mmcCreateNamespaceA"));
            mmcCreateNamespaceW = function_cast<mmcCreateNamespaceW_FNPTRT>(__mmcLib.GetProcAddress("mmcCreateNamespaceW"));
            mmcCreateValueObject = function_cast<mmcCreateValueObject_FNPTRT>(__mmcLib.GetProcAddress("mmcCreateValueObject"));
            mmcDeleteObject = function_cast<mmcDeleteObject_FNPTRT>(__mmcLib.GetProcAddress("mmcDeleteObject"));
            mmcDisposeHandle = function_cast<mmcDisposeHandle_FNPTRT>(__mmcLib.GetProcAddress("mmcDisposeHandle"));
            mmcDuplicateHandle = function_cast<mmcDuplicateHandle_FNPTRT>(__mmcLib.GetProcAddress("mmcDuplicateHandle"));
            mmcGetCallHandle = function_cast<mmcGetCallHandle_FNPTRT>(__mmcLib.GetProcAddress("mmcGetCallHandle"));
            mmcGetCallTypes = function_cast<mmcGetCallTypes_FNPTRT>(__mmcLib.GetProcAddress("mmcGetCallTypes"));
            mmcGetChildren = function_cast<mmcGetChildren_FNPTRT>(__mmcLib.GetProcAddress("mmcGetChildren"));
            mmcGetCoreHandle = function_cast<mmcGetCoreHandle_FNPTRT>(__mmcLib.GetProcAddress("mmcGetCoreHandle"));
            mmcGetHandleSize = function_cast<mmcGetHandleSize_FNPTRT>(__mmcLib.GetProcAddress("mmcGetHandleSize"));
            mmcGetHandleType = function_cast<mmcGetHandleType_FNPTRT>(__mmcLib.GetProcAddress("mmcGetHandleType"));
            mmcGetModuleTypes = function_cast<mmcGetModuleTypes_FNPTRT>(__mmcLib.GetProcAddress("mmcGetModuleTypes"));
            mmcGetName = function_cast<mmcGetName_FNPTRT>(__mmcLib.GetProcAddress("mmcGetName"));
            mmcGetObjectHandleA = function_cast<mmcGetObjectHandleA_FNPTRT>(__mmcLib.GetProcAddress("mmcGetObjectHandleA"));
            mmcGetObjectHandleW = function_cast<mmcGetObjectHandleW_FNPTRT>(__mmcLib.GetProcAddress("mmcGetObjectHandleW"));
            mmcGetObjectValue = function_cast<mmcGetObjectValue_FNPTRT>(__mmcLib.GetProcAddress("mmcGetObjectValue"));
            mmcGetValue = function_cast<mmcGetValue_FNPTRT>(__mmcLib.GetProcAddress("mmcGetValue"));
            mmcGetValueArraySize = function_cast<mmcGetValueArraySize_FNPTRT>(__mmcLib.GetProcAddress("mmcGetValueArraySize"));
            mmcGetValueSize = function_cast<mmcGetValueSize_FNPTRT>(__mmcLib.GetProcAddress("mmcGetValueSize"));
            mmcGetValueType = function_cast<mmcGetValueType_FNPTRT>(__mmcLib.GetProcAddress("mmcGetValueType"));
            mmcGetVersionInfo = function_cast<mmcGetVersionInfo_FNPTRT>(__mmcLib.GetProcAddress("mmcGetVersionInfo"));
            mmcIsHandleValid = function_cast<mmcIsHandleValid_FNPTRT>(__mmcLib.GetProcAddress("mmcIsHandleValid"));
            mmcLoadConfigurationA = function_cast<mmcLoadConfigurationA_FNPTRT>(__mmcLib.GetProcAddress("mmcLoadConfigurationA"));
            mmcLoadConfigurationW = function_cast<mmcLoadConfigurationW_FNPTRT>(__mmcLib.GetProcAddress("mmcLoadConfigurationW"));
            mmcLoadProjectA = function_cast<mmcLoadProjectA_FNPTRT>(__mmcLib.GetProcAddress("mmcLoadProjectA"));
            mmcLoadProjectW = function_cast<mmcLoadProjectW_FNPTRT>(__mmcLib.GetProcAddress("mmcLoadProjectW"));
            mmcQuickstartA = function_cast<mmcQuickstartA_FNPTRT>(__mmcLib.GetProcAddress("mmcQuickstartA"));
            mmcQuickstartRegistryA = function_cast<mmcQuickstartRegistryA_FNPTRT>(__mmcLib.GetProcAddress("mmcQuickstartRegistryA"));
            mmcQuickstartRegistryW = function_cast<mmcQuickstartRegistryW_FNPTRT>(__mmcLib.GetProcAddress("mmcQuickstartRegistryW"));
            mmcQuickstartW = function_cast<mmcQuickstartW_FNPTRT>(__mmcLib.GetProcAddress("mmcQuickstartW"));
            mmcRemoveCallbackA = function_cast<mmcRemoveCallbackA_FNPTRT>(__mmcLib.GetProcAddress("mmcRemoveCallbackA"));
            mmcRemoveCallbackW = function_cast<mmcRemoveCallbackW_FNPTRT>(__mmcLib.GetProcAddress("mmcRemoveCallbackW"));
            mmcSaveProjectA = function_cast<mmcSaveProjectA_FNPTRT>(__mmcLib.GetProcAddress("mmcSaveProjectA"));
            mmcSaveProjectW = function_cast<mmcSaveProjectW_FNPTRT>(__mmcLib.GetProcAddress("mmcSaveProjectW"));
            mmcSelectConfigSetA = function_cast<mmcSelectConfigSetA_FNPTRT>(__mmcLib.GetProcAddress("mmcSelectConfigSetA"));
            mmcSelectConfigSetW = function_cast<mmcSelectConfigSetW_FNPTRT>(__mmcLib.GetProcAddress("mmcSelectConfigSetW"));
            mmcSetObjectValue = function_cast<mmcSetObjectValue_FNPTRT>(__mmcLib.GetProcAddress("mmcSetObjectValue"));
            mmcSetValue = function_cast<mmcSetValue_FNPTRT>(__mmcLib.GetProcAddress("mmcSetValue"));
            mmcSetValueArraySize = function_cast<mmcSetValueArraySize_FNPTRT>(__mmcLib.GetProcAddress("mmcSetValueArraySize"));

            if (mmcapiIsLibraryLoaded()) {
                return true;
            }
        } catch(...) {
        }
        mmcapiUnloadLibrary();
        return false;
    }


    /**
     * Load the MegaMol™ Core library with the specified name.
     *
     * @param libfilename The name of the library to load.
     *
     * @return 'true' if the library was successfully loaded, 'false' otherwise.
     */
    bool mmcapiLoadLibraryW(const wchar_t* filename) {
        try {
            if (__mmcLib.IsLoaded()) {
                __mmcLib.Free();
            }
            if (!__mmcLib.Load(filename)) {
                throw 0;
            }
            mmcAddCallbackA = function_cast<mmcAddCallbackA_FNPTRT>(__mmcLib.GetProcAddress("mmcAddCallbackA"));
            mmcAddCallbackW = function_cast<mmcAddCallbackW_FNPTRT>(__mmcLib.GetProcAddress("mmcAddCallbackW"));
            mmcCall = function_cast<mmcCall_FNPTRT>(__mmcLib.GetProcAddress("mmcCall"));
            mmcClear = function_cast<mmcClear_FNPTRT>(__mmcLib.GetProcAddress("mmcClear"));
            mmcCompareHandles = function_cast<mmcCompareHandles_FNPTRT>(__mmcLib.GetProcAddress("mmcCompareHandles"));
            mmcConnectVLStatics = function_cast<mmcConnectVLStatics_FNPTRT>(__mmcLib.GetProcAddress("mmcConnectVLStatics"));
            mmcConvertValueType = function_cast<mmcConvertValueType_FNPTRT>(__mmcLib.GetProcAddress("mmcConvertValueType"));
            mmcCreateCallA = function_cast<mmcCreateCallA_FNPTRT>(__mmcLib.GetProcAddress("mmcCreateCallA"));
            mmcCreateCallW = function_cast<mmcCreateCallW_FNPTRT>(__mmcLib.GetProcAddress("mmcCreateCallW"));
            mmcCreateCore = function_cast<mmcCreateCore_FNPTRT>(__mmcLib.GetProcAddress("mmcCreateCore"));
            mmcCreateModuleA = function_cast<mmcCreateModuleA_FNPTRT>(__mmcLib.GetProcAddress("mmcCreateModuleA"));
            mmcCreateModuleW = function_cast<mmcCreateModuleW_FNPTRT>(__mmcLib.GetProcAddress("mmcCreateModuleW"));
            mmcCreateNamespaceA = function_cast<mmcCreateNamespaceA_FNPTRT>(__mmcLib.GetProcAddress("mmcCreateNamespaceA"));
            mmcCreateNamespaceW = function_cast<mmcCreateNamespaceW_FNPTRT>(__mmcLib.GetProcAddress("mmcCreateNamespaceW"));
            mmcCreateValueObject = function_cast<mmcCreateValueObject_FNPTRT>(__mmcLib.GetProcAddress("mmcCreateValueObject"));
            mmcDeleteObject = function_cast<mmcDeleteObject_FNPTRT>(__mmcLib.GetProcAddress("mmcDeleteObject"));
            mmcDisposeHandle = function_cast<mmcDisposeHandle_FNPTRT>(__mmcLib.GetProcAddress("mmcDisposeHandle"));
            mmcDuplicateHandle = function_cast<mmcDuplicateHandle_FNPTRT>(__mmcLib.GetProcAddress("mmcDuplicateHandle"));
            mmcGetCallHandle = function_cast<mmcGetCallHandle_FNPTRT>(__mmcLib.GetProcAddress("mmcGetCallHandle"));
            mmcGetCallTypes = function_cast<mmcGetCallTypes_FNPTRT>(__mmcLib.GetProcAddress("mmcGetCallTypes"));
            mmcGetChildren = function_cast<mmcGetChildren_FNPTRT>(__mmcLib.GetProcAddress("mmcGetChildren"));
            mmcGetCoreHandle = function_cast<mmcGetCoreHandle_FNPTRT>(__mmcLib.GetProcAddress("mmcGetCoreHandle"));
            mmcGetHandleSize = function_cast<mmcGetHandleSize_FNPTRT>(__mmcLib.GetProcAddress("mmcGetHandleSize"));
            mmcGetHandleType = function_cast<mmcGetHandleType_FNPTRT>(__mmcLib.GetProcAddress("mmcGetHandleType"));
            mmcGetModuleTypes = function_cast<mmcGetModuleTypes_FNPTRT>(__mmcLib.GetProcAddress("mmcGetModuleTypes"));
            mmcGetName = function_cast<mmcGetName_FNPTRT>(__mmcLib.GetProcAddress("mmcGetName"));
            mmcGetObjectHandleA = function_cast<mmcGetObjectHandleA_FNPTRT>(__mmcLib.GetProcAddress("mmcGetObjectHandleA"));
            mmcGetObjectHandleW = function_cast<mmcGetObjectHandleW_FNPTRT>(__mmcLib.GetProcAddress("mmcGetObjectHandleW"));
            mmcGetObjectValue = function_cast<mmcGetObjectValue_FNPTRT>(__mmcLib.GetProcAddress("mmcGetObjectValue"));
            mmcGetValue = function_cast<mmcGetValue_FNPTRT>(__mmcLib.GetProcAddress("mmcGetValue"));
            mmcGetValueArraySize = function_cast<mmcGetValueArraySize_FNPTRT>(__mmcLib.GetProcAddress("mmcGetValueArraySize"));
            mmcGetValueSize = function_cast<mmcGetValueSize_FNPTRT>(__mmcLib.GetProcAddress("mmcGetValueSize"));
            mmcGetValueType = function_cast<mmcGetValueType_FNPTRT>(__mmcLib.GetProcAddress("mmcGetValueType"));
            mmcGetVersionInfo = function_cast<mmcGetVersionInfo_FNPTRT>(__mmcLib.GetProcAddress("mmcGetVersionInfo"));
            mmcIsHandleValid = function_cast<mmcIsHandleValid_FNPTRT>(__mmcLib.GetProcAddress("mmcIsHandleValid"));
            mmcLoadConfigurationA = function_cast<mmcLoadConfigurationA_FNPTRT>(__mmcLib.GetProcAddress("mmcLoadConfigurationA"));
            mmcLoadConfigurationW = function_cast<mmcLoadConfigurationW_FNPTRT>(__mmcLib.GetProcAddress("mmcLoadConfigurationW"));
            mmcLoadProjectA = function_cast<mmcLoadProjectA_FNPTRT>(__mmcLib.GetProcAddress("mmcLoadProjectA"));
            mmcLoadProjectW = function_cast<mmcLoadProjectW_FNPTRT>(__mmcLib.GetProcAddress("mmcLoadProjectW"));
            mmcQuickstartA = function_cast<mmcQuickstartA_FNPTRT>(__mmcLib.GetProcAddress("mmcQuickstartA"));
            mmcQuickstartRegistryA = function_cast<mmcQuickstartRegistryA_FNPTRT>(__mmcLib.GetProcAddress("mmcQuickstartRegistryA"));
            mmcQuickstartRegistryW = function_cast<mmcQuickstartRegistryW_FNPTRT>(__mmcLib.GetProcAddress("mmcQuickstartRegistryW"));
            mmcQuickstartW = function_cast<mmcQuickstartW_FNPTRT>(__mmcLib.GetProcAddress("mmcQuickstartW"));
            mmcRemoveCallbackA = function_cast<mmcRemoveCallbackA_FNPTRT>(__mmcLib.GetProcAddress("mmcRemoveCallbackA"));
            mmcRemoveCallbackW = function_cast<mmcRemoveCallbackW_FNPTRT>(__mmcLib.GetProcAddress("mmcRemoveCallbackW"));
            mmcSaveProjectA = function_cast<mmcSaveProjectA_FNPTRT>(__mmcLib.GetProcAddress("mmcSaveProjectA"));
            mmcSaveProjectW = function_cast<mmcSaveProjectW_FNPTRT>(__mmcLib.GetProcAddress("mmcSaveProjectW"));
            mmcSelectConfigSetA = function_cast<mmcSelectConfigSetA_FNPTRT>(__mmcLib.GetProcAddress("mmcSelectConfigSetA"));
            mmcSelectConfigSetW = function_cast<mmcSelectConfigSetW_FNPTRT>(__mmcLib.GetProcAddress("mmcSelectConfigSetW"));
            mmcSetObjectValue = function_cast<mmcSetObjectValue_FNPTRT>(__mmcLib.GetProcAddress("mmcSetObjectValue"));
            mmcSetValue = function_cast<mmcSetValue_FNPTRT>(__mmcLib.GetProcAddress("mmcSetValue"));
            mmcSetValueArraySize = function_cast<mmcSetValueArraySize_FNPTRT>(__mmcLib.GetProcAddress("mmcSetValueArraySize"));

            if (mmcapiIsLibraryLoaded()) {
                return true;
            }
        } catch(...) {
        }
        mmcapiUnloadLibrary();
        return false;
    }


    /**
     * Unload the MegaMol™ Core library. This method should only be used if
     * it is really necessary. Usually it is not, because the library will be
     * released on application exit.
     */
    void mmcapiUnloadLibrary(void) {
        try {
            if (__mmcLib.IsLoaded()) {
                __mmcLib.Free();
            }
        } catch(...) {
        }    
        mmcAddCallbackA = NULL;
        mmcAddCallbackW = NULL;
        mmcCall = NULL;
        mmcClear = NULL;
        mmcCompareHandles = NULL;
        mmcConnectVLStatics = NULL;
        mmcConvertValueType = NULL;
        mmcCreateCallA = NULL;
        mmcCreateCallW = NULL;
        mmcCreateCore = NULL;
        mmcCreateModuleA = NULL;
        mmcCreateModuleW = NULL;
        mmcCreateNamespaceA = NULL;
        mmcCreateNamespaceW = NULL;
        mmcCreateValueObject = NULL;
        mmcDeleteObject = NULL;
        mmcDisposeHandle = NULL;
        mmcDuplicateHandle = NULL;
        mmcGetCallHandle = NULL;
        mmcGetCallTypes = NULL;
        mmcGetChildren = NULL;
        mmcGetCoreHandle = NULL;
        mmcGetHandleSize = NULL;
        mmcGetHandleType = NULL;
        mmcGetModuleTypes = NULL;
        mmcGetName = NULL;
        mmcGetObjectHandleA = NULL;
        mmcGetObjectHandleW = NULL;
        mmcGetObjectValue = NULL;
        mmcGetValue = NULL;
        mmcGetValueArraySize = NULL;
        mmcGetValueSize = NULL;
        mmcGetValueType = NULL;
        mmcGetVersionInfo = NULL;
        mmcIsHandleValid = NULL;
        mmcLoadConfigurationA = NULL;
        mmcLoadConfigurationW = NULL;
        mmcLoadProjectA = NULL;
        mmcLoadProjectW = NULL;
        mmcQuickstartA = NULL;
        mmcQuickstartRegistryA = NULL;
        mmcQuickstartRegistryW = NULL;
        mmcQuickstartW = NULL;
        mmcRemoveCallbackA = NULL;
        mmcRemoveCallbackW = NULL;
        mmcSaveProjectA = NULL;
        mmcSaveProjectW = NULL;
        mmcSelectConfigSetA = NULL;
        mmcSelectConfigSetW = NULL;
        mmcSetObjectValue = NULL;
        mmcSetValue = NULL;
        mmcSetValueArraySize = NULL;
    }


} /* end namespace core */
} /* end namespace megamol */
#endif /* MEGAMOLCOREAPI_SINGLEFILE */

/*****************************************************************************/

#endif /* MEGAMOLCOREDYNAMIC_H_INCLUDED */

#endif /* MEGAMOLVIEWER_USESTATIC */


#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* MEGAMOLCORE_MEGALMOLCORE_H_INCLUDED */
