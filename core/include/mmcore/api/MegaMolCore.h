/*
 * MegaMolCore.h
 *
 * Copyright (C) 2006 - 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_MEGALMOLCORE_H_INCLUDED
#define MEGAMOLCORE_MEGALMOLCORE_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#include <cstdint>

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


#include "MegaMolCore.std.h"


/*****************************************************************************/
/** TYPES */

/** Possible values for CONFIGURATION ID */
typedef enum _mmcConfigID : int {
    MMC_CFGID_INVALID, // an invalid object!
    MMC_CFGID_APPLICATION_DIR,
    MMC_CFGID_CONFIG_FILE,
    MMC_CFGID_VARIABLE // a configured variable set-tag
} mmcConfigID;

/** Possible operating systems */
#ifndef MMCOSYSENUM_DEFINED
#define MMCOSYSENUM_DEFINED 1
typedef enum _mmcOSysEnum : int {
    MMC_OSYSTEM_WINDOWS,
    MMC_OSYSTEM_LINUX,
    MMC_OSYSTEM_UNKNOWN
} mmcOSys;
#endif /* MMCOSYSENUM_DEFINED */

/** Possible hardware architectures */
#ifndef MMCHARCHENUM_DEFINED
#define MMCHARCHENUM_DEFINED 1
typedef enum _mmcHArchEnum : int {
    MMC_HARCH_I86,
    MMC_HARCH_X64,
    MMC_HARCH_UNKNOWN
} mmcHArch;
#endif /* MMCHARCHENUM_DEFINED */

#ifndef MMCBINARYVERSIONINFO_DEFINED
#define MMCBINARYVERSIONINFO_DEFINED 1
typedef struct _mmcBinaryVersionInfo_t {
    const char *VersionNumber[3];
    mmcOSys SystemType;
    mmcHArch HardwareArchitecture;
    unsigned int Flags;
    const char *NameStr;
    const char *CopyrightStr;
    const char *CommentStr;
} mmcBinaryVersionInfo;
#endif /* MMCBINARYVERSIONINFO_DEFINED */

/** Possible handle types */
typedef enum _mmcHandleTypeEnum : int {
    MMC_HTYPE_INVALID, // The handle is invalid or no handle at all.
                       // which type.
    MMC_HTYPE_COREINSTANCE, // Core Instance handle.
    MMC_HTYPE_VIEWINSTANCE, // View Instance handle.
    MMC_HTYPE_JOBINSTANCE, // Job Instance handle.
    MMC_HTYPE_PARAMETER, // A Parameter handle.
    MMC_HTYPE_UNKNOWN // The handle is a valid handle, but it is unknown of
} mmcHandleType;

/** Possible error codes */
typedef enum _mmcErrorCodeEnum : int {
    MMC_ERR_NO_ERROR = 0, // No Error. This denotes success.
    MMC_ERR_MEMORY, // Generic memory error.
    MMC_ERR_HANDLE, // Generic handle error.
    MMC_ERR_INVALID_HANDLE, // The handle specified was invalid.
    MMC_ERR_NOT_INITIALISED, // The object was not initialised.
    MMC_ERR_STATE, // The object was in a incompatible state.
    MMC_ERR_TYPE, // Generic type error (normally incompatible type or cast 
                  // failed).
    MMC_ERR_NOT_IMPLEMENTED, // Function not implemented.
    MMC_ERR_LICENSING, // Requested action not possible due to licensing
    MMC_ERR_UNKNOWN // Unknown error.
} mmcErrorCode;

/** Possible value types. */
typedef enum _mmcValueTypeEnum : int {
    MMC_TYPE_INT32, // 32 bit signed integer.(Pointer to!)
    MMC_TYPE_UINT32, // 32 bit unsigned integer.(Pointer to!)
    MMC_TYPE_INT64, // 64 bit signed integer.(Pointer to!)
    MMC_TYPE_UINT64, // 64 bit unsigned integer.(Pointer to!)
    MMC_TYPE_BYTE, // 8 bit unsigned integer.(Pointer to!)
    MMC_TYPE_BOOL, // bool (platform specific integer size) (Pointer to!)
    MMC_TYPE_FLOAT, // 32 bit float (Pointer to!)
    MMC_TYPE_CSTR, // Ansi string (Pointer or Array of ansi characters).
    MMC_TYPE_WSTR, // Unicode string (Pointer or Array of wide characters).
#if defined(UNICODE) || defined(_UNICODE)
#define MMC_TYPE_TSTR MMC_TYPE_WSTR
#else /* defined(UNICODE) || defined(_UNICODE) */
#define MMC_TYPE_TSTR MMC_TYPE_CSTR
#endif /* defined(UNICODE) || defined(_UNICODE) */
    MMC_TYPE_VOIDP // Manuel type convertion. Use with care!
} mmcValueType;

/** Possible initialisation values */
typedef enum _mmcInitValueEnum : int {
    MMC_INITVAL_CFGFILE, // The configuration file to load.
    MMC_INITVAL_CFGSET, // A configuration set to be added. // TODO: deprecated but retained in case someone is using numbers instead of enum values
    MMC_INITVAL_LOGFILE, // The log file to use.
    MMC_INITVAL_LOGLEVEL, // The log level to use.
    MMC_INITVAL_LOGECHOLEVEL, // The log echo level to use.
    MMC_INITVAL_INCOMINGLOG, // Connects an incoming log object to the one of 
                             // the core instance IS NOT DEPRECATED
    MMC_INITVAL_LOGECHOFUNC, // The log echo function to use.
    MMC_INITVAL_CORELOG, // Returns the pointer to the core log
    MMC_INITVAL_CFGOVERRIDE // a config value to override from the command line
} mmcInitValue;

/**
 * Possible input modifier values.
 *
 * See megamol::core::view::Modifiers
 */
typedef int mmcInputModifiers;

/**
 * Possible input key codes.
 *
 * See megamol::core::view::Key
 */
typedef int mmcInputKey;

/**
 * Possible input key states.
 *
 * See megamol::core::view::KeyAction
 */
typedef int mmcInputKeyAction;

/**
 * Possible input mouse buttons.
 *
 * See megamol::core::view::MouseButton
 */
typedef int mmcInputButton;

/**
 * Possible input mouse action states.
 *
 * See megamol::core::view::MouseButtonAction
 */
typedef int mmcInputButtonAction;

/** User context for mmcRenderView function. */
typedef struct _mmcRenderViewContext {
    /**
     * The size of this structure (Must remain first member in any future 
     * version and must always be four-byte integer).
     */
    INT32 Size;

    /**
     * Boolean to receive whether or not a continuous redraw of this view is 
     * required (Must remain second member at offset 4 Bytes in any future 
     * version).
     */
    bool ContinuousRedraw;

    // D3D defines the GPU affinity via the device object, therefore we do not
    // need the device index in this case.
    union {
        /** (Dumb) pointer to a Direct3D device if a D3D renderer is active. */
        void *Direct3DDevice;

        /**
         * Specifies the the GPU that the viewer uses for rendering the view.
         * If the viewer does not know about the GPU or the GPU is not relevant,
         * this handle is nullptr.
         *
         * On a Windows machine with NVDIA GPU, the pointer is actually a
         * HGPUNV; we do not use the type here in order to prevent the OpenGL
         * extension header being included in the MegaMol core API header.
         *
         * In all other cases, this parameter currently has no meaning.
         */
        void *GpuAffinity;
    };

    /** (Dumb) pointer to the Direct3D render target. */
    void *Direct3DRenderTarget;

    /** The instance time, which is generated by mmcRenderView. */
    double InstanceTime;

    /**
     * If positive, force the time to this value in mmcRenderView().
     * Otherwise, use the time generated by the view module.
     * If negative, store this time value into the context afterwards.
     */
    double Time;
} mmcRenderViewContext;

/** Library building flags */
#define MMC_BFLAG_DEBUG     0x00000001  // debug build
#define MMC_BFLAG_DIRTY     0x00000002  // dirty build (DO NOT RELEASE!)

/**
 * Function pointer type for log echo target functions.
 *
 * @param level The level of the log message
 * @param message The text of the log message
 */
typedef void (MEGAMOLCORE_CALLBACK *mmcLogEchoFunction)(unsigned int level,
    const char* message);

/**
 * Function pointer type for string enumeration functions.
 * ANSI version.
 *
 * @param str The incoming string (zero-terminated). Do not modify or free the
 *            memory. The memory is only valid with the function called.
 * @param data The user specified pointer.
 */
typedef void (MEGAMOLCORE_CALLBACK *mmcEnumStringAFunction)(const char *str,
    void *data);

/**
 * Function pointer type for string enumeration functions.
 * ANSI version.
 *
 * @param str The incoming string (zero-terminated). Do not modify or free the
 *            memory. The memory is only valid with the function called.
 * @param data The user specified pointer.
 */
typedef void (MEGAMOLCORE_CALLBACK *mmcEnumStringWFunction)(const wchar_t *str,
    void *data);

/**
 * Function pointer type for view close requests.
 *
 * @param data The user specified pointer.
 */
typedef void (MEGAMOLCORE_CALLBACK *mmcViewCloseRequestFunction)(void *data);

#if defined(UNICODE) || defined(_UNICODE)
#define mmcEnumStringFunction mmcEnumStringWFunction
#else /* defined(UNICODE) || defined(_UNICODE) */
#define mmcEnumStringFunction mmcEnumStringAFunction
#endif /* defined(UNICODE) || defined(_UNICODE) */

/*****************************************************************************/
/** DEFINES */

#define MMC_USING_VERIFY mmcErrorCode __mmc_verify_error__;

#define MMC_VERIFY_THROW(call) \
    if ((__mmc_verify_error__ = call) != MMC_ERR_NO_ERROR) {\
        vislib::StringA str; \
        str.Format("MegaMolCore Error %d", __mmc_verify_error__); \
        throw vislib::Exception(str, __FILE__, __LINE__);\
    }


/*****************************************************************************/
/** FUNCTIONS */

/**
 * Returns the binary version info of the MegaMol core.
 *
 * Applications should check this version directly after startup to ensure
 * compatibility with this core.
 *
 * @return The newly allocated binary version info data
 *
 * @remarks Use mmcFreeVersionInfo to free the returned memory
 */
MEGAMOLCORE_API mmcBinaryVersionInfo* MEGAMOLCORE_CALL mmcGetVersionInfo(void);

/**
 * Frees the memory of a binary version info previously retunred by mmcGetVersionInfo
 *
 * @param info The info memory to be freed
 */
MEGAMOLCORE_API void MEGAMOLCORE_CALL mmcFreeVersionInfo(mmcBinaryVersionInfo* info);

/**
 * Returns the size needed to store a handle. All handles used by the 
 * MegaMolCore API have the same size.
 *
 * @return the size in bytes for the a handle
 */
MEGAMOLCORE_API unsigned int MEGAMOLCORE_CALL mmcGetHandleSize(void);

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
MEGAMOLCORE_API void MEGAMOLCORE_CALL mmcDisposeHandle(void *hndl);

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
MEGAMOLCORE_API bool MEGAMOLCORE_CALL mmcIsHandleValid(void *hndl);

/**
 * Answers the type of the supplied handle.
 *
 * @param hndl The handle to be tested.
 *
 * @return The type of the specified handle.
 */
MEGAMOLCORE_API mmcHandleType MEGAMOLCORE_CALL mmcGetHandleType(void *hndl);

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
MEGAMOLCORE_API mmcErrorCode MEGAMOLCORE_CALL mmcCreateCore(void *hCore);

/**
 * Sets a initialisation value. The core instance must not be initialised yet.
 *
 * @param hCore The core instance handle.
 * @param key Specifies which value to set.
 * @param type Specifies the value type of 'value'.
 * @param value The value to set the initialisation value to. The type of the
 *              variable specified depends on 'type'.
 *
 * @return 'MMC_ERR_NO_ERROR' on success or an nonzero error code if the 
 *         function fails.
 */
MEGAMOLCORE_API mmcErrorCode MEGAMOLCORE_CALL mmcSetInitialisationValue(
    void *hCore, mmcInitValue key, mmcValueType type, const void* value);

/**
 * Initialises the core instance. A core instance must not be initialised 
 * twice!
 *
 * @param hCore The core instance handle.
 *
 * @return 'MMC_ERR_NO_ERROR' on success or an nonzero error code if the 
 *         function fails.
 */
MEGAMOLCORE_API mmcErrorCode MEGAMOLCORE_CALL mmcInitialiseCoreInstance(
    void *hCore);

/**
 * Answer a configuration value of the core instance. The memory the returned
 * pointer points to remains valid as long as the instance is not disposed and
 * until this method is called the next time. The caller must not free the
 * returned memory.
 *
 * @param hCore The core instance handle.
 * @param id The id of the configuration value to be returned.
 * @param name The name of the configuration value to be returned. The effect
 *             of this parameter depends on the value of 'id'.
 * @param outType Pointer to a variable receiving the type of the returned
 *                value.
 *
 * @return The requested configuration value.
 */
#if defined(UNICODE) || defined(_UNICODE)
#define mmcGetConfigurationValue mmcGetConfigurationValueW
#else
#define mmcGetConfigurationValue mmcGetConfigurationValueA
#endif

/**
 * ANSI Implementation of mmcGetConfigurationValue
 * @see mmcGetConfigurationValue
 */
MEGAMOLCORE_API const void * MEGAMOLCORE_CALL mmcGetConfigurationValueA(
    void *hCore, mmcConfigID id, const char *name, mmcValueType *outType);

/**
 * Unicode Implementation of mmcGetConfigurationValue
 * @see mmcGetConfigurationValue
 */
MEGAMOLCORE_API const void * MEGAMOLCORE_CALL mmcGetConfigurationValueW(
    void *hCore, mmcConfigID id, const wchar_t *name,
    mmcValueType *outType);

/**
 * Sets a configuration value of the core instance.
 *
 * @param hCore The core instance handle.
 * @param id The id of the configuration value to be set.
 * @param name The name of the configuration value to be set. The effect
 *             of this parameter depends on the value of 'id'.
 * @param val Value as string to be set.
 *
 * @return True on success
 */
#if defined(UNICODE) || defined(_UNICODE)
#define mmcSetConfigurationValue mmcSetConfigurationValueW
#else
#define mmcSetConfigurationValue mmcSetConfigurationValueA
#endif

/**
 * ANSI Implementation of mmcSetConfigurationValue
 * @see mmcSetConfigurationValue
 */
MEGAMOLCORE_API bool MEGAMOLCORE_CALL mmcSetConfigurationValueA(
    void *hCore, mmcConfigID id, const char *name, const char *val);

/**
 * Unicode Implementation of mmcSetConfigurationValue
 * @see mmcSetConfigurationValue
 */
MEGAMOLCORE_API bool MEGAMOLCORE_CALL mmcSetConfigurationValueW(
    void *hCore, mmcConfigID id, const wchar_t *name, const wchar_t* val);

/**
 * Request all available instances.
 *
 * @param hCore The core instance handle.
 */
MEGAMOLCORE_API void MEGAMOLCORE_CALL mmcRequestAllInstances(void *hCore);

/**
 * Requests the instantiation of a job or a view.
 *
 * @param hCore The core instance handle.
 * @param name The name of the job or view to be instantiated.
 * @param id The identifier name to be set for the instance.
 */
#if defined(UNICODE) || defined(_UNICODE)
#define mmcRequestInstance mmcRequestInstanceW
#else
#define mmcRequestInstance mmcRequestInstanceA
#endif

/**
 * ANSI Implementation of mmcRequestInstance
 * @see mmcRequestInstance
 */
MEGAMOLCORE_API void MEGAMOLCORE_CALL mmcRequestInstanceA(
    void *hCore, const char *name, const char *id);

/**
 * Unicode Implementation of mmcRequestInstance
 * @see mmcRequestInstance
 */
MEGAMOLCORE_API void MEGAMOLCORE_CALL mmcRequestInstanceW(
    void *hCore, const wchar_t *name, const wchar_t *id);

/**
 * Answer whether the core has pending requests of instantiations of views.
 *
 * @param hCore The core instance handle.
 *
 * @return 'true' if there are pending view instantiation requests,
 *         'false' otherwise.
 */
MEGAMOLCORE_API bool MEGAMOLCORE_CALL mmcHasPendingViewInstantiationRequests(
    void *hCore);

/**
 * Answer the name of the next pending view instance
 *
 * @param hCore The core instance handle.
 *
 * @return The name of the next pending view instance. This string will remain
 *         valid until this function is called the next time.
 */
MEGAMOLCORE_API const char* MEGAMOLCORE_CALL mmcGetPendingViewInstanceName(void *hCore);

/**
 * Creates a view instance out of the next pending view instantiation request.
 * If there is no such request the functions returns 'false' immediately.
 *
 * The caller is responsible that hView points to an allocated memory block
 * of sufficient size. The size must be determined by calling 
 * 'mmcGetHandleSize'. The caller remains owner of the memory and must ensure
 * that the memory is not freed or moved until 'mmcDisposeHandle' has returned.
 * The first byte of the memory block speified by hView must be set to zero. 
 * If hView point to a memory block already holding a valid core handle the 
 * method fails.
 *
 * Warning: DO NOT CHANGE the data hView points to, as long as a valid core
 * handle is placed there.
 *
 * @param hCore The core instance handle.
 * @param hView Pointer to the memory receiving the handle to the new view 
 *              instance.
 *
 * @return 'true' if a new view handle has been created, 'false' otherwise.
 */
MEGAMOLCORE_API bool MEGAMOLCORE_CALL mmcInstantiatePendingView(void *hCore,
    void *hView);

/**
 * Answer whether the core has pending requests of instantiations of jobs.
 *
 * @param hCore The core instance handle.
 *
 * @return 'true' if there are pending job instantiation requests,
 *         'false' otherwise.
 */
MEGAMOLCORE_API bool MEGAMOLCORE_CALL mmcHasPendingJobInstantiationRequests(
    void *hCore);

/**
 * Creates a job instance out of the next pending job instantiation request.
 * If there is no such request the functions returns 'false' immediately.
 *
 * The caller is responsible that hJob points to an allocated memory block
 * of sufficient size. The size must be determined by calling 
 * 'mmcGetHandleSize'. The caller remains owner of the memory and must ensure
 * that the memory is not freed or moved until 'mmcDisposeHandle' has returned.
 * The first byte of the memory block speified by hJob must be set to zero. 
 * If hJob point to a memory block already holding a valid core handle the 
 * method fails.
 *
 * Warning: DO NOT CHANGE the data hJob points to, as long as a valid core
 * handle is placed there.
 *
 * @param hCore The core instance handle.
 * @param hView Pointer to the memory receiving the handle to the new job 
 *              instance.
 *
 * @return 'true' if a new job handle has been created, 'false' otherwise.
 */
MEGAMOLCORE_API bool MEGAMOLCORE_CALL mmcInstantiatePendingJob(void *hCore,
    void *hJob);

/**
 * Renders a view into the currently active OpenGL context.
 *
 * @param hView The view instance handle.
 * @param context Context structure to transfer data from and to the view. 
 *                Please ensure that the 'Size' parameter has been initialised
 *                with sizeof(mmcRenderViewContext) before the call is made.
 * @param frameID the count of rendered frames so far               
 */
MEGAMOLCORE_API void MEGAMOLCORE_CALL mmcRenderView(void *hView, mmcRenderViewContext* context, uint32_t frameID);

/**
 * Registers a view close request function for the view. This function will be
 * called by the core if a view wants to close itself.
 *
 * @param hView The view on which to register the function
 * @param func The function to be registered
 * @param data The user data pointer which will be passed to the function upon
 *             call
 */
MEGAMOLCORE_API void MEGAMOLCORE_CALL mmcRegisterViewCloseRequestFunction(
    void *hView, mmcViewCloseRequestFunction func, void *data);

/**
 * Resizes a view.
 *
 * @param hView The view instance handle.
 * @param width The new width of the view.
 * @param height The new height of the view.
 */
MEGAMOLCORE_API void MEGAMOLCORE_CALL mmcResizeView(void *hView,
    unsigned int width, unsigned int height);

/**
 * Emits a key event.
 *
 * @param hView The view instance handle.
 * @param key The key code.
 * @param act The key action.
 * @param mods The input modifiers.
 *
 * @return True if the event was consumed, otherwise false.
 */
MEGAMOLCORE_API bool MEGAMOLCORE_CALL mmcSendKeyEvent(void *hView,
	mmcInputKey key, mmcInputKeyAction act, mmcInputModifiers mods);

/**
 * Emits a character event.
 *
 * @param hView The view instance handle.
 * @param cp The unicode code point.
 *
 * @return True if the event was consumed, otherwise false.
 */
MEGAMOLCORE_API bool MEGAMOLCORE_CALL mmcSendCharEvent(void* hView, 
	unsigned int cp);

/**
 * Emits a mouse button event.
 *
 * @param hView The view instance handle.
 * @param btn The mouse button.
 * @param act The mouse button action.
 * @param mods The input modifiers.
 *
 * @return True if the event was consumed, otherwise false.
 */
MEGAMOLCORE_API bool MEGAMOLCORE_CALL mmcSendMouseButtonEvent(void *hView,
    mmcInputButton btn, mmcInputButtonAction act, mmcInputModifiers mods);

/**
 * Emits a mouse move event.
 *
 * @param hView The view instance handle.
 * @param x The x coordinate of the mouse.
 * @param y The y coordinate of the mouse.
 *
 * @return True if the event was consumed, otherwise false.
 */
MEGAMOLCORE_API bool MEGAMOLCORE_CALL mmcSendMouseMoveEvent(void *hView,
    float x, float y);

/**
 * Sets the state of a button of the 2d mouse.
 *
 * @param hView The view instance handle.
 * @param dx The x coordinate offset of the mouse.
 * @param dy The y coordinate offset of the mouse.
 *
 * @return True if the event was consumed, otherwise false.
 */
MEGAMOLCORE_API bool MEGAMOLCORE_CALL mmcSendMouseScrollEvent(void *hView,
    float dx, float dy);

/**
 * Answers the desired window position configuration of this view.
 *
 * @param hView The view instance handle.
 * @param x To receive the coordinate of the upper left corner
 * @param y To recieve the coordinate of the upper left corner
 * @param w To receive the width
 * @param h To receive the height
 * @param nd To receive the flag deactivating window decorations
 *
 * @return 'true' if this view has a desired window position configuration,
 *         'false' if not. In the latter case the value the parameters are
 *         pointing to are not altered.
 */
MEGAMOLCORE_API bool MEGAMOLCORE_CALL mmcDesiredViewWindowConfig(void *hView,
    int *x, int *y, int *w, int *h, bool *nd);

/**
 * Gets whether or not a given job is running.
 *
 * @param hJob The job to be tested.
 *
 * @return 'true' if the job is still running, 'false' otherwise.
 */
MEGAMOLCORE_API bool MEGAMOLCORE_CALL mmcIsJobRunning(void *hJob);

/**
 * Gets whether or not a given view is running.
 *
 * @param hView The view to be tested.
 *
 * @return 'true' if the view is still running, 'false' otherwise.
 */
MEGAMOLCORE_API bool MEGAMOLCORE_CALL mmcIsViewRunning(void *hView);

/**
 * Starts a job.
 *
 * @param hJob The job to be terminated.
 *
 * @return 'true' if the job has been started
 */
MEGAMOLCORE_API bool MEGAMOLCORE_CALL mmcStartJob(void *hJob);

/**
 * Termiantes a job.
 *
 * @param hJob The job to be terminated.
 */
MEGAMOLCORE_API void MEGAMOLCORE_CALL mmcTerminateJob(void *hJob);

/**
 * Sets a parameter to a value.
 *
 * @param hParam The parameter handle.
 * @param value The value for the parameter.
 */
#if defined(UNICODE) || defined(_UNICODE)
#define mmcSetParameterValue mmcSetParameterValueW
#else
#define mmcSetParameterValue mmcSetParameterValueA
#endif

/**
 * ANSI Implementation of mmcSetParameterValue
 * @see mmcSetParameterValue
 */
MEGAMOLCORE_API void MEGAMOLCORE_CALL mmcSetParameterValueA(void *hParam,
    const char *value);

/**
 * Unicode Implementation of mmcSetParameterValue
 * @see mmcSetParameterValue
 */
MEGAMOLCORE_API void MEGAMOLCORE_CALL mmcSetParameterValueW(void *hParam,
    const wchar_t *value);

/**
 * Loads a project into the core.
 *
 * @param hCore The core instance handle.
 * @param filename The path to the project file to load.
 */
#if defined(UNICODE) || defined(_UNICODE)
#define mmcLoadProject mmcLoadProjectW
#else
#define mmcLoadProject mmcLoadProjectA
#endif

/**
 * ANSI Implementation of mmcLoadProject
 * @see mmcLoadProject
 */
MEGAMOLCORE_API void MEGAMOLCORE_CALL mmcLoadProjectA(void *hCore,
    const char *filename);

/**
 * Unicode Implementation of mmcLoadProject
 * @see mmcLoadProject
 */
MEGAMOLCORE_API void MEGAMOLCORE_CALL mmcLoadProjectW(void *hCore,
    const wchar_t *filename);

/**
 * Gets the parameter handle for the parameter with the given name.
 *
 * The caller is responsible that 'hParam' points to an allocated memory block
 * of sufficient size. The size must be determined by calling 
 * 'mmcGetHandleSize'. The caller remains owner of the memory and must ensure
 * that the memory is not freed or moved until 'mmcDisposeHandle' has returned.
 * The first byte of the memory block speified by 'hParam' must be set to zero.
 * If 'hParam' point to a memory block already holding a valid core handle the
 * method fails.
 *
 * @param hCode The core instance handle.
 * @param name The full name of the parameter to return its handle.
 * @param hParam Pointer to the memory receiving the parameter handle.
 * @param bCreate Create a StringParam if the name is not found
 *
 * @return 'true' if the operation was successful and hParam now holds a valid
 *         handle. 'false' if the operation failed. In that case hParam does
 *         not hold a valid handle (and thus you do not need to release it).
 */
#if defined(UNICODE) || defined(_UNICODE)
#define mmcGetParameter mmcGetParameterW
#else
#define mmcGetParameter mmcGetParameterA
#endif

/**
 * ANSI Implementation of mmcGetParameter
 * @see mmcGetParameter
 */
MEGAMOLCORE_API bool MEGAMOLCORE_CALL mmcGetParameterA(void *hCore,
    const char *name, void *hParam, bool bCreate = false);

/**
 * Unicode Implementation of mmcGetParameter
 * @see mmcGetParameter
 */
MEGAMOLCORE_API bool MEGAMOLCORE_CALL mmcGetParameterW(void *hCore,
    const wchar_t *name, void *hParam, bool bCreate = false);

/**
 * Gets the value of a parameter. The memory of the returned pointer remains
 * valid as long as the core instance is not destroied and as long as this
 * method is not called again. The caller must not free the returned memory.
 *
 * @param hParam The parameter handle.
 *
 * @return The value of the parameter or NULL if there is no parameter with
 *         the specified name.
 */
#if defined(UNICODE) || defined(_UNICODE)
#define mmcGetParameterValue mmcGetParameterValueW
#else
#define mmcGetParameterValue mmcGetParameterValueA
#endif

/**
 * ANSI Implementation of mmcGetParameter
 * @see mmcGetParameter
 */
MEGAMOLCORE_API const char * MEGAMOLCORE_CALL mmcGetParameterValueA(
    void *hParam);

/**
 * Unicode Implementation of mmcGetParameter
 * @see mmcGetParameter
 */
MEGAMOLCORE_API const wchar_t * MEGAMOLCORE_CALL mmcGetParameterValueW(
    void *hParam);


/**
 * Enumerates all parameters. The callback function is called for each
 * parameter name.
 *
 * @param hCore The core instance handle.
 * @param func The callback function.
 * @param data The user specified pointer to be passed to the callback
 *             function.
 */
#if defined(UNICODE) || defined(_UNICODE)
#define mmcEnumParameters mmcEnumParametersW
#else
#define mmcEnumParameters mmcEnumParametersA
#endif

/**
 * ANSI Implementation of mmcEnumParameters
 * @see mmcEnumParameters
 */
MEGAMOLCORE_API void MEGAMOLCORE_CALL mmcEnumParametersA(void *hCore,
    mmcEnumStringAFunction func, void *data);

/**
 * Unicode Implementation of mmcEnumParameters
 * @see mmcEnumParameters
 */
MEGAMOLCORE_API void MEGAMOLCORE_CALL mmcEnumParametersW(void *hCore,
    mmcEnumStringWFunction func, void *data);

/**
 * Get the instance identifier of a view, job or parameter instance.
 *
 * @param hInst The view, job or parameter instance handle.
 * @param buf The buffer to receive the identifier. Can be NULL when
 *            requesting the required size of the buffer.
 * @param len Pointer to the size of the buffer. If 'buf' is NULL the variable
 *            'len' points to is set to the size required to store the whole
 *            identifier string (in characters, including the terminating
 *            zero). If 'buf' is not NULL, the identifier string will be copied
 *            to the memory 'buf' points to, but will only write '*len'
 *            characters at most.
 */
#if defined(UNICODE) || defined(_UNICODE)
#define mmcGetInstanceID mmcGetInstanceIDW
#else
#define mmcGetInstanceID mmcGetInstanceIDA
#endif

/**
 * ANSI Implementation of mmcGetInstanceID
 * @see mmcGetInstanceID
 */
MEGAMOLCORE_API void MEGAMOLCORE_CALL mmcGetInstanceIDA(void *hInst,
    char *buf, unsigned int *len);

/**
 * Unicode Implementation of mmcGetInstanceID
 * @see mmcGetInstanceID
 */
MEGAMOLCORE_API void MEGAMOLCORE_CALL mmcGetInstanceIDW(void *hInst,
    wchar_t *buf, unsigned int *len);

/**
 * Answers whether a parameter is relevant for the given view or job instance.
 *
 * @param hInst The view or job instance handle.
 * @param param The full name of the parameter to test.
 *
 * @return 'true' if 'param' is relevant for 'hInst', 'false' if not.
 */
MEGAMOLCORE_API bool MEGAMOLCORE_CALL mmcIsParameterRelevant(void *hInst,
    void *hParam);

/**
 * Gets the type description of a parameter.
 *
 * @param hParam The parameter handle.
 * @param buf The buffer to receive the type description. Can be NULL when
 *            requesting the required size of the buffer.
 * @param len Pointer to the size of the buffer. If 'buf' is NULL the variable
 *            'len' points to is set to the size required to store the whole
 *            identifier string (in bytes). If 'buf' is not NULL, the
 *            identifier string will be copied to the memory 'buf' points to,
 *            but will only write '*len' bytes at most.
 */
MEGAMOLCORE_API void MEGAMOLCORE_CALL mmcGetParameterTypeDescription(
    void *hParam, unsigned char *buf, unsigned int *len);

/**
 * Updates global parameter hash and returns it.
 * Comparison of parameter info is expensive.
 *
 * @param hCore The core handle.
 *
 * @return      Updated parameter hash.
 */
MEGAMOLCORE_API size_t MEGAMOLCORE_CALL mmcGetGlobalParameterHash(void *hCore);

/**
 * Freezes, updates or unfreezes a view (and all associated modules)
 *
 * @param hView The view to be frozen or updated
 * @param freeze The action flag: true means freeze or update,
 *               false means unfreeze.
 */
MEGAMOLCORE_API void MEGAMOLCORE_CALL mmcFreezeOrUpdateView(
    void *hView, bool freeze);

/**
 * Performs a quickstart of a data file
 *
 * @param hCore The core instance handle
 * @param filename The path to the data file to quickstart
 */
#if defined(UNICODE) || defined(_UNICODE)
#define mmcQuickstart mmcQuickstartW
#else
#define mmcQuickstart mmcQuickstartA
#endif

/**
 * ANSI Impiementation of mmcQuickstart
 * @see mmcQuickstart
 */
MEGAMOLCORE_API void MEGAMOLCORE_CALL mmcQuickstartA(void *hCore, const char *filename);

/**
 * Unicode implementation of mmcQuickstart
 * @see mmcQuickstart
 */
MEGAMOLCORE_API void MEGAMOLCORE_CALL mmcQuickstartW(void *hCore, const wchar_t *filename);

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
 */
#if defined(UNICODE) || defined(_UNICODE)
#define mmcQuickstartRegistry mmcQuickstartRegistryW
#else
#define mmcQuickstartRegistry mmcQuickstartRegistryA
#endif

/**
 * ANSI Impiementation of mmcQuickstartRegistry
 * @see mmcQuickstartRegistry
 */
MEGAMOLCORE_API void MEGAMOLCORE_CALL mmcQuickstartRegistryA(void *hCore,
    const char *frontend, const char *feparams,
    const char *filetype, bool unreg, bool overwrite);

/**
 * Unicode implementation of mmcQuickstartRegistry
 * @see mmcQuickstartRegistry
 */
MEGAMOLCORE_API void MEGAMOLCORE_CALL mmcQuickstartRegistryW(void *hCore,
    const wchar_t *frontend, const wchar_t *feparams,
    const wchar_t *filetype, bool unreg, bool overwrite);

#ifndef MEGAMOLCORE_EXT_API
#define MEGAMOLCORE_EXT_API 1
#define MEGAMOLCORE_EXT_APICALL(A, B) MEGAMOLCORE_API A MEGAMOLCORE_CALL B
#endif /* MEGAMOLCORE_EXT_API */

MEGAMOLCORE_API void MEGAMOLCORE_CALL mmcWriteStateToXMLA(void *hCore, const char *outFilename);

/**
 * Perform all queued graph modification requests: delete modules/calls,
 * then create new modules/calls.
 */
MEGAMOLCORE_API void MEGAMOLCORE_CALL mmcPerformGraphUpdates(void *hCore);

#ifdef __cplusplus
} /* extern "C" */
#endif



#endif /* MEGAMOLCORE_MEGALMOLCORE_H_INCLUDED */
