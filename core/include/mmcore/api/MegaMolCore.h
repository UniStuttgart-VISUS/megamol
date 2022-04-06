/*
 * MegaMolCore.h
 *
 * Copyright (C) 2006 - 2008 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#pragma once
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

/*****************************************************************************/
/** TYPES */

/** Possible values for CONFIGURATION ID */
typedef enum _mmcConfigID : int {
    MMC_CFGID_INVALID, // an invalid object!
    MMC_CFGID_APPLICATION_DIR,
    MMC_CFGID_CONFIG_FILE,
    MMC_CFGID_VARIABLE // a configured variable set-tag
} mmcConfigID;

/** Possible error codes */
typedef enum _mmcErrorCodeEnum : int {
    MMC_ERR_NO_ERROR = 0,    // No Error. This denotes success.
    MMC_ERR_MEMORY,          // Generic memory error.
    MMC_ERR_HANDLE,          // Generic handle error.
    MMC_ERR_INVALID_HANDLE,  // The handle specified was invalid.
    MMC_ERR_NOT_INITIALISED, // The object was not initialised.
    MMC_ERR_STATE,           // The object was in a incompatible state.
    MMC_ERR_TYPE,            // Generic type error (normally incompatible type or cast
                             // failed).
    MMC_ERR_NOT_IMPLEMENTED, // Function not implemented.
    MMC_ERR_LICENSING,       // Requested action not possible due to licensing
    MMC_ERR_UNKNOWN          // Unknown error.
} mmcErrorCode;

/** Possible value types. */
typedef enum _mmcValueTypeEnum : int {
    MMC_TYPE_INT32,  // 32 bit signed integer.(Pointer to!)
    MMC_TYPE_UINT32, // 32 bit unsigned integer.(Pointer to!)
    MMC_TYPE_INT64,  // 64 bit signed integer.(Pointer to!)
    MMC_TYPE_UINT64, // 64 bit unsigned integer.(Pointer to!)
    MMC_TYPE_BYTE,   // 8 bit unsigned integer.(Pointer to!)
    MMC_TYPE_BOOL,   // bool (platform specific integer size) (Pointer to!)
    MMC_TYPE_FLOAT,  // 32 bit float (Pointer to!)
    MMC_TYPE_CSTR,   // Ansi string (Pointer or Array of ansi characters).
    MMC_TYPE_WSTR,   // Unicode string (Pointer or Array of wide characters).
#if defined(UNICODE) || defined(_UNICODE)
#define MMC_TYPE_TSTR MMC_TYPE_WSTR
#else /* defined(UNICODE) || defined(_UNICODE) */
#define MMC_TYPE_TSTR MMC_TYPE_CSTR
#endif             /* defined(UNICODE) || defined(_UNICODE) */
    MMC_TYPE_VOIDP // Manuel type convertion. Use with care!
} mmcValueType;

/** Possible initialisation values */
typedef enum _mmcInitValueEnum : int {
    MMC_INITVAL_CFGFILE, // The configuration file to load.
    MMC_INITVAL_CFGSET, // A configuration set to be added. // TODO: deprecated but retained in case someone is using numbers instead of enum values
    MMC_INITVAL_LOGFILE,      // The log file to use.
    MMC_INITVAL_LOGLEVEL,     // The log level to use.
    MMC_INITVAL_LOGECHOLEVEL, // The log echo level to use.
    MMC_INITVAL_INCOMINGLOG,  // Connects an incoming log object to the one of
                              // the core instance IS NOT DEPRECATED
    MMC_INITVAL_LOGECHOFUNC,  // The log echo function to use.
    MMC_INITVAL_CORELOG,      // Returns the pointer to the core log
    MMC_INITVAL_CFGOVERRIDE   // a config value to override from the command line
} mmcInitValue;

/**
 * Function pointer type for log echo target functions.
 *
 * @param level The level of the log message
 * @param message The text of the log message
 */
typedef void (*mmcLogEchoFunction)(unsigned int level, const char* message);

/**
 * Function pointer type for view close requests.
 *
 * @param data The user specified pointer.
 */
typedef void (*mmcViewCloseRequestFunction)(void* data);

#ifdef __cplusplus
} /* extern "C" */
#endif
