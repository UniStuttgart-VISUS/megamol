/*
 * mmstd.moldyn.h
 *
 * Copyright (C) 2009 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */
#ifndef MMSTD.MOLDYN_H_INCLUDED
#define MMSTD.MOLDYN_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */


#ifdef _WIN32
// The following ifdef block is the standard way of creating macros which make exporting 
// from a DLL simpler. All files within this DLL are compiled with the TRISOUPPLUGIN_EXPORTS
// symbol defined on the command line. this symbol should not be defined on any project
// that uses this DLL. This way any other project whose source files include this file see 
// TRISOUPPLUGIN_API functions as being imported from a DLL, whereas this DLL sees symbols
// defined with this macro as being exported.
#ifdef MMSTD.MOLDYN_EXPORTS
#define MMSTD.MOLDYN_API __declspec(dllexport)
#else
#define MMSTD.MOLDYN_API __declspec(dllimport)
#endif
#else /* _WIN32 */
#define MMSTD.MOLDYN_API
#endif /* _WIN32 */


#ifdef __cplusplus
extern "C" {
#endif

/**
 * Returns the version of the MegaMol™ plugin api used by this plugin.
 *
 * @return The used MegaMol™ plugin api
 */
MMSTD.MOLDYN_API int mmplgPluginAPIVersion(void);

/**
 * Answer the name of the plugin in UTF8/ASCII7
 *
 * @return The name of the plugin in UTF8/ASCII7
 */
MMSTD.MOLDYN_API const char * mmplgPluginName(void);

/**
 * Answer the description of the plugin in UTF8/ASCII7
 *
 * @return The description of the plugin in UTF8/ASCII7
 */
MMSTD.MOLDYN_API const char * mmplgPluginDescription(void);

/**
 * Answer the core compatibility information
 *
 * @return The core compatibility information
 */
MMSTD.MOLDYN_API const void * mmplgCoreCompatibilityValue(void);

/**
 * Answer the number of exported modules
 *
 * @return The number of exported modules
 */
MMSTD.MOLDYN_API int mmplgModuleCount(void);

/**
 * Answer the module definition object of the idx-th module
 *
 * @param idx The zero-based index
 *
 * @return The module definition
 */
MMSTD.MOLDYN_API void* mmplgModuleDescription(int idx);

/**
 * Answer the number of exported calls
 *
 * @return The number of exported calls
 */
MMSTD.MOLDYN_API int mmplgCallCount(void);

/**
 * Answer the call definition object of the idx-th call
 *
 * @param idx The zero-based index
 *
 * @return The call definition
 */
MMSTD.MOLDYN_API void* mmplgCallDescription(int idx);

/**
 * Connects static objects to the core. (See docu for more information)
 *
 * @param which A numberic value identifying the static object
 * @param value The value to connect the static object to
 *
 * @return True if this static object has been connected, false if the object
 *         either does not exist or if there was an error.
 */
MMSTD.MOLDYN_API bool mmplgConnectStatics(int which, void* value);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* MMSTD.MOLDYN_H_INCLUDED */
