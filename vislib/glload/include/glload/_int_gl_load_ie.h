#ifndef _INT_GL_LOAD_IMPORT_EXPORT_H_INCLUDED
#define _INT_GL_LOAD_IMPORT_EXPORT_H_INCLUDED
#pragma once

#ifdef _WIN32
#ifdef GL_LOAD_DLL
#ifdef GL_LOAD_DLL_EXPORT
#define GLLOADAPI __declspec(dllexport) 
#else
#define GLLOADAPI __declspec(dllimport) 
#endif
#else
#define GLLOADAPI
#endif
#else
#define GLLOADAPI
#endif

#endif /* _INT_GL_LOAD_IMPORT_EXPORT_H_INCLUDED */
