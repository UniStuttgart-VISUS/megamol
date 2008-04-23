/*
 * DynamicFunctionPointer.h
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#ifndef VISLIB_DYNAMICFUNCTIONPOINTER_H_INCLUDED
#define VISLIB_DYNAMICFUNCTIONPOINTER_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#ifdef _WIN32


#include <vislib/UnsupportedOperationException.h>
#include <windows.h>


namespace vislib {
namespace sys {


    /**
     * Wrapper for dynamically loaded functions from modules. The template 
     * parameter specifies the function pointer type.
     */
    template <class T> class DynamicFunctionPointer {
    public:

        /**
         * Ctor. Creates the function pointer for the function functionName in 
         * the module moduleName. The mudule must already be loaded! The 
         * modules reference counter is
         *
         * @param moduleName The name of the module.
         * @param functionName The name of the function.
         */
        DynamicFunctionPointer(char *moduleName, char *functionName);

        /** Dtor. */
        ~DynamicFunctionPointer(void);

        /**
         * Answer whether the function pointer is valid.
         *
         * @return true if the function pointer is valid, false otherwise.
         */
        inline bool IsValid(void) {
            return static_cast<void*>(func) != NULL;
        }

        /**
         * Cast operator to the function pointer for calling the function.
         *
         * @return The function pointer.
         */
        inline operator T&(void) {
            return this->func;
        }

    private:

        /** forbidden Ctor. */
        DynamicFunctionPointer(void);

        /** the function pointer */
        T func;
    };


    /*
     * vislib::sys::DynamicFunctionPointer::DynamicFunctionPointer
     */
    template<class T>
    vislib::sys::DynamicFunctionPointer<T>::DynamicFunctionPointer(char *moduleName, char *functionName) {
        this->func = static_cast<T>(static_cast<void*>(GetProcAddress(GetModuleHandleA(moduleName), functionName)));
    }


    /*
     * vislib::sys::DynamicFunctionPointer::~DynamicFunctionPointer
     */
    template<class T>
    vislib::sys::DynamicFunctionPointer<T>::~DynamicFunctionPointer(void) {
        // Do not delete/free/remove/whatever the function member this->func
    }


    /*
     * vislib::sys::DynamicFunctionPointer::DynamicFunctionPointer
     */
    template<class T>
    vislib::sys::DynamicFunctionPointer<T>::DynamicFunctionPointer(void) {
        throw UnsupportedOperationException("DynamicFunctionPointer ctor", __FILE__, __LINE__);
    }

} /* end namespace sys */
} /* end namespace vislib */

#endif /* _WIN32 */
#endif /* VISLIB_DYNAMICFUNCTIONPOINTER_H_INCLUDED */

