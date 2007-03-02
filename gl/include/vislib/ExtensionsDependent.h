/*
 * ExtensionsDependent.h
 *
 * Copyright (C) 2006 - 2007 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#ifndef VISLIB_EXTENSIONSDEPENDENT_H_INCLUDED
#define VISLIB_EXTENSIONSDEPENDENT_H_INCLUDED
#if (_MSC_VER > 1000)
#pragma once
#endif /* (_MSC_VER > 1000) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#if (_MSC_VER > 1000)
#pragma warning(disable: 4996)
#endif /* (_MSC_VER > 1000) */
#include "glh/glh_extensions.h"
#if (_MSC_VER > 1000)
#pragma warning(default: 4996)
#endif /* (_MSC_VER > 1000) */


namespace vislib {
namespace graphics {
namespace gl {


    /**
     * Base template class for classes which need opengl extensions. These
     * classes inherit from this template instanciated with the child class.
     * The child class must implement a public static method
     * "static const char * RequiredExtensions(void);" returning an ANSI string
     * containing the space-separated names of the required openGL extensions.
     * Leading spaces, trailing spaces and multiple spaces are allowed.
     *
     * Normally, when using classes which inherit from this template, 
     * "InitialiseExtensions" must be called before creating objects of these
     * classes.
     * 
     * Template parameter T: this subclass derived from this class.
     */
    template<class T> class ExtensionsDependent {
    public:

        /**
         * Initialise the extensions that are required for objects of the 
         * dependent class. This method must be called before using any objects
         * of the dependent classes.
         *
         * @return true, if all required extension could be loaded, 
         *         false otherwise.
         */
        static bool InitialiseExtensions(void);

    protected:

        /**
         * Disallow direct instances of this class. 
         */
        inline ExtensionsDependent(void) { }
    };


    /*
     * ExtensionsDependent::InitialiseExtensions
     */
    template<class T> bool ExtensionsDependent<T>::InitialiseExtensions(void) {
        return (::glh_init_extensions(T::RequiredExtensions()) != 0);
    }
    
} /* end namespace gl */
} /* end namespace graphics */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_EXTENSIONSDEPENDENT_H_INCLUDED */

