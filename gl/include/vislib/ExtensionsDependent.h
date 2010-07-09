/*
 * ExtensionsDependent.h
 *
 * Copyright (C) 2006 - 2007 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#ifndef VISLIB_EXTENSIONSDEPENDENT_H_INCLUDED
#define VISLIB_EXTENSIONSDEPENDENT_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma warning(disable: 4996)
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#ifndef VISLIB_SYMBOL_EXPORT
#define GLH_EXT_IMPORT  // TODO: This is extremely hughly because of LNK4049, but works.
#endif /* !VISLIB_SYMBOL_EXPORT */
#include "glh/glh_extensions.h"
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma warning(default: 4996)
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#include "vislib/String.h"
#include "vislib/StringTokeniser.h"


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
    
        /**
         * Answers whether the required extensions are available.
         *
         * @return True if all required extensions are available and 
         *         initialisation should be successful, false otherwise.
         */
        static bool AreExtensionsAvailable(void);

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


    /*
     * ExtensionsDependent::InitialiseExtensions
     */
    template<class T> bool ExtensionsDependent<T>::AreExtensionsAvailable(void) {
        // check required extensions individually
        StringTokeniserA extensions(T::RequiredExtensions(), ' ');
        while (extensions.HasNext()) {
            StringA str = extensions.Next();
            str.TrimSpaces();
            if (str.IsEmpty()) continue;
            if (glh_extension_supported(str.PeekBuffer()) == 0) {
                return false; // this extension is missing
            }
        }
        return true; // no extension was missing
    }
    
} /* end namespace gl */
} /* end namespace graphics */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_EXTENSIONSDEPENDENT_H_INCLUDED */

