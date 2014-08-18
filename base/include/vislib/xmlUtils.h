/*
 * xmlUtils.h
 *
 * Copyright (C) 2006 - 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 * Copyright (C) 2008 by Sebastian Grottel. Alle Rechte vorbehalten.
 */

#ifndef VISLIB_XMLUTILS_H_INCLUDED
#define VISLIB_XMLUTILS_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#include "vislib/String.h"


namespace vislib {
namespace xml {

    /**
     * Replaces the following characters with their corresponding xml entities:
     *  '&'  &amp;
     *  '<'  &lt;
     *  '>'  &gt;
     *  '"'  &quot;
     *  '''  &apos;
     *
     * @param inOutStr The string in which the replace operations will occure.
     */
    template<class T> void EncodeEntities(vislib::String<T>& inOutStr) {
        // TODO: Rewrite as soon as multi-replace is supported
        inOutStr.Replace(static_cast<typename T::Char>('&'),
            vislib::String<T>("&amp;"));
        inOutStr.Replace(static_cast<typename T::Char>('<'),
            vislib::String<T>("&lt;"));
        inOutStr.Replace(static_cast<typename T::Char>('>'),
            vislib::String<T>("&gt;"));
        inOutStr.Replace(static_cast<typename T::Char>('\"'),
            vislib::String<T>("&quot;"));
        inOutStr.Replace(static_cast<typename T::Char>('\''),
            vislib::String<T>("&apos;"));
    }

    /**
     * Replaces the following xml entities to their corresponding characters:
     *  &amp;  '&'
     *  &lt;   '<'
     *  &gt;   '>'
     *  &quot; '"'
     *  &apos; '''
     * Be aware of the fact that after replacing the known entities additional
     * and unknown entities could emerge:
     *  e.g. "&amp;lt;" => "&lt;" => "<"
     *
     * @param inOutStr The string in which the replace operations will occure.
     */
    template<class T> void DecodeEntities(vislib::String<T>& inOutStr) {
        // TODO: Rewrite as soon as multi-replace is supported
        // TODO: support numeric character reference entities.
        //   (e.g. &#x3E; &62; also with leading zeros!)
        //   This will need to do all the replaceings by hand!
        inOutStr.Replace(vislib::String<T>("&apos;"),
            static_cast<typename T::Char>('\''));
        inOutStr.Replace(vislib::String<T>("&quot;"),
            static_cast<typename T::Char>('\"'));
        inOutStr.Replace(vislib::String<T>("&gt;"),
            static_cast<typename T::Char>('>'));
        inOutStr.Replace(vislib::String<T>("&lt;"),
            static_cast<typename T::Char>('<'));
        inOutStr.Replace(vislib::String<T>("&amp;"),
            static_cast<typename T::Char>('&'));
    }


} /* end namespace xml */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_XMLUTILS_H_INCLUDED */

