/*
 * D3DException.h
 *
 * Copyright (C) 2006 - 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#ifndef VISLIB_D3DEXCEPTION_H_INCLUDED
#define VISLIB_D3DEXCEPTION_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#include <windows.h>
#include <d3d9.h>

#include "vislib/Exception.h"


namespace vislib {
namespace graphics {
namespace d3d {


    /**
     * Create a new exception that represents a Direct3D error.
     */
    class D3DException : public vislib::Exception {

    public:

        /**
         * Create a new exception that represents a Direct3D error.
         *
         * @param result The Direct3D error code.
         * @param file   The file the exception was thrown in.
         * @param line   The line the exception was thrown in.
         */
        D3DException(const HRESULT result, const char *file, const int line);

        /**
         * Clone 'rhs'.
         *
         * @param rhs The object to be cloned.
         */
        D3DException(const D3DException& rhs);

        /** Dtor. */
        virtual ~D3DException(void);

        /**
         * Answer the error code that was the reason for this exception.
         *
         * @return The error code.
         */
        inline HRESULT GetResult(void) const {
            return this->result;
        }

        /**
         * Assignment.
         *
         * @param rhs The right hand side operand.
         *
         * @return *this.
         */
        D3DException& operator =(const D3DException& rhs);

    protected:

        /** The error code that was the reason for this exception. */
        HRESULT result;
    };
    
} /* end namespace d3d */
} /* end namespace graphics */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_D3DEXCEPTION_H_INCLUDED */
