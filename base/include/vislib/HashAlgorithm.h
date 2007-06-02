/*
 * HashAlgorithm.h
 *
 * Copyright (C) 2006 - 2007 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#ifndef VISLIB_HASHALGORITHM_H_INCLUDED
#define VISLIB_HASHALGORITHM_H_INCLUDED
#if (_MSC_VER > 1000)
#pragma once
#endif /* (_MSC_VER > 1000) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#include "vislib/String.h"
#include "vislib/types.h"


namespace vislib {


    /**
     * The base class all cryptographic hash algorithms are derived from.
     */
    class HashAlgorithm {

    public:

        /** Ctor. */
        HashAlgorithm(void);

        /** Dtor. */
        virtual ~HashAlgorithm(void);

        /**
         * Answer the hash value until now.
         *
         * If the hash has not yet been finalised, it will be temporarily 
         * finalised.
         *
         * @param outHash   Receives the hash.
         * @param inOutSize Pass the size of 'outHash' in this variable. After
         *                  the method returns, the actual size of the has
         *                  has been written to this variable, regardless 
         *                  whether the hash has been written to 'outHash'.
         *
         * @return true, if the hash was completely returned to 'outHash', 
         *         false, if 'outHash' was too small, but the hash could be 
         *         retrieved in principle. 'inOutSize' is set to the required
         *         size of 'outHash' in the latter case.
         *
         * @throws IllegalStateException If the hash has not been initialised.
         */
        virtual bool GetHash(BYTE *outHash, SIZE_T& inOutSize) const = 0;

        /**
         * Initialise the hash. This method must be called before any other 
         * operation on the hash object is performed.
         *
         * If the hash has already been initialised, this method erases all
         * previous data and reinitialises it.
         */
        virtual void Initialise(void) = 0;

        /**
         * Update the hash with a new block of 'cntInput' bytes.
         *
         * It is safe to pass a NULL pointer as 'input'. In this case, nothing
         * happens.
         *
         * @param input    The data to be added to the hash.
         * @param cntInput The number of bytes in 'input'.
         *
         * @throws IllegalStateException If the hash has not been initialised 
         *                               or was already finalised.
         */
        virtual void TransformBlock(const BYTE *input, 
            const SIZE_T cntInput) = 0;

        /**
         * Update the hash with a new block of 'cntInput' bytes and finalise
         * it afterwards. No data can be added after the final block was
         * transformed.
         *
         * It is safe to pass a NULL pointer as 'input'. In this case, the 
         * hash is not updated, but nevertheless finalised.
         *
         * @param input    The data to be added to the hash.
         * @param cntInput The number of bytes in 'input'.
         *
         * @throws IllegalStateException If the hash has not been initialised 
         *                               or was already finalised.
         */
        virtual void TransformFinalBlock(const BYTE *input, 
            const SIZE_T cntInput) = 0;

        /**
         * Answer a hexadecimal string representation of the hash.
         *
         * @return The hash as string.
         */
        virtual StringA ToStringA(void) const;

        /**
         * Answer a hexadecimal string representation of the hash.
         *
         * @return The hash as string.
         */
        virtual StringW ToStringW(void) const;
    };
    
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_HASHALGORITHM_H_INCLUDED */
