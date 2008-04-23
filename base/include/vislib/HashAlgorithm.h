/*
 * HashAlgorithm.h
 *
 * Copyright (C) 2006 - 2007 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#ifndef VISLIB_HASHALGORITHM_H_INCLUDED
#define VISLIB_HASHALGORITHM_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#include "vislib/String.h"
#include "vislib/types.h"


namespace vislib {


    /**
     * The base class all cryptographic hash algorithms are derived from.
     *
     * All subclasses must implement the initialisation, the transformation of
     * a block and the transformation of the final block. This superclass 
     * implements some convenience methods, which faciliate the use of the
     * hash algorithms.
     */
    class HashAlgorithm {

    public:

        /** Ctor. */
        HashAlgorithm(void);

        /** Dtor. */
        virtual ~HashAlgorithm(void);

        /**
         * Computes the hash of 'input'. This is a convenience method that 
         * performs an automatic reinitalisation of the hash. Using it is 
         * equivalent to calling Initialise followed by TransformFinalBlock.
         *
         * @param outHash   Receives the hash.
         * @param inOutSize Pass the size of 'outHash' in this variable. After
         *                  the method returns, the actual size of the has
         *                  has been written to this variable, regardless 
         *                  whether the hash has been written to 'outHash'.
         * @param input     The data to be added to the hash.
         * @param cntInput  The number of bytes in 'input'.
         *
         * @return true, if the hash was completely returned to 'outHash', 
         *         false, if 'outHash' was too small, but the hash could be 
         *         retrieved in principle. 'inOutSize' is set to the required
         *         size of 'outHash' in the latter case.
         */
        bool ComputeHash(BYTE *outHash, SIZE_T& inOutSize, 
            const BYTE *input, const SIZE_T cntInput);

        /**
         * Computes the hash of the zero-terminated string 'input'. This is 
         * a convenience method that performs an automatic reinitalisation of 
         * the hash. Using it is equivalent to calling Initialise followed by 
         * TransformFinalBlock with the length of 'input' as size parameter for
         * the input length.
         *
         * @param outHash   Receives the hash.
         * @param inOutSize Pass the size of 'outHash' in this variable. After
         *                  the method returns, the actual size of the has
         *                  has been written to this variable, regardless 
         *                  whether the hash has been written to 'outHash'.
         * @param input     The data to be added to the hash.
         * @param cntInput  The number of bytes in 'input'.
         *
         * @return true, if the hash was completely returned to 'outHash', 
         *         false, if 'outHash' was too small, but the hash could be 
         *         retrieved in principle. 'inOutSize' is set to the required
         *         size of 'outHash' in the latter case.
         */
        bool ComputeHash(BYTE *outHash, SIZE_T& inOutSize, 
            const char *input);

        /**
         * Computes the hash of the zero-terminated string 'input'. This is 
         * a convenience method that performs an automatic reinitalisation of 
         * the hash. Using it is equivalent to calling Initialise followed by 
         * TransformFinalBlock with the length of 'input' as size parameter for
         * the input length.
         *
         * @param outHash   Receives the hash.
         * @param inOutSize Pass the size of 'outHash' in this variable. After
         *                  the method returns, the actual size of the has
         *                  has been written to this variable, regardless 
         *                  whether the hash has been written to 'outHash'.
         * @param input     The data to be added to the hash.
         * @param cntInput  The number of bytes in 'input'.
         *
         * @return true, if the hash was completely returned to 'outHash', 
         *         false, if 'outHash' was too small, but the hash could be 
         *         retrieved in principle. 'inOutSize' is set to the required
         *         size of 'outHash' in the latter case.
         */
        bool ComputeHash(BYTE *outHash, SIZE_T& inOutSize, 
            const wchar_t *input);

        /**
         * Answer the size of the hash value in bytes.
         *
         * @return The size of the hash in bytes.
         */
        SIZE_T GetHashSize(void) const;

        /**
         * Answer the hash value until now.
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
         */
        bool GetHashValue(BYTE *outHash, SIZE_T& inOutSize) const;

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
         */
        virtual void TransformBlock(const BYTE *input, 
            const SIZE_T cntInput) = 0;

        /**
         * Update the hash with a new block of 'cntInput' bytes and compute the
         * hash value afterwards.
         *
         * It is safe to pass a NULL pointer as 'input'. In this case, the 
         * hash is not updated, but only its current value is returned. This
         * call is equivalent to calling GetHashValue.
         *
         * @param outHash   Receives the hash.
         * @param inOutSize Pass the size of 'outHash' in this variable. After
         *                  the method returns, the actual size of the has
         *                  has been written to this variable, regardless 
         *                  whether the hash has been written to 'outHash'.
         * @param input     The data to be added to the hash.
         * @param cntInput  The number of bytes in 'input'.
         */
        virtual bool TransformFinalBlock(BYTE *outHash, SIZE_T& inOutSize, 
            const BYTE *input, const SIZE_T cntInput) = 0;

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

    protected:

        /**
         * Copy ctor.
         *
         * @param rhs The object to be cloned.
         */
        HashAlgorithm(const HashAlgorithm& rhs);

        /**
         * Assignment operator.
         *
         * @param rhs The right hand side operand.
         *
         * @return *this.
         */
        HashAlgorithm& operator =(const HashAlgorithm& rhs);

    };
    
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_HASHALGORITHM_H_INCLUDED */
