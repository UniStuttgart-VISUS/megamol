/*
 * MD5HashProvider.h
 *
 * Copyright (C) 2006 - 2007 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#ifndef VISLIB_MD5HASHPROVIDER_H_INCLUDED
#define VISLIB_MD5HASHPROVIDER_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#include "vislib/HashAlgorithm.h"


namespace vislib {


    /**
     * This class provides MD5 hashes as defined in RFC 1321. It produces a
     * 128-bit fingerprint of the input.
     *
     * This code is derived from the RSA Data Security, Inc. MD5 
     * Message-Digest Algorithm.
     */
    class MD5HashProvider : public HashAlgorithm {

    public:

        /** 
         * Initialises the hash. It is not necessary to call Initialise before
         * the first use.
         */
        MD5HashProvider(void);

        /** Dtor. */
        virtual ~MD5HashProvider(void);

        /**
         * Initialise the hash. This method must be called before any other 
         * operation on the hash object is performed.
         *
         * If the hash has already been initialised, this method erases all
         * previous data and reinitialises it.
         */
        virtual void Initialise(void);

        /**
         * Update the hash with a new block of 'cntInput' bytes.
         *
         * It is safe to pass a NULL pointer as 'input'. In this case, nothing
         * happens.
         *
         * @param input    The data to be added to the hash.
         * @param cntInput The number of bytes in 'input'.
         *
         * @throws IllegalStateException If the hash has not been initialised.
         */
        virtual void TransformBlock(const BYTE *input, const SIZE_T cntInput);

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
         *
         * @throws IllegalStateException If the hash has not been initialised.
         */
        virtual bool TransformFinalBlock(BYTE *outHash, SIZE_T& inOutSize,
            const BYTE *input, const SIZE_T cntInput);

    private:

        /** MD5 context. */
        typedef struct MD5_CTX_t {
            UINT32 state[4];    // state (ABCD)
            UINT32 count[2];    // number of bits, modulo 2^64 (lsb first)
            BYTE buffer[64];    // input buffer
        } MD5_CTX;

        /**
         * Decodes 'input' into 'output'. Assumes len is a multiple of 4.
         *
         * @param output Receives the decoded output.
         * @param input  The stream to decode.
         * @param len    The number of elements in 'input'.
         */
        static void decode(UINT32 *output, const BYTE *input, const UINT len);

        /**
         * Encodes 'input' into 'output'. Assumes 'len' is a multiple of 4.
         *
         * @param output Receives the decoded output.
         * @param input  The stream to decode.
         * @param len    The number of elements in 'input'.
         */
        static void encode(BYTE *output, const UINT32 *input, const UINT len);

        /**
         * MD5 finalization. Ends an MD5 message-digest operation, writing the
         * the message digest and zeroizing the context.
         *
         * @param output  Receives the digest.
         * @param context The context to compute the digest for and which will
         *                be erased afterwards.
         */
        static void finalise(BYTE *output, MD5_CTX *context);

        /**
         * MD5 basic transformation. Transforms 'state' based on 'block'. 
         *
         * @param state The state of the MD5 context that is to be transformed.
         * @param block The input block.
         */
        static void transform(UINT32 state[4], const BYTE block[64]);

        /**
         * MD5 block update operation. Continues an MD5 message-digest
         * operation, processing another message block, and updating the 
         * context.
         *
         * @param context  The context to work on.
         * @param input    A new block of data to be added to the hash.
         * @param cntInput The size of the input in bytes.
         */
        static void update(MD5_CTX *context, const BYTE *input, 
            const SIZE_T cntInput);

        /** The size of the MD5 hash in bytes. */
        static const SIZE_T HASH_SIZE;

        /**
         * Forbidden assignemt operator.
         *
         * @param rhs The right hand side operand.
         *
         * @return *this
         *
         * @throws IllegalParamException If this != &rhs.
         */
        MD5HashProvider& operator =(const MD5HashProvider& rhs);

        /** Context of the currently computed hash. */
        MD5_CTX context;
    };
    
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_MD5HASHPROVIDER_H_INCLUDED */
