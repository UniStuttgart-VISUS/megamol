/*
 * SHA1HashProvider.h
 *
 * Copyright (C) 2006 - 2007 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#ifndef VISLIB_SHA1HASHPROVIDER_H_INCLUDED
#define VISLIB_SHA1HASHPROVIDER_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#include "vislib/HashAlgorithm.h"


namespace vislib {


    /**
     * This class provides message digests according to the the SHA-1 
     * (Secure Hash Algorithm 1) as defined in RFC 3174. It is intended
     * for input with a length of less than 2^64 bits.
     *
     * This implementation is derived from the reference implementation in the
     * RFC 3174 stanard.
     */
    class SHA1HashProvider : public HashAlgorithm {

    public:

        /** Ctor. */
        SHA1HashProvider(void);

        /** Dtor. */
        virtual ~SHA1HashProvider(void);

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

        /** The size of the SHA-1 in bytes. */
        static const SIZE_T HASH_SIZE = 20;

        /**
         * This structure will hold context information for the SHA-1
         * hashing operation.
         */
        typedef struct SHA1Context_t {
            UINT32 Intermediate_Hash[HASH_SIZE / 4];    // Message Digest

            UINT32 Length_Low;          // Message length in bits
            UINT32 Length_High;         // Message length in bits

            UINT16 Message_Block_Index; // Index into message block array
            BYTE Message_Block[64];     // 512-bit message blocks

            int Computed;               // Is the digest computed?
            int Corrupted;              // Is the message digest corrupted?
        } SHA1Context;

        /** Sha state errors. */
        enum {
            shaSuccess = 0,
            shaNull,                    // Null pointer parameter
            shaInputTooLong,            // input data too long
            shaStateError               // called Input after Result
        };

        /**
         * This function accepts an array of octets as the next portion of the 
         * message.
         *
         * @param context   The SHA context to update.
         * @param input     An array of characters representing the next 
         *                  portion of the message.
         * @param cntInput  The length of the message in message_array
         *
         * @throws IllegalStateException If the hash designated by 'context' was
         *                               already computed,
         *                               If the context was corrupted.
         */
        void input(SHA1Context *context, const BYTE *input, 
            const UINT cntInput);

        /**
         * According to the standard, the message must be padded to an even
         * 512 bits. The first padding bit must be a '1'. The last 64 bits 
         * represent the length of the original message. All bits in between 
         * should be 0. This function will pad the message according to those 
         * rules by filling the Message_Block array accordingly. It will also 
         * call the ProcessMessageBlock function provided appropriately.  When 
         * it returns, it can be assumed that the message digest has been 
         * computed.
         *
         * @param context The context to pad
         */
        static void padMessage(SHA1Context *context);

        /*
         * This function will process the next 512 bits of the message stored in 
         * the Message_Block array.
         *
         * @param context The context to use to calculate the SHA-1 hash.
         */
        static void processMessageBlock(SHA1Context *context);

        /**
         * This function will return the 160-bit message digest into the 
         * 'messageDigest' array  provided by the caller.
         * NOTE: The first octet of hash is stored in the 0th element, the last 
         * octet of hash in the 19th element.
         *
         * @param messageDigest Where the digest is returned.
         * @param context       The context to use to calculate the SHA-1 hash.
         *
         * @throws IllegalStateException If the context was corrupted.
         */
        static void result(UINT8 *messageDigest, SHA1Context *context);

        /**
         * Forbidden assignemt operator.
         *
         * @param rhs The right hand side operand.
         *
         * @return *this
         *
         * @throws IllegalParamException If this != &rhs.
         */
        SHA1HashProvider& operator =(const SHA1HashProvider& rhs);

        /** The context used for the hashing operations. */
        SHA1Context context;

    };
    
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_SHA1HASHPROVIDER_H_INCLUDED */
