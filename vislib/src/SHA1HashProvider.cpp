/*
 * SHA1HashProvider.cpp
 *
 * Copyright (C) 2006 - 2007 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#include "vislib/SHA1HashProvider.h"

#include "vislib/assert.h"
#include "vislib/IllegalParamException.h"
#include "vislib/IllegalStateException.h"
#include "vislib/memutils.h"


/* Define the SHA1 circular left shift macro. */
#define SHA1CircularShift(bits, word) \
    (((word) << (bits)) | ((word) >> (32 - (bits))))



/*
 * vislib::SHA1HashProvider::SHA1HashProvider
 */
vislib::SHA1HashProvider::SHA1HashProvider(void) {
    this->Initialise();
}


/*
 * vislib::SHA1HashProvider::~SHA1HashProvider
 */
vislib::SHA1HashProvider::~SHA1HashProvider(void) {
    ::SecureZeroMemory(&this->context, sizeof(SHA1Context));
}


/*
 * vislib::SHA1HashProvider::Initialise
 */
void vislib::SHA1HashProvider::Initialise(void) {
    this->context.Length_Low = 0;
    this->context.Length_High = 0;
    this->context.Message_Block_Index = 0;

    this->context.Intermediate_Hash[0] = 0x67452301;
    this->context.Intermediate_Hash[1] = 0xEFCDAB89;
    this->context.Intermediate_Hash[2] = 0x98BADCFE;
    this->context.Intermediate_Hash[3] = 0x10325476;
    this->context.Intermediate_Hash[4] = 0xC3D2E1F0;

    this->context.Computed = 0;
    this->context.Corrupted = 0;
}


/*
 * vislib::SHA1HashProvider::TransformBlock
 */
void vislib::SHA1HashProvider::TransformBlock(const BYTE *input, 
                                              const SIZE_T cntInput) {
    // Must be initialised, as ctor does this.
    if ((input != NULL) && (cntInput > 0)) {
        SHA1HashProvider::input(&this->context, input, 
            static_cast<UINT>(cntInput));
    }
}


/*
 * vislib::SHA1HashProvider::TransformFinalBlock
 */
bool vislib::SHA1HashProvider::TransformFinalBlock(BYTE *outHash, 
        SIZE_T& inOutSize, const BYTE *input, const SIZE_T cntInput) {
    SHA1Context ctx = this->context;    // Local context to be finalised.
    bool retval = false;                // Remember whether output was copied.

    /* Transform final block before finalising. */
    this->TransformBlock(input, cntInput);

    /* Finalise and remember that already finalised. */
    if ((outHash != NULL) && (inOutSize >= SHA1HashProvider::HASH_SIZE)) {
        SHA1HashProvider::result(outHash, &ctx);
        retval = true;
    }

    inOutSize = SHA1HashProvider::HASH_SIZE;
    return retval;
}


/*
 * vislib::SHA1HashProvider::input
 */
void vislib::SHA1HashProvider::input(SHA1Context *context, const BYTE *input, 
        const UINT cntInput) {
    ASSERT(context != NULL);
    ASSERT(input != NULL);

    UINT length = cntInput;

    if (!length) {
        /* Nothing to do. */
        return;
    }

    // vislib enforces this.
    //if (!context || !input) {
    //    return shaNull;
    //}

    if (context->Computed) {
        context->Corrupted = shaStateError;
        throw IllegalStateException("vislib::SHA1HashProvider::input called "
            "after hash was already computed.", __FILE__, __LINE__);
    }

    if (context->Corrupted) {
         throw IllegalStateException("SHA1Context is corrupted.", __FILE__, 
             __LINE__);
    }

    while (length-- && !context->Corrupted) {
        context->Message_Block[context->Message_Block_Index++] 
            = (*input & 0xFF);

        context->Length_Low += 8;
        if (context->Length_Low == 0) {
            context->Length_High++;
        
            if (context->Length_High == 0) {
                /* Message is too long */
                context->Corrupted = 1;
            }
        }

        if (context->Message_Block_Index == 64) {
            SHA1HashProvider::processMessageBlock(context);
        }

        input++;
    }
}


/*
 * vislib::SHA1HashProvider::padMessage
 */
void vislib::SHA1HashProvider::padMessage(SHA1Context *context) {
    /*
     * Check to see if the current message block is too small to hold
     * the initial padding bits and length. If so, we will pad the
     * block, process it, and then continue padding into a second
     * block.
     */
    if (context->Message_Block_Index > 55) {
        context->Message_Block[context->Message_Block_Index++] = 0x80;
        
        while(context->Message_Block_Index < 64) {
            context->Message_Block[context->Message_Block_Index++] = 0;
        }

        SHA1HashProvider::processMessageBlock(context);

        while(context->Message_Block_Index < 56) {
            context->Message_Block[context->Message_Block_Index++] = 0;
        }

    } else {
        context->Message_Block[context->Message_Block_Index++] = 0x80;
        
        while(context->Message_Block_Index < 56) {
            context->Message_Block[context->Message_Block_Index++] = 0;
        }
    }

    /* Store the message length as the last 8 octets. */
    context->Message_Block[56] = context->Length_High >> 24;
    context->Message_Block[57] = context->Length_High >> 16;
    context->Message_Block[58] = context->Length_High >> 8;
    context->Message_Block[59] = context->Length_High;
    context->Message_Block[60] = context->Length_Low >> 24;
    context->Message_Block[61] = context->Length_Low >> 16;
    context->Message_Block[62] = context->Length_Low >> 8;
    context->Message_Block[63] = context->Length_Low;

    SHA1HashProvider::processMessageBlock(context);
}


/*
 * vislib::SHA1HashProvider::processMessageBlock
 */
void vislib::SHA1HashProvider::processMessageBlock(SHA1Context *context) {
    const UINT32 K[] = {    // Constants defined in SHA-1
        0x5A827999, 0x6ED9EBA1, 0x8F1BBCDC, 0xCA62C1D6 };
    UINT32 temp;            // Temporary word value
    UINT32 W[80];           // Word sequence
    UINT32 A, B, C, D, E;   // Word buffers 

    /* Initialize the first 16 words in the array W */
    for (int t = 0; t < 16; t++) {
        W[t] = context->Message_Block[t * 4] << 24;
        W[t] |= context->Message_Block[t * 4 + 1] << 16;
        W[t] |= context->Message_Block[t * 4 + 2] << 8;
        W[t] |= context->Message_Block[t * 4 + 3];
    }

    for (int t = 16; t < 80; t++) {
       W[t] = SHA1CircularShift(1, W[t-3] ^ W[t-8] ^ W[t-14] ^ W[t-16]);
    }

    A = context->Intermediate_Hash[0];
    B = context->Intermediate_Hash[1];
    C = context->Intermediate_Hash[2];
    D = context->Intermediate_Hash[3];
    E = context->Intermediate_Hash[4];

    for (int t = 0; t < 20; t++) {
        temp =  SHA1CircularShift(5, A) + ((B & C) | ((~B) & D)) + E + W[t] 
            + K[0];
        E = D;
        D = C;
        C = SHA1CircularShift(30,B);

        B = A;
        A = temp;
    }

    for (int t = 20; t < 40; t++) {
        temp = SHA1CircularShift(5, A) + (B ^ C ^ D) + E + W[t] + K[1];
        E = D;
        D = C;
        C = SHA1CircularShift(30, B);
        B = A;
        A = temp;
    }

    for (int t = 40; t < 60; t++) {
        temp = SHA1CircularShift(5, A) + ((B & C) | (B & D) | (C & D)) + E 
            + W[t] + K[2];
        E = D;
        D = C;
        C = SHA1CircularShift(30, B);
        B = A;
        A = temp;
    }

    for (int t = 60; t < 80; t++) {
        temp = SHA1CircularShift(5, A) + (B ^ C ^ D) + E + W[t] + K[3];
        E = D;
        D = C;
        C = SHA1CircularShift(30, B);
        B = A;
        A = temp;
    }

    context->Intermediate_Hash[0] += A;
    context->Intermediate_Hash[1] += B;
    context->Intermediate_Hash[2] += C;
    context->Intermediate_Hash[3] += D;
    context->Intermediate_Hash[4] += E;

    context->Message_Block_Index = 0;
}


/*
 * vislib::SHA1HashProvider::result
 */
void vislib::SHA1HashProvider::result(UINT8 *messageDigest, 
                                      SHA1Context *context) {
    ASSERT(messageDigest != NULL);
    ASSERT(context != NULL);

    // vislib enforces this.
    //if (!context || !messageDigest) {
    //    return shaNull;
    //}

    if (context->Corrupted) {
         throw IllegalStateException("SHA1Context is corrupted.", __FILE__, 
             __LINE__);
    }

    if (!context->Computed) {
        SHA1HashProvider::padMessage(context);
        for (int i = 0; i < 64; i++){
            /* message may be sensitive, clear it out */
            context->Message_Block[i] = 0;
        }
        context->Length_Low = 0;    /* and clear length */
        context->Length_High = 0;
        context->Computed = 1;
    }

    for (UINT i = 0; i < SHA1HashProvider::HASH_SIZE; i++) {
        messageDigest[i] = context->Intermediate_Hash[i >> 2] >> 8 
            * (3 - (i & 0x03));
    }
}


/*
 * vislib::SHA1HashProvider::operator =
 */
vislib::SHA1HashProvider& vislib::SHA1HashProvider::operator =(
        const SHA1HashProvider& rhs) {
    if (this != &rhs) {
        throw IllegalParamException("rhs", __FILE__, __LINE__);
    }

    return *this;
}
