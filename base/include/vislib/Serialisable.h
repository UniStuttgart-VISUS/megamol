/*
 * Serialisable.h
 *
 * Copyright (C) 2006 - 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#ifndef VISLIB_SERIALISABLE_H_INCLUDED
#define VISLIB_SERIALISABLE_H_INCLUDED
#if (_MSC_VER > 1000)
#pragma once
#endif /* (_MSC_VER > 1000) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#include "vislib/types.h"


namespace vislib {


    /**
     * This is the interface that classes must implement for VISlib binary
     * serialisation mechansims.
     */
    class Serialisable {

    public:

        /** Dtor. */
        virtual ~Serialisable(void);

        /**
         * The implementing class must return the size in bytes of the 
         * serialised data that are accessed via GetSerialisationData().
         *
         * @return The size of the serialised data in bytes.
         */
        virtual SIZE_T GetSerialisationSize(void) const = 0;

        /**
         * Answer a pointer to the serialised data. This must be a block
         * of GetSerialisationSize() bytes.
         *
         * The callee remains owner of the data designated by the returned
         * pointer.
         *
         * @return A pointer to the serialised data.
         */
        virtual const BYTE *GetSerialisationData(void) const = 0;

        /**
         * Answer a pointer to the serialised data. The 'minSize' parameter
         * is used to specify the minimum size of the returned data block
         * for deserialisation. 
         *
         * Implementing subclasses must provide the following behaviour:
         * If 'minSize' is 0, the method must behave exactly like
         * GetSerialisationData(void) and return the pointer to the serialised
         * data of GetSerialisationSize() bytes.
         * If 'minSize' is not 0, the method must allocate at least 'minSize'
         * bytes for receiving serialised data. The returned pointer needs not
         * to designate meaningful data in this case. The callee might return
         * NULL if it cannot provide 'minSize' bytes.
         *
         * @param minSize Indicates that the standard serialised data should be
         *                returned if 0, otherwise, the returned pointer must
         *                designate a block of at least 'minSize' bytes.
         *
         * @return A pointer to the serialised data if 'minSize' == 0,
         *         a pointer to at least 'minSize' bytes to receive 
         *         serialisation content otherwise,
         *         and NULL if 'minSize' bytes cannot not be provided.
         */
        virtual BYTE *GetSerialisationData(const SIZE_T minSize = 0) = 0;

        /**
         * Serialises call this method after writing data to a pointer retrieved
         * by GetSerialisationData(const SIZE_T). The pointer 'data' is the one
         * that was previously returned by GetSerialisationData(const SIZE_T)
         * and designates the serialisation data written by the deserialiser.
         * This method gives implementing classes the possibility to post-process
         * serialisation data.
         *
         * Serialisers must guarantee that no call to other methods of the 
         * Serialisable interface is made between a call to 
         * GetSerialisationData(const SIZE_T) and to this method.
         *
         * @param data The pointer that the serialiser received from 
         *             GetSerialisationData(const SIZE_T).
         *
         * @return false if the serialisation data could not be interpreted, 
         *         true otherwise (this is the default implementation).
         */
        virtual bool OnSerialisationDataReceived(BYTE *data);

    protected:

        /** Ctor. */
        inline Serialisable(void) {}

        /**
         * Clone 'rhs'.
         *
         * @param rhs The object to be cloned.
         */
        inline Serialisable(const Serialisable& rhs) {}

        /**
         * Assignment operator.
         *
         * @param rhs The right hand side operand.
         *
         * @return *this.
         */
        inline Serialisable& operator = (const Serialisable& rhs) {
            return *this;
        }
    };
    
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_SERIALISABLE_H_INCLUDED */
