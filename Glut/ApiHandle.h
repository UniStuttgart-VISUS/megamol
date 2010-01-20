/*
 * ApiHandle.h
 *
 * Copyright (C) 2006 - 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLVIEWER_APIHANDLE_H_INCLUDED
#define MEGAMOLVIEWER_APIHANDLE_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */


namespace megamol {
namespace viewer {

    /** 
     * Base class for core handles 
     */
    class ApiHandle {
    public:
        /**
         * Returns the size of a megaMolHandle data struct in bytes.
         *
         * @return The size of a megaMolHandle data struct in bytes.
         */
        static inline unsigned int GetHandleSize(void);

        /**
         * Creates a megaMolHandle data structure at the memory block hndl 
         * points to and assigns the given object to the handle.
         *
         * @param hndl Pointer to the memory block.
         * @param obj Object to be assigned to the handle. The object must be
         *            created with the std new.
         *
         * @return true if success, false if hndl already holds an valid 
         *         handle.
         */
        static inline bool CreateHandle(void *hndl, ApiHandle* obj);

        /**
         * Tries to destroy a core handle.
         *
         * @param hndl Pointer to the memory block possibly holding a handle to
         *             be destroyed.
         */
        static inline void DestroyHandle(void *hndl);

        /**
         * Tries to interpret a memory block as core handle of template type.
         *
         * @param hndl Pointer to the memory block possibly holding a handle.
         *
         * @return Pointer to the Handle object or NULL if the memory block
         *         could not be interpreted.
         */
        template<class T> static inline T * InterpretHandle(void *hndl);

        /** default ctor */
        ApiHandle(void);

        /** dtor */
        virtual ~ApiHandle(void);

        /** The user data pointer */
        void *UserData;

    private:

        /** the size of the uuid string in characters. */
        static const int uuidSize = 16;

        /** uuid used to identify the megaMolHandle structure */
        static const unsigned char magicMegaMolUUID[uuidSize];

        /** the megaMolEntryHandle structure */
        struct megaMolHandle {
            unsigned char uuid[uuidSize];
            class ApiHandle *obj;
        };

    };


    /*
     * ApiHandle::GetHandleSize
     */
    unsigned int ApiHandle::GetHandleSize(void) {
        return sizeof(struct megaMolHandle);
    }


    /*
     * ApiHandle::CreateHandle
     */
    bool ApiHandle::CreateHandle(void *hndl, ApiHandle* obj) {
        if (hndl == NULL) return false;
        struct megaMolHandle *s = static_cast<struct megaMolHandle *>(hndl);
        if (memcmp(s->uuid, magicMegaMolUUID, sizeof(magicMegaMolUUID)) != 0) {
            memcpy(s->uuid, magicMegaMolUUID, sizeof(magicMegaMolUUID));
            s->obj = obj;
        } else {
            delete obj;
            return false; // already entry handle present
        }
        return true;
    }


    /*
     * ApiHandle::DestroyHandle
     */
    void ApiHandle::DestroyHandle(void *hndl) {
        if (hndl == NULL) return;
        struct megaMolHandle *s = static_cast<struct megaMolHandle *>(hndl);
        if (memcmp(s->uuid, magicMegaMolUUID, sizeof(magicMegaMolUUID)) == 0) {
            delete s->obj;
            memset(hndl, 0, sizeof(struct megaMolHandle));
        }
    }


    /*
     * ApiHandle::InterpretHandle
     */
    template<class T> T * ApiHandle::InterpretHandle(void *hndl) {
        if (hndl == NULL) return NULL;
        struct megaMolHandle *s = static_cast<struct megaMolHandle *>(hndl);
        return (memcmp(s->uuid, magicMegaMolUUID, sizeof(magicMegaMolUUID)) == 0) 
            ? dynamic_cast<T *>(s->obj) : NULL;
    }

} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLVIEWER_APIHANDLE_H_INCLUDED */
