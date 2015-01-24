/*
 * AutoHandle.h
 *
 * Copyright (C) 2006 - 2007 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#ifndef VISLIB_AUTOHANDLE_H_INCLUDED
#define VISLIB_AUTOHANDLE_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */

#ifdef _WIN32
#include <windows.h>


namespace vislib {
namespace sys {


    /**
     * This is a convenience class for Windows which wraps a system handle. If
     * the object is deleted, the handle is automatically closed. Copies of the
     * object result in duplication of the handle.
     *
     * WARNING: Do not use this object for handles which must not be closed, 
     * e. g. handles retrieved from GetModuleHandle(NULL) or GetCurrentProcess().
     *
     * WARNING: The use of the HANDLE *-casts is extremely dangerous.
     */
    class AutoHandle {

    public:

        /** 
         * Creates a new handle which is initialised to be NULL if 'isNull' is
         * set true. Otherwise, it will initialised using INVALID_HANDLE_VALUE.
         *
         * @param isNull Use NULL for initialisation if true.
         */
        AutoHandle(const bool isNull = false);

        /**
         * Creates a new AutoHandle. If 'takeOwnership' is true, the object
         * takes ownership of tha handle. Otherwise, a duplicate of 'handle'
         * will be created and the caller remains owner of 'handle'.
         *
         * @param handle The handle wrapped by the object.
         */
        AutoHandle(HANDLE handle, const bool takeOwnership = true);

        /**
         * Clone 'rhs'. Note that this operation will duplicate the handle.
         *
         * @param rhs The object to be cloned.
         *
         * @throws SystemException If the handle could not be duplicated.
         */
        AutoHandle(const AutoHandle& rhs);

        /** Dtor. */
        ~AutoHandle(void);

        /**
         * Close the handle. The handle will become invalid after that.
         */
        void Close(void);

        /**
         * Answer whether the handle is invalid. Note that this checks only for
         * the handle being INVALID_HANDLE_VALUE. The handle might also be 
         * invalid if the method returns true, e. g. if it is NULL or a dangling
         * reference.
         *
         * @return true if the handle is invalid, false otherwise.
         */
        inline bool IsInvalid(void) const {
            return (this->handle == INVALID_HANDLE_VALUE);
        }

        /**
         * Answer whether the handle is NULL.
         *
         * @return true if the handle is NULL, false otherwise.
         */
        inline bool IsNull(void) const {
            return (this->handle == NULL);
        }

        /**
         * Answer whether the handle is neither invalid nor NULL. Note that the
         * handle might be useless even if the method returns true, e. g. if the
         * handle is a dangling reference.
         *
         * @return true if the handle seems to be valid, false otherwise.
         */
        inline bool IsValid(void) const {
            return (!this->IsInvalid() && !this->IsNull());
        }

        /**
         * Change the handle wrapped by the object to 'handle'. If 
         * 'takeOwnership' is true, the object takes ownership of tha handle. 
         * Otherwise, a duplicate of 'handle' will be created and the caller 
         * remains owner of 'handle'.
         *
         * Note that the currently wrapped handle will be closed before the new
         * handle is duplicated.
         *
         * It is safe to pass the handle wrapped by the object as 'handle' as
         * this will be checked before closing the current handle.
         *
         * @param handle        The new value of the handle.
         * @param takeOwnership If true, the object takes ownership of 'handle'.
         *                      If false, a duplicate will be created.
         *
         * @throws SystemException If the handle could not be duplicated.
         */
        void Set(HANDLE handle, const bool takeOwnership);

        /**
         * Assignment. The handle wrapped by 'rhs' will be duplicated into this
         * object.
         *
         * Note that the currently wrapped handle will be closed before the new
         * handle is duplicated.
         *
         * @param rhs The right hand side operand which will be duplicated.
         *
         * @return *this.
         *
         * @throws SystemException If the handle could not be duplicated.
         */
        inline AutoHandle& operator =(const AutoHandle& rhs) {
            this->Set(rhs.handle, false);
            return *this;
        }

        /**
         * Assignment. The object takes ownership of 'rhs'.
         *
         * Note that the currently wrapped handle will be closed before the new
         * handle is duplicated.
         *
         * @aram rhs The right hand side operand. The object takes ownership of 
         *           'rhs'. 
         *
         * @return *this.
         */
        inline AutoHandle& operator =(HANDLE rhs) {
            this->Set(rhs, true);
            return *this;
        }

        /**
         * Test for equality.
         *
         * @param rhs The right hand side operand.
         * 
         * @return true if this object and 'rhs' are equal, false otherwise.
         */
        inline bool operator ==(const AutoHandle& rhs) const {
            return (this->handle == rhs.handle);
        }

        /**
         * Test for equality.
         *
         * @param rhs The right hand side operand.
         * 
         * @return true if this object and 'rhs' are equal, false otherwise.
         */
        inline bool operator ==(const HANDLE rhs) const {
            return (this->handle == rhs);
        }

        /**
         * Test for inequality.
         *
         * @param rhs The right hand side operand.
         * 
         * @return true if this object and 'rhs' are not equal, false otherwise.
         */
        inline bool operator !=(const AutoHandle& rhs) const {
            return (this->handle != rhs.handle);
        }

        /**
         * Test for inequality.
         *
         * @param rhs The right hand side operand.
         * 
         * @return true if this object and 'rhs' are not equal, false otherwise.
         */
        inline bool operator !=(const HANDLE rhs) const {
            return (this->handle != rhs);
        }

        /**
         * Cast to HANDLE. Note that the object remains owner of the HANDLE.
         *
         * @return The handle wrapped by the object.
         */
        inline operator HANDLE(void) {
            return this->handle;
        }

        /**
         * Cast to HANDLE. Note that the object remains owner of the HANDLE.
         *
         * @return The handle wrapped by the object.
         */
        inline operator const HANDLE(void) const {
            return this->handle;
        }

        /**
         * Cast to HANDLE *. Note that the object remains owner of the handle
         * designated by the returned pointer.
         *
         * Warning: This operator is extremely dangerous! If you assign any 
         * value to the pointer returned, the object takes ownership of this 
         * handle. You should ensure that you have called Close() before 
         * assigning any value to the objected designatedby the returned 
         * pointer.
         *
         * @return A pointer to the wrapped handle.
         */
        inline operator HANDLE *(void) {
            return &(this->handle);
        }

        /**
         * Cast to HANDLE *. Note that the object remains owner of the handle
         * designated by the returned pointer.
         *
         * Warning: This operator is extremely dangerous! If you assign any 
         * value to the pointer returned, the object takes ownership of this 
         * handle. You should ensure that you have called Close() before 
         * assigning any value to the objected designatedby the returned 
         * pointer.
         *
         * @return A pointer to the wrapped handle.
         */
        inline operator const HANDLE *(void) const {
            return &(this->handle);
        }

    private:

        /** The handle wrapped by this object. */
        HANDLE handle;

    };
    
} /* end namespace sys */
} /* end namespace vislib */
#endif /* _WIN32 */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_AUTOHANDLE_H_INCLUDED */
