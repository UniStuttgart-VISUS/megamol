/*
 * SystemException.h  09.08.2006 (mueller)
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#pragma once
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#include "vislib/Exception.h"
#include "vislib/sys/SystemMessage.h"
#include "vislib/types.h"


namespace vislib::sys {

/**
 * This exception class represents a system error. It is instantiated with
 * a system error code and retrieves the error message from the system.
 *
 * @author Christoph Mueller (christoph.mueller@vis.uni-stuttgart.de)
 */
class SystemException : public Exception {

public:
    /**
     * Ctor.
     *
     * @param errorCode A system dependent error code.
     * @param file      The file the exception was thrown in.
     * @param line      The line the exception was thrown in.
     */
    SystemException(const DWORD errorCode, const char* file, const int line);

    /**
     * Create a SystemException with the current error code retrieved using
     * ::GetLastError().
     *
     * @param file The file the exception was thrown in.
     * @param line The line the exception was thrown in.
     */
    SystemException(const char* file, const int line);

    /**
     * Create a clone of 'rhs'.
     *
     * @param rhs The object to be cloned.
     */
    SystemException(const SystemException& rhs);

    /** Dtor. */
    ~SystemException() override;

    /**
     * Answer the system dependent error code associated with this
     * exception.
     *
     * @return The system error code.
     */
    inline DWORD GetErrorCode() const {
        return this->sysMsg.GetErrorCode();
    }

    /**
     * Answer the file the exception description text. Behaves like
     * Exception::GetMsgA.
     *
     * @return The exception message.
     */
    const char* GetMsgA() const override;

    /**
     * Answer the file the exception description text. Behaves like
     * Exception::GetMsgW.
     *
     * @return The exception message.
     */
    const wchar_t* GetMsgW() const override;

    /**
     * Assignment operator.
     *
     * @param rhs The right hand side operand.
     *
     * @return *this.
     */
    SystemException& operator=(const SystemException& rhs);

private:
    /**
     * The system message for the error code. This member is used instead
     * of Exception::msg.
     */
    SystemMessage sysMsg;
};

} // namespace vislib::sys

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
