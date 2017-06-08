/*
 * IllegalParamException.cpp
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 * Copyright (C) 2005 by Christoph Mueller (christoph.mueller@vis.uni-stuttgart.de). Alle Rechte vorbehalten.
 */


#include "vislib/IllegalParamException.h"


/*
 * vislib::IllegalParamException::IllegalParamException
 */
vislib::IllegalParamException::IllegalParamException(const char *paramName,
        const char *file, const int line) 
        : Exception(file, line) {
	Exception::formatMsg("Parameter '%s' has an invalid value.", paramName);
}


/*
 * vislib::IllegalParamException::IllegalParamException
 */
vislib::IllegalParamException::IllegalParamException(const wchar_t *paramName,
        const char *file, const int line) 
        : Exception(file, line) {
	Exception::formatMsg(L"Parameter '%s' has an invalid value.", paramName);
}


/*
 * vislib::IllegalParamException::IllegalParamException
 */
vislib::IllegalParamException::IllegalParamException(
		const IllegalParamException& rhs) 
		: Exception(rhs) {
}


/*
 * vislib::IllegalParamException::~IllegalParamException
 */
vislib::IllegalParamException::~IllegalParamException(void) {
}


/*
 * vislib::IllegalParamException::operator =
 */
vislib::IllegalParamException& vislib::IllegalParamException::operator =(
		const IllegalParamException& rhs) {
	Exception::operator =(rhs);
	return *this;
}

