/*
 * OutOfRangeException.cpp
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 * Copyright (C) 2005 by Christoph Mueller. All rights reserved.
 */


#include "vislib/OutOfRangeException.h"


/*
 * vislib::OutOfRangeException::OutOfRangeException
 */
vislib::OutOfRangeException::OutOfRangeException(const int val, 
        const int minVal, const int maxVal, const char *file, const int line)
        : Exception(file, line) {
	Exception::formatMsg(_T("%d is not within [%d, %d]."), val, minVal, maxVal);
}


/*
 * vislib::OutOfRangeException::OutOfRangeException
 */
vislib::OutOfRangeException::OutOfRangeException(
		const OutOfRangeException& rhs) 
		: Exception(rhs) {
}


/*
 * vislib::OutOfRangeException::~OutOfRangeException
 */
vislib::OutOfRangeException::~OutOfRangeException(void) {
}


/*
 * vislib::OutOfRangeException::operator =
 */
vislib::OutOfRangeException& vislib::OutOfRangeException::operator =(
		const OutOfRangeException& rhs) {
	Exception::operator =(rhs);
	return *this;
}

