/*
 * VersionNumber.cpp
 *
 * Copyright (C) 2006 - 2007 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */


#include "vislib/VersionNumber.h"
#include "the/text/string_builder.h"
#include "the/text/string_converter.h"


/*
 * vislib::VersionNumber::VersionNumber
 */
vislib::VersionNumber::VersionNumber(VersionInt majorNumber, 
        VersionInt minorNumber, VersionInt buildNumber, 
        VersionInt revisionNumber) 
        : majorNumber(majorNumber), minorNumber(minorNumber), 
        buildNumber(buildNumber), revisionNumber(revisionNumber) {
}


/*
 * vislib::VersionNumber::VersionNumber
 */
vislib::VersionNumber::VersionNumber(const VersionNumber& rhs) 
        : majorNumber(0), minorNumber(0), buildNumber(0), revisionNumber(0) {
    *this = rhs;
}


/*
 * vislib::VersionNumber::VersionNumber
 */
vislib::VersionNumber::VersionNumber(const char *ver) 
        : majorNumber(0), minorNumber(0), buildNumber(0), revisionNumber(0) {
    this->Parse(ver);
}


/*
 * vislib::VersionNumber::VersionNumber
 */
vislib::VersionNumber::VersionNumber(const wchar_t *ver) 
        : majorNumber(0), minorNumber(0), buildNumber(0), revisionNumber(0) {
    this->Parse(ver);
}


/*
 * vislib::VersionNumber::~VersionNumber
 */
vislib::VersionNumber::~VersionNumber(void) {
}


/*
 * vislib::VersionNumber::Parse
 */
unsigned int vislib::VersionNumber::Parse(const char *verStr) {
    int retval = 0;
    int value = 0;
    int charValue;

    this->majorNumber = 0;
    this->minorNumber = 0;
    this->buildNumber = 0;
    this->revisionNumber = 0;

    for (int state = 1; state > 0; verStr++) {
        charValue = static_cast<int>(*verStr) - static_cast<int>('0');
        switch (state) {
            case 1: // first character of a number
                if ((charValue < 0) || (charValue > 9)) {
                    state = 0; // unexpected; expecting a digit. 
                } else {
                    value = charValue;
                    state = 2;
                }
                break;
            case 2: // later character of a number
                if ((charValue < 0) || (charValue > 9)) {
                    switch (retval) {
                        case 0: this->majorNumber = value; break;
                        case 1: this->minorNumber = value; break;
                        case 2: this->buildNumber = value; break;
                        case 3: this->revisionNumber = value; break;
                    }
                    retval++;
                    if (retval > 3) {
                        retval = 4;
                        state = 0; // finished.
                    } else {
                        if ((*verStr == '.') || (*verStr == ',')) {
                            state = 1; // valid separator
                        } else {
                            state = 0; // end of valid string
                        }
                    }
                } else {
                    value = 10 * value + charValue;
                }
                break;
            default: // something went really wrong!
                state = 0;
                break;
        }
    }

    return retval;
}


/*
 * vislib::VersionNumber::Parse
 */
unsigned int vislib::VersionNumber::Parse(const wchar_t *verStr) {
    int retval = 0;
    int value = 0;
    int charValue;

    this->majorNumber = 0;
    this->minorNumber = 0;
    this->buildNumber = 0;
    this->revisionNumber = 0;

    for (int state = 1; state > 0; verStr++) {
        charValue = static_cast<int>(*verStr) - static_cast<int>(L'0');
        switch (state) {
            case 1: // first character of a number
                if ((charValue < 0) || (charValue > 9)) {
                    state = 0; // unexpected; expecting a digit. 
                } else {
                    value = charValue;
                    state = 2;
                }
                break;
            case 2: // later character of a number
                if ((charValue < 0) || (charValue > 9)) {
                    switch (retval) {
                        case 0: this->majorNumber = value; break;
                        case 1: this->minorNumber = value; break;
                        case 2: this->buildNumber = value; break;
                        case 3: this->revisionNumber = value; break;
                    }
                    retval++;
                    if (retval > 3) {
                        retval = 4;
                        state = 0; // finished.
                    } else {
                        if ((*verStr == L'.') || (*verStr == L',')) {
                            state = 1; // valid separator
                        } else {
                            state = 0; // end of valid string
                        }
                    }
                } else {
                    value = 10 * value + charValue;
                }
                break;
            default: // something went really wrong!
                state = 0;
                break;
        }
    }

    return retval;
}


/*
 * vislib::VersionNumber::ToStringA
 */
the::astring vislib::VersionNumber::ToStringA(unsigned int num) const {
    the::astring tmp;
    if (num == 0) {
        num = 1;
        if (this->minorNumber > 0) num = 2;
        if (this->buildNumber > 0) num = 3;
        if (this->revisionNumber > 0) num = 4;
    }
    if (num == 1) {
        the::text::astring_builder::format_to(tmp, "%u", static_cast<unsigned int>(this->majorNumber));
    } else if (num == 2) {
        the::text::astring_builder::format_to(tmp, "%u.%u", static_cast<unsigned int>(this->majorNumber), 
            static_cast<unsigned int>(this->minorNumber));
    } else if (num == 3) {
        the::text::astring_builder::format_to(tmp, "%u.%u.%u", static_cast<unsigned int>(this->majorNumber), 
            static_cast<unsigned int>(this->minorNumber),
            static_cast<unsigned int>(this->buildNumber));
    } else if (num >= 4) {
        the::text::astring_builder::format_to(tmp, "%u.%u.%u.%u", static_cast<unsigned int>(this->majorNumber),
            static_cast<unsigned int>(this->minorNumber), 
            static_cast<unsigned int>(this->buildNumber), 
            static_cast<unsigned int>(this->revisionNumber));
    }
    return tmp;
}


/*
 * vislib::VersionNumber::ToStringW
 */
the::wstring vislib::VersionNumber::ToStringW(unsigned int num) const {
#ifdef _WIN32
    the::wstring tmp;
    if (num == 0) {
        num = 1;
        if (this->minorNumber > 0) num = 2;
        if (this->buildNumber > 0) num = 3;
        if (this->revisionNumber > 0) num = 4;
    }
    if (num == 1) {
        the::text::wstring_builder::format_to(tmp, L"%u", static_cast<unsigned int>(this->majorNumber));
    } else if (num == 2) {
        the::text::wstring_builder::format_to(tmp, L"%u.%u", static_cast<unsigned int>(this->majorNumber), 
            static_cast<unsigned int>(this->minorNumber));
    } else if (num == 3) {
        the::text::wstring_builder::format_to(tmp, L"%u.%u.%u", static_cast<unsigned int>(this->majorNumber),
            static_cast<unsigned int>(this->minorNumber), 
            static_cast<unsigned int>(this->buildNumber));
    } else if (num >= 4) {
        the::text::wstring_builder::format_to(tmp, L"%u.%u.%u.%u", 
            static_cast<unsigned int>(this->majorNumber), 
            static_cast<unsigned int>(this->minorNumber),
            static_cast<unsigned int>(this->buildNumber),
            static_cast<unsigned int>(this->revisionNumber));
    }
    return tmp;
#else /* _WIN32 */
    // I hate Linux
    return THE_A2W(this->ToStringA(num));
#endif /* _WIN32 */
}


/*
 * vislib::VersionNumber::operator=
 */
vislib::VersionNumber& vislib::VersionNumber::operator=(const VersionNumber& rhs) {
    this->majorNumber = rhs.majorNumber;
    this->minorNumber = rhs.minorNumber;
    this->buildNumber = rhs.buildNumber;
    this->revisionNumber = rhs.revisionNumber;
    return *this;
}
