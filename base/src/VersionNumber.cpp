/*
 * VersionNumber.cpp
 *
 * Copyright (C) 2006 - 2007 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */


#include "vislib/VersionNumber.h"
#include "vislib/StringConverter.h"


/*
 * vislib::VersionNumber::VersionNumber
 */
vislib::VersionNumber::VersionNumber(VersionInt major, VersionInt minor, VersionInt build, VersionInt revision) 
        : major(major), minor(minor), build(build), revision(revision) {
}


/*
 * vislib::VersionNumber::VersionNumber
 */
vislib::VersionNumber::VersionNumber(const VersionNumber& rhs) : major(0), minor(0), build(0), revision(0) {
    *this = rhs;
}


/*
 * vislib::VersionNumber::VersionNumber
 */
vislib::VersionNumber::VersionNumber(const char *ver) : major(0), minor(0), build(0), revision(0) {
    this->Parse(ver);
}


/*
 * vislib::VersionNumber::VersionNumber
 */
vislib::VersionNumber::VersionNumber(const wchar_t *ver) : major(0), minor(0), build(0), revision(0) {
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
    // TODO: Implement
    return 0;
}


/*
 * vislib::VersionNumber::Parse
 */
unsigned int vislib::VersionNumber::Parse(const wchar_t *verStr) {
    // TODO: Implement
    return 0;
}


/*
 * vislib::VersionNumber::ToStringA
 */
vislib::StringA vislib::VersionNumber::ToStringA(unsigned int num) const {
    vislib::StringA tmp;
    if (num == 1) {
        tmp.Format("%u", static_cast<unsigned int>(this->major));
    } else if (num == 2) {
        tmp.Format("%u.%u", static_cast<unsigned int>(this->major), static_cast<unsigned int>(this->minor));
    } else if (num == 3) {
        tmp.Format("%u.%u.%u", static_cast<unsigned int>(this->major), static_cast<unsigned int>(this->minor)
            , static_cast<unsigned int>(this->build));
    } else if (num >= 4) {
        tmp.Format("%u.%u.%u.%u", static_cast<unsigned int>(this->major), static_cast<unsigned int>(this->minor)
            , static_cast<unsigned int>(this->build), static_cast<unsigned int>(this->revision));
    }
    return tmp;
}


/*
 * vislib::VersionNumber::ToStringW
 */
vislib::StringW vislib::VersionNumber::ToStringW(unsigned int num) const {
#ifdef _WIN32
    vislib::StringW tmp;
    if (num == 1) {
        tmp.Format(L"%u", static_cast<unsigned int>(this->major));
    } else if (num == 2) {
        tmp.Format(L"%u.%u", static_cast<unsigned int>(this->major), static_cast<unsigned int>(this->minor));
    } else if (num == 3) {
        tmp.Format(L"%u.%u.%u", static_cast<unsigned int>(this->major), static_cast<unsigned int>(this->minor)
            , static_cast<unsigned int>(this->build));
    } else if (num >= 4) {
        tmp.Format(L"%u.%u.%u.%u", static_cast<unsigned int>(this->major), static_cast<unsigned int>(this->minor)
            , static_cast<unsigned int>(this->build), static_cast<unsigned int>(this->revision));
    }
    return tmp;
#else /* _WIN32 */
    // I hate Linux
    return A2W(this->ToStringA(num));
#endif /* _WIN32 */
}


/*
 * vislib::VersionNumber::operator=
 */
vislib::VersionNumber& vislib::VersionNumber::operator=(const VersionNumber& rhs) {
    this->major = rhs.major;
    this->minor = rhs.minor;
    this->build = rhs.build;
    this->revision = rhs.revision;
    return *this;
}
