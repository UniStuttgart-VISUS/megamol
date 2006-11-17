/*
 * sysfunctions.cpp
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */


#include "vislib/sysfunctions.h"

#ifdef _WIN32
#else /* _WIN32 */
#include <unistd.h>
#endif /* _WIN32 */

#include "vislib/memutils.h"
#include "vislib/IllegalParamException.h"

#include "vislib/IOException.h"


/*
 * vislib::sys::ReadLineFromFileA
 */
vislib::StringA vislib::sys::ReadLineFromFileA(File& input, unsigned int size) {
    char *buf = new char[size + 1];
    unsigned int pos;

    try {
        for (pos = 0; pos < size; pos++) {
            if (input.Read(&buf[pos], sizeof(char)) != sizeof(char)) {
                // almost sure end of file
                break;
            }
            if ((buf[pos] == '\n') || (buf[pos] == '\r')) {
                // line break
                if (buf[pos] == '\r') {
                    // \n might follow
                    if (input.Read(&buf[pos + 1], sizeof(char)) != sizeof(char)) {
                        // and almost sure end of file
                        break;
                    }
                    if (buf[pos + 1] != '\n') {
                        // no \n so better do an ungetc
                        input.Seek(-int(sizeof(char)), vislib::sys::File::CURRENT);
                    }
                }
                break;
            }
        }
        buf[pos] = '\0';

    } catch(IOException e) {
        ARY_SAFE_DELETE(buf);
        throw IOException(e);
    } catch(Exception e) {
        ARY_SAFE_DELETE(buf);
        throw Exception(e);
    } catch(...) {
        ARY_SAFE_DELETE(buf);
        throw Exception("Unexcepted exception", __FILE__, __LINE__);
    }
    StringA str(buf);
    delete[] buf;
    return str;
}


/*
 * vislib::sys::ReadLineFromFileW
 */
vislib::StringW vislib::sys::ReadLineFromFileW(File& input, unsigned int size) {
    wchar_t *buf = new wchar_t[size + 1];
    unsigned int pos;

    try {
        for (pos = 0; pos < size; pos++) {
            if (input.Read(&buf[pos], sizeof(wchar_t)) != sizeof(wchar_t)) {
                // almost sure end of file
                break;
            }
            if ((buf[pos] == L'\n') || (buf[pos] == L'\r')) {
                // line break
                if (buf[pos] == L'\r') {
                    // \n might follow
                    if (input.Read(&buf[pos + 1], sizeof(wchar_t)) != sizeof(wchar_t)) {
                        // and almost sure end of file
                        break;
                    }
                    if (buf[pos + 1] != L'\n') {
                        // no \n so better do an ungetc
                        input.Seek(-int(sizeof(wchar_t)), vislib::sys::File::CURRENT);
                    }
                }
                break;
            }
        }
        buf[pos] = L'\0';

    } catch(IOException e) {
        ARY_SAFE_DELETE(buf);
        throw IOException(e);
    } catch(Exception e) {
        ARY_SAFE_DELETE(buf);
        throw Exception(e);
    } catch(...) {
        ARY_SAFE_DELETE(buf);
        throw Exception("Unexcepted exception", __FILE__, __LINE__);
    }
    StringW str(buf);
    delete[] buf;
    return str;
}
