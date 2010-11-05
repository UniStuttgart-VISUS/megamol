/*
 * sysfunctions.cpp
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */


#include "vislib/sysfunctions.h"

#ifdef _WIN32
#else /* _WIN32 */
#include <unistd.h>
#include <time.h>
#include <sys/ipc.h>
#include <sys/time.h>
#include <climits>
#endif /* _WIN32 */

#include "vislib/assert.h"
#include "vislib/File.h"
#include "vislib/IllegalParamException.h"
#include "vislib/IOException.h"
#include "vislib/memutils.h"
#include "vislib/Path.h"
#include "vislib/String.h"
#include "vislib/StringConverter.h"
#include "vislib/SystemException.h"
#include "vislib/Trace.h"


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


/*
 * vislib::sys::ReadTextFile
 */
bool vislib::sys::ReadTextFile(vislib::StringA& outStr, const char *filename) {
    File file;
    if (!file.Open(filename, File::READ_ONLY, File::SHARE_READ, 
            File::OPEN_ONLY)) {
        VLTRACE(Trace::LEVEL_ERROR, "Text file \"%s\" could not be opened.", 
            filename);
        return false;
    }
    return ReadTextFile(outStr, file);
}


/*
 * vislib::sys::ReadTextFile
 */
bool vislib::sys::ReadTextFile(vislib::StringA& outStr, const wchar_t *filename) {
    File file;
    if (!file.Open(filename, File::READ_ONLY, File::SHARE_READ, 
            File::OPEN_ONLY)) {
        VLTRACE(Trace::LEVEL_ERROR, "Text file \"%s\" could not be opened.", 
            filename);
        return false;
    }
    return ReadTextFile(outStr, file);
}


/*
 * vislib::sys::ReadTextFile
 */
bool vislib::sys::ReadTextFile(vislib::StringA& outStr, 
        vislib::sys::File& file) {
    File::FileSize size; // Size of the remainig file in bytes.
    char *src = NULL;    // Array to hold source.

    size = file.GetSize() - file.Tell();
    ASSERT(size < INT_MAX);
    src = outStr.AllocateBuffer(static_cast<vislib::StringA::Size>(size));

    /* Read source and ensure binary zero at end. */
    file.Read(src, size); 
    src[size] = 0;

    return true;
}


/*
 * vislib::sys::GetTicksOfDay
 */
unsigned int vislib::sys::GetTicksOfDay(void) {
#ifdef _WIN32
    SYSTEMTIME systemTime;
    ::GetLocalTime(&systemTime);
    return static_cast<unsigned int>(systemTime.wMilliseconds) 
        + 1000 * (static_cast<unsigned int>(systemTime.wSecond) + 60 * (systemTime.wMinute + 60 * systemTime.wHour));

#else /* _WIN32 */
    struct timeval tv;
    struct tm tm;

    if (::gettimeofday(&tv, NULL) == 0) {

        if (::gmtime_r(&tv.tv_sec, &tm) != NULL) {
            return (tv.tv_usec / 1000) 
                + 1000 * (static_cast<unsigned int>(tm.tm_sec) + 60 * (tm.tm_min + 60 * tm.tm_hour));

        } else {
            return tv.tv_usec / 1000 + 1000 * (tv.tv_sec % 86400);
        }

    } else {
        return 0; // ultimate linux failure.
    }

#endif /* _WIN32 */
}


#ifdef _WIN32
/*
 * vislib::sys::GetDLLVersion
 */
HRESULT vislib::sys::GetDLLVersion(DLLVERSIONINFO& outVersion, 
                                   const char *moduleName) {
    DLLGETVERSIONPROC dllGetVersion = NULL;
    HMODULE hModule = NULL;

    if ((hModule = ::LoadLibraryA(moduleName)) == NULL) {
        throw SystemException(__FILE__, __LINE__);
    }

    if ((dllGetVersion = reinterpret_cast<DLLGETVERSIONPROC>(::GetProcAddress(
            hModule, "DllGetVersion"))) == NULL) {
        throw SystemException(__FILE__, __LINE__);
    }

    return dllGetVersion(&outVersion);
}


/*
 * vislib::sys::GetDLLVersion
 */
HRESULT vislib::sys::GetDLLVersion(DLLVERSIONINFO& outVersion, 
                                   const wchar_t * moduleName) {
    DLLGETVERSIONPROC dllGetVersion = NULL;
    HMODULE hModule = NULL;

    if ((hModule = ::LoadLibraryW(moduleName)) == NULL) {
        throw SystemException(__FILE__, __LINE__);
    }

    if ((dllGetVersion = reinterpret_cast<DLLGETVERSIONPROC>(::GetProcAddress(
            hModule, "DllGetVersion"))) == NULL) {
        throw SystemException(__FILE__, __LINE__);
    }

    return dllGetVersion(&outVersion);
}
#endif /* _WIN32 */


/*
 * vislib::sys::RemoveKernelNamespace
 */
vislib::StringA vislib::sys::RemoveKernelNamespace(const char *name) {
    StringA n(name);

    if (n.StartsWithInsensitive("global\\") 
            || n.StartsWithInsensitive("local\\")) {
        n.Remove(0, n.Find('\\') + 1);
    }

    return n;
}


/*
 * vislib::sys::RemoveKernelNamespace
 */
vislib::StringW vislib::sys::RemoveKernelNamespace(const wchar_t *name) {
    StringW n(name);

    if (n.StartsWithInsensitive(L"global\\") 
            || n.StartsWithInsensitive(L"local\\")) {
        n.Remove(0, n.Find(L'\\') + 1);
    }

    return n;
}


/*
 * vislib::sys::TranslateWinIpc2PosixName
 */
vislib::StringA vislib::sys::TranslateWinIpc2PosixName(const char *name) {
    StringA retval = RemoveKernelNamespace(name);
    retval.Prepend('/');
    return retval;
}


/*
 * vislib::sys::TranslateWinIpc2PosixName
 */
vislib::StringW vislib::sys::TranslateWinIpc2PosixName(const wchar_t *name) {
    StringW retval = RemoveKernelNamespace(name);
    retval.Prepend(L'/');
    return retval;
}


#ifndef _WIN32
/*
 * vislib::sys::TranslateIpcName
 */
key_t vislib::sys::TranslateIpcName(const char *name) {
    key_t retval = -1;
    
    
    /* Remove Windows kernel namespaces from the name. */
    StringA n = RemoveKernelNamespace(name);
    ASSERT(n.Length() > 0);

    // TODO: Ist das Verzeichnis sinnvoll? Eher nicht ...
    retval = ::ftok(Path::GetUserHomeDirectoryA().PeekBuffer(), n.HashCode());
    if (retval == -1) {
        throw SystemException(__FILE__, __LINE__);
    }

    VLTRACE(Trace::LEVEL_VL_INFO, "TranslateIpcName(\"%s\") = %u\n", name, 
        retval);
    return retval;
}
#endif /* !_WIN32 */
