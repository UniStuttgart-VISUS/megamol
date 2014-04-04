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

#include "the/assert.h"
#include "vislib/File.h"
#include "the/argument_exception.h"
#include "the/system/io/io_exception.h"
#include "the/memory.h"
#include "vislib/Path.h"
#include "the/string.h"
#include "the/text/string_converter.h"
#include "the/system/system_exception.h"
#include "the/trace.h"
#include "the/not_supported_exception.h"


#ifdef _WIN32
/**
 * Common implementation of the vislib::sys::LoadResource functions.
 *
 * @param out     A RawStorage that will receive the resource data. 
 * @param hModule The module handle passed to vislib::sys::LoadResource.
 * @param hRes    The handle of the resource to be retrieved. That must not
 *                be NULL.
 *
 * @returns 'out'.
 *
 * @throws the::system::system_exception If the resource lookup or loading the resource
 *                         failed.
 */
static vislib::RawStorage& loadResource(vislib::RawStorage& out, 
        HMODULE hModule, HRSRC hRes) {
    THE_ASSERT(hRes != NULL);
    HGLOBAL hGlobal = NULL;
    void *data = NULL;
    unsigned int size = 0;

    if ((hGlobal = ::LoadResource(hModule, hRes)) == NULL) {
        // From MSDN: The return type of LoadResource is HGLOBAL for backward 
        // compatibility, not because the function returns a handle to a 
        // global memory block. Do not pass this handle to the GlobalLock 
        // or GlobalFree function.
        throw the::system::system_exception(__FILE__, __LINE__);
    }

    if ((size = ::SizeofResource(hModule, hRes)) == 0) {
        throw the::system::system_exception(__FILE__, __LINE__);
    }

    data = ::LockResource(hGlobal);
    THE_ASSERT(data != NULL);
    out.EnforceSize(0);
    out.Append(data, size);
    UnlockResource(hGlobal);
    return out;
}
#endif /* _WIN32 */


/*
 * vislib::sys::LoadResource
 */
#ifdef _WIN32
vislib::RawStorage& vislib::sys::LoadResource(RawStorage& out, HMODULE hModule,
        const char *resourceID, const char *resourceType) {
    HRSRC hRes = NULL;
    
    if ((hRes = ::FindResourceA(hModule, resourceID, resourceType)) == NULL) {
        throw the::system::system_exception(__FILE__, __LINE__);
    }
    
    return ::loadResource(out, hModule, hRes);
}
#else /* _WIN32 */
vislib::RawStorage& vislib::sys::LoadResource(RawStorage& out, void *hModule, 
        const char *resourceID, const char *resourceType) {
    throw the::not_supported_exception("LoadResource", __FILE__, __LINE__);
}
#endif /* _WIN32 */



/*
 * vislib::sys::LoadResource
 */
#ifdef _WIN32
vislib::RawStorage& vislib::sys::LoadResource(RawStorage& out, HMODULE hModule,
        const wchar_t *resourceID, const wchar_t *resourceType) {
    HRSRC hRes = NULL;
    
    if ((hRes = ::FindResourceW(hModule, resourceID, resourceType)) == NULL) {
        throw the::system::system_exception(__FILE__, __LINE__);
    }
    
    return ::loadResource(out, hModule, hRes);
}
#else /* _WIN32 */
vislib::RawStorage& vislib::sys::LoadResource(RawStorage& out, void *hModule, 
        const wchar_t *resourceID, const wchar_t *resourceType) {
    throw the::not_supported_exception("LoadResource", __FILE__, __LINE__);
}
#endif /* _WIN32 */      


/*
 * vislib::sys::ReadLineFromFileA
 */
the::astring vislib::sys::ReadLineFromFileA(File& input, unsigned int size) {
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

    } catch(the::system::io::io_exception e) {
        the::safe_array_delete(buf);
        throw the::system::io::io_exception(e);
    } catch(the::exception e) {
        the::safe_array_delete(buf);
        throw the::exception(e);
    } catch(...) {
        the::safe_array_delete(buf);
        throw the::exception("Unexcepted exception", __FILE__, __LINE__);
    }
    the::astring str(buf);
    delete[] buf;
    return str;
}


/*
 * vislib::sys::ReadLineFromFileW
 */
the::wstring vislib::sys::ReadLineFromFileW(File& input, unsigned int size) {
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

    } catch(the::system::io::io_exception e) {
        the::safe_array_delete(buf);
        throw the::system::io::io_exception(e);
    } catch(the::exception e) {
        the::safe_array_delete(buf);
        throw the::exception(e);
    } catch(...) {
        the::safe_array_delete(buf);
        throw the::exception("Unexcepted exception", __FILE__, __LINE__);
    }
    the::wstring str(buf);
    delete[] buf;
    return str;
}


/**
 * Interprets BOM data if possible
 *
 * @param bom The BOM bytes
 * @param bomSize The number of bytes 'bom' points to. When successfully
 *                detected a valid BOM the correct length of the BOM in bytes
 *                is set.
 *
 * @return The recognized BOM or 'TEXTFF_UNSPECIFIC'
 */
vislib::sys::TextFileFormat interpretBOM(unsigned char *bom,
        unsigned int& bomSize) {
    if ((bomSize >= 3) && (bom[0] == 0xEF) && (bom[1] == 0xBB)
            && (bom[2] == 0xBF)) {
        bomSize = 3;
        return vislib::sys::TEXTFF_UTF8;
    }
    if ((bomSize >= 4) && (bom[0] == 0x00) && (bom[1] == 0x00)
            && (bom[2] == 0xFE) && (bom[3] == 0xFF)) {
        bomSize = 4;
        return vislib::sys::TEXTFF_UTF32_BE;
    }
    if ((bomSize >= 4) && (bom[0] == 0xFF) && (bom[1] == 0xFE)
            && (bom[2] == 0x00) && (bom[3] == 0x00)) {
        bomSize = 4;
        return vislib::sys::TEXTFF_UTF32;
    }
    if ((bomSize >= 2) && (bom[0] == 0xFE) && (bom[1] == 0xFF)) {
        bomSize = 2;
        return vislib::sys::TEXTFF_UTF16_BE;
    }
    if ((bomSize >= 2) && (bom[0] == 0xFF) && (bom[1] == 0xFE)) {
        bomSize = 2;
        return vislib::sys::TEXTFF_UTF16;
    }
    if ((bomSize >= 4) && (bom[0] == 0x2B) && (bom[1] == 0x2F)
            && (bom[2] == 0x76) && ((bom[3] == 0x38) || (bom[3] == 0x39)
                || (bom[3] == 0x2B) || (bom[3] == 0x2F))) {
        bomSize = 0; // because forth byte is only partially BOM the decoder
                     // has to know the data and must retest for BOM
        return vislib::sys::TEXTFF_UTF7;
    }
    if ((bomSize >= 3) && (bom[0] == 0xF7) && (bom[1] == 0x64)
            && (bom[2] == 0x4C)) {
        bomSize = 3;
        return vislib::sys::TEXTFF_UTF1;
    }
    if ((bomSize >= 4) && (bom[0] == 0xDD) && (bom[1] == 0x73)
            && (bom[2] == 0x66) && (bom[3] == 0x73)) {
        bomSize = 4;
        return vislib::sys::TEXTFF_UTF_EBCDIC;
    }
    if ((bomSize >= 3) && (bom[0] == 0x0E) && (bom[1] == 0xFE)
            && (bom[2] == 0xFF)) {
        bomSize = 3;
        return vislib::sys::TEXTFF_SCSU;
    }
    if ((bomSize >= 3) && (bom[0] == 0xFB) && (bom[1] == 0xEE)
            && (bom[2] == 0x28)) {
        if ((bomSize == 4) && (bom[3] != 0xFF)) {
            bomSize = 3;
        }
        return vislib::sys::TEXTFF_BOCU1;
    }
    if ((bomSize >= 4) && (bom[0] == 0x84) && (bom[1] == 0x31)
            && (bom[2] == 0x95) && (bom[3] == 0x33)) {
        bomSize = 4;
        return vislib::sys::TEXTFF_GB18030;
    }

    return vislib::sys::TEXTFF_UNSPECIFIC;
}


/**
 * Checks the specified file format against the file stream
 */
void checkFileFormat(vislib::sys::File& file,
        vislib::sys::TextFileFormat& format,
        vislib::sys::TextFileFormat fallback) {
    vislib::sys::File::FileSize start = file.Tell();

    if (start == 0) {
        unsigned char bom[4];
        unsigned int bomSize = static_cast<unsigned int>(file.Read(bom, 4));
        vislib::sys::TextFileFormat bomFF = interpretBOM(bom, bomSize);
        if (bomFF != vislib::sys::TEXTFF_UNSPECIFIC) { // BOM detected
            format = bomFF; // BOM always better than manual format
                            // here we know that 'forceFormat' was false
        } else {
            bomSize = 0;
        }
        file.Seek(start + bomSize);
    }
    if (format == vislib::sys::TEXTFF_UNSPECIFIC) {
        format = fallback;
    }
}


/*
 * vislib::sys::ReadTextFile
 */
bool vislib::sys::ReadTextFile(the::astring& outStr, 
        vislib::sys::File& file, vislib::sys::TextFileFormat format,
        bool forceFormat) {
    if ((format == TEXTFF_UNSPECIFIC) || !forceFormat) {
        checkFileFormat(file, format,
            (format == TEXTFF_UNSPECIFIC) ? TEXTFF_ASCII : format);
    }
    THE_ASSERT(format != TEXTFF_UNSPECIFIC);
    File::FileSize len = file.GetSize() - file.Tell();

    switch (format) {
    case TEXTFF_ASCII: {
        outStr = the::astring(len + 1, '\0');
        char *src = const_cast<char*>(outStr.c_str());
        len = file.Read(src, len);
        src[len] = 0;
        return true;
    } break;
    case TEXTFF_UNICODE:
#ifdef _WIN32
CASE_TEXTFF_UNICODE: 
#endif /* _WIN32 */
    {
        the::wstring tmp(static_cast<unsigned int>(
            (len / sizeof(wchar_t)) + sizeof(wchar_t)), L'\0');
        wchar_t *src = const_cast<wchar_t*>(tmp.c_str());
        len = file.Read(src, len - len % sizeof(wchar_t));
        src[len / sizeof(wchar_t)] = 0;
        the::text::string_converter::convert(outStr, tmp);
        return true;
    } break;
    case TEXTFF_UTF8: {
        the::astring bytes(static_cast<unsigned int>(len + 1), '\0');
        char *src = const_cast<char *>(bytes.c_str());
        len = file.Read(src, len);
        src[len] = 0;
        the::text::string_converter::convert_from_utf8(outStr, bytes);
        return true;
    } break;
    case TEXTFF_UTF16:
#ifdef _WIN32
        goto CASE_TEXTFF_UNICODE;
#else /* _WIN32 */
        THE_TRACE(THE_TRCCHL_DEFAULT, THE_TRCLVL_ERROR, "Unsupported text file format %d\n", format);
#endif /* _WIN32 */
        break;
    case TEXTFF_UTF16_BE:
        THE_TRACE(THE_TRCCHL_DEFAULT, THE_TRCLVL_ERROR, "Unsupported text file format %d\n", format);
        break;
    case TEXTFF_UTF32:
        THE_TRACE(THE_TRCCHL_DEFAULT, THE_TRCLVL_ERROR, "Unsupported text file format %d\n", format);
        break;
    case TEXTFF_UTF32_BE:
        THE_TRACE(THE_TRCCHL_DEFAULT, THE_TRCLVL_ERROR, "Unsupported text file format %d\n", format);
        break;
    case TEXTFF_UTF7:
        THE_TRACE(THE_TRCCHL_DEFAULT, THE_TRCLVL_ERROR, "Unsupported text file format %d\n", format);
        break;
    case TEXTFF_UTF1:
        THE_TRACE(THE_TRCCHL_DEFAULT, THE_TRCLVL_ERROR, "Unsupported text file format %d\n", format);
        break;
    case TEXTFF_UTF_EBCDIC:
        THE_TRACE(THE_TRCCHL_DEFAULT, THE_TRCLVL_ERROR, "Unsupported text file format %d\n", format);
        break;
    case TEXTFF_SCSU:
        THE_TRACE(THE_TRCCHL_DEFAULT, THE_TRCLVL_ERROR, "Unsupported text file format %d\n", format);
        break;
    case TEXTFF_BOCU1:
        THE_TRACE(THE_TRCCHL_DEFAULT, THE_TRCLVL_ERROR, "Unsupported text file format %d\n", format);
        break;
    case TEXTFF_GB18030:
        THE_TRACE(THE_TRCCHL_DEFAULT, THE_TRCLVL_ERROR, "Unsupported text file format %d\n", format);
        break;
    default:
        THE_TRACE(THE_TRCCHL_DEFAULT, THE_TRCLVL_ERROR, "Unknown text file format %d\n", format);
        break;
    }

    return false;
}


/*
 * vislib::sys::ReadTextFile
 */
bool vislib::sys::ReadTextFile(the::wstring& outStr, 
        vislib::sys::File& file, vislib::sys::TextFileFormat format,
        bool forceFormat) {
    if ((format == TEXTFF_UNSPECIFIC) || !forceFormat) {
        checkFileFormat(file, format,
            (format == TEXTFF_UNSPECIFIC) ? TEXTFF_UNICODE : format);
    }
    THE_ASSERT(format != TEXTFF_UNSPECIFIC);
    File::FileSize len = file.GetSize() - file.Tell();

    switch (format) {
    case TEXTFF_ASCII: {
        the::astring tmp(static_cast<unsigned int>(len + 1), '\0');
        char *src = const_cast<char*>(tmp.c_str());
        len = file.Read(src, len);
        src[len] = 0;
        the::text::string_converter::convert(outStr, tmp);
        return true;
    } break;
    case TEXTFF_UNICODE:
#ifdef _WIN32
CASE_TEXTFF_UNICODE:
#endif /* WIN32 */
    {
        outStr = the::wstring(static_cast<unsigned int>(
            (len / sizeof(wchar_t)) + sizeof(wchar_t)), L'\0');
        wchar_t *src = const_cast<wchar_t*>(outStr.c_str());
        len = file.Read(src, len - len % sizeof(wchar_t));
        src[len / sizeof(wchar_t)] = 0;
        return true;
    } break;
    case TEXTFF_UTF8: {
        the::astring bytes(static_cast<unsigned int>(len + 1), '\0');
        char *src = const_cast<char*>(bytes.c_str());
        len = file.Read(src, len);
        src[len] = 0;
        the::text::string_converter::convert_from_utf8(outStr, bytes);
        return true;
    } break;
    case TEXTFF_UTF16:
#ifdef _WIN32
        goto CASE_TEXTFF_UNICODE;
#else /* _WIN32 */
        THE_TRACE(THE_TRCCHL_DEFAULT, THE_TRCLVL_ERROR, "Unsupported text file format %d\n", format);
#endif /* _WIN32 */
        break;
    case TEXTFF_UTF16_BE:
        THE_TRACE(THE_TRCCHL_DEFAULT, THE_TRCLVL_ERROR, "Unsupported text file format %d\n", format);
        break;
    case TEXTFF_UTF32:
        THE_TRACE(THE_TRCCHL_DEFAULT, THE_TRCLVL_ERROR, "Unsupported text file format %d\n", format);
        break;
    case TEXTFF_UTF32_BE:
        THE_TRACE(THE_TRCCHL_DEFAULT, THE_TRCLVL_ERROR, "Unsupported text file format %d\n", format);
        break;
    case TEXTFF_UTF7:
        THE_TRACE(THE_TRCCHL_DEFAULT, THE_TRCLVL_ERROR, "Unsupported text file format %d\n", format);
        break;
    case TEXTFF_UTF1:
        THE_TRACE(THE_TRCCHL_DEFAULT, THE_TRCLVL_ERROR, "Unsupported text file format %d\n", format);
        break;
    case TEXTFF_UTF_EBCDIC:
        THE_TRACE(THE_TRCCHL_DEFAULT, THE_TRCLVL_ERROR, "Unsupported text file format %d\n", format);
        break;
    case TEXTFF_SCSU:
        THE_TRACE(THE_TRCCHL_DEFAULT, THE_TRCLVL_ERROR, "Unsupported text file format %d\n", format);
        break;
    case TEXTFF_BOCU1:
        THE_TRACE(THE_TRCCHL_DEFAULT, THE_TRCLVL_ERROR, "Unsupported text file format %d\n", format);
        break;
    case TEXTFF_GB18030:
        THE_TRACE(THE_TRCCHL_DEFAULT, THE_TRCLVL_ERROR, "Unsupported text file format %d\n", format);
        break;
    default:
        THE_TRACE(THE_TRCCHL_DEFAULT, THE_TRCLVL_ERROR, "Unknown text file format %d\n", format);
        break;
    }

    return false;
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
        throw the::system::system_exception(__FILE__, __LINE__);
    }

    if ((dllGetVersion = reinterpret_cast<DLLGETVERSIONPROC>(::GetProcAddress(
            hModule, "DllGetVersion"))) == NULL) {
        throw the::system::system_exception(__FILE__, __LINE__);
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
        throw the::system::system_exception(__FILE__, __LINE__);
    }

    if ((dllGetVersion = reinterpret_cast<DLLGETVERSIONPROC>(::GetProcAddress(
            hModule, "DllGetVersion"))) == NULL) {
        throw the::system::system_exception(__FILE__, __LINE__);
    }

    return dllGetVersion(&outVersion);
}
#endif /* _WIN32 */


/*
 * vislib::sys::RemoveKernelNamespace
 */
the::astring vislib::sys::RemoveKernelNamespace(const char *name) {
    the::astring n(name);

    if (the::text::string_utility::starts_with(n, "global\\", false)
            || the::text::string_utility::starts_with(n, "local\\", false)) {
        n.erase(n.begin(), std::find(n.begin(), n.end(), '\\'));
    }

    return n;
}


/*
 * vislib::sys::RemoveKernelNamespace
 */
the::wstring vislib::sys::RemoveKernelNamespace(const wchar_t *name) {
    the::wstring n(name);

    if (the::text::string_utility::starts_with(n, L"global\\", false)
            || the::text::string_utility::starts_with(n, L"local\\", false)) {
        n.erase(n.begin(), std::find(n.begin(), n.end(), '\\'));
    }

    return n;
}


/*
 * vislib::sys::TranslateWinIpc2PosixName
 */
the::astring vislib::sys::TranslateWinIpc2PosixName(const char *name) {
    the::astring retval = RemoveKernelNamespace(name);
    retval.insert(0, "//");
    return retval;
}


/*
 * vislib::sys::TranslateWinIpc2PosixName
 */
the::wstring vislib::sys::TranslateWinIpc2PosixName(const wchar_t *name) {
    the::wstring retval = RemoveKernelNamespace(name);
    retval.insert(0, L"//");
    return retval;
}


#ifndef _WIN32
/*
 * vislib::sys::TranslateIpcName
 */
key_t vislib::sys::TranslateIpcName(const char *name) {
    key_t retval = -1;
    
    
    /* Remove Windows kernel namespaces from the name. */
    the::astring n = RemoveKernelNamespace(name);
    THE_ASSERT(n.size() > 0);

    // TODO: Ist das Verzeichnis sinnvoll? Eher nicht ...
    retval = ::ftok(Path::GetUserHomeDirectoryA().c_str(), the::text::string_utility::hash_code(n));
    if (retval == -1) {
        throw the::system::system_exception(__FILE__, __LINE__);
    }

    THE_TRACE(THE_TRCCHL_DEFAULT, THE_TRCLVL_INFO, "TranslateIpcName(\"%s\") = %u\n", name, 
        retval);
    return retval;
}
#endif /* !_WIN32 */


/*
 * vislib::sys::WriteTextFile
 */
bool vislib::sys::WriteTextFile(vislib::sys::File& file,
        const the::astring& text, vislib::sys::TextFileFormat format,
        TextFileFormatBOM bom) {
    switch (format) {
    case TEXTFF_UNSPECIFIC:
        goto CASE_TEXTFF_ASCII;
        break;
    case TEXTFF_ASCII:
CASE_TEXTFF_ASCII: {
        // no BOM possible
        the::astring::size_type len = text.size();
        return (static_cast<the::astring::size_type>(file.Write(text.c_str(), len))
            == len);
    } break;
    case TEXTFF_UNICODE:
#ifdef _WIN32
CASE_TEXTFF_UNICODE:
#endif /* _WIN32 */
        // write BOM as sfx
        return WriteTextFile(file, the::text::string_converter::to_w(text), format, bom);
        break;
    case TEXTFF_UTF8: {
        the::astring bytes;
        the::text::string_converter::convert_to_utf8(bytes, text);
        if (bom != TEXTFF_BOM_NO) {
            unsigned char BOM[] = { 0xEF, 0xBB, 0xBF };
            file.Write(BOM, 3);
        }
        the::astring::size_type len = bytes.size();
        return (static_cast<the::astring::size_type>(file.Write(bytes.c_str(), len)) == len);
    } break;
    case TEXTFF_UTF16:
#ifdef _WIN32
        goto CASE_TEXTFF_UNICODE;
#else /* _WIN32 */
        THE_TRACE(THE_TRCCHL_DEFAULT, THE_TRCLVL_ERROR, "Unsupported text file format %d\n", format);
#endif /* _WIN32 */
        break;
    case TEXTFF_UTF16_BE:
        THE_TRACE(THE_TRCCHL_DEFAULT, THE_TRCLVL_ERROR, "Unsupported text file format %d\n", format);
        break;
    case TEXTFF_UTF32:
        THE_TRACE(THE_TRCCHL_DEFAULT, THE_TRCLVL_ERROR, "Unsupported text file format %d\n", format);
        break;
    case TEXTFF_UTF32_BE:
        THE_TRACE(THE_TRCCHL_DEFAULT, THE_TRCLVL_ERROR, "Unsupported text file format %d\n", format);
        break;
    case TEXTFF_UTF7:
        THE_TRACE(THE_TRCCHL_DEFAULT, THE_TRCLVL_ERROR, "Unsupported text file format %d\n", format);
        break;
    case TEXTFF_UTF1:
        THE_TRACE(THE_TRCCHL_DEFAULT, THE_TRCLVL_ERROR, "Unsupported text file format %d\n", format);
        break;
    case TEXTFF_UTF_EBCDIC:
        THE_TRACE(THE_TRCCHL_DEFAULT, THE_TRCLVL_ERROR, "Unsupported text file format %d\n", format);
        break;
    case TEXTFF_SCSU:
        THE_TRACE(THE_TRCCHL_DEFAULT, THE_TRCLVL_ERROR, "Unsupported text file format %d\n", format);
        break;
    case TEXTFF_BOCU1:
        THE_TRACE(THE_TRCCHL_DEFAULT, THE_TRCLVL_ERROR, "Unsupported text file format %d\n", format);
        break;
    case TEXTFF_GB18030:
        THE_TRACE(THE_TRCCHL_DEFAULT, THE_TRCLVL_ERROR, "Unsupported text file format %d\n", format);
        break;
    default:
        THE_TRACE(THE_TRCCHL_DEFAULT, THE_TRCLVL_ERROR, "Unknown text file format %d\n", format);
        break;
    }
    //size_t len = text.Length();
    //return (file.Write(text.c_str(), len) == len);

    return false;
}


/*
 * vislib::sys::WriteTextFile
 */
bool vislib::sys::WriteTextFile(vislib::sys::File& file,
        const the::wstring& text, vislib::sys::TextFileFormat format,
        TextFileFormatBOM bom) {
    switch (format) {
    case TEXTFF_UNSPECIFIC:
        goto CASE_TEXTFF_UTF8; // because it is platform independent unicode
        break;
    case TEXTFF_ASCII:
        // no BOM possible
        return WriteTextFile(file, the::text::string_converter::to_a(text), format, bom);
        break;
    case TEXTFF_UNICODE:
#ifdef _WIN32
CASE_TEXTFF_UNICODE:
#endif /* _WIN32 */
    {
#ifdef _WIN32
        if (bom != TEXTFF_BOM_NO) {
            unsigned char BOM[] = { 0xFF, 0xFE };
            file.Write(BOM, 2);
        }
#endif /* _WIN32 */
        the::wstring::size_type len = text.size() * sizeof(wchar_t);
        return (static_cast<the::wstring::size_type>(file.Write(text.c_str(), len))
            == len);
    } break;
    case TEXTFF_UTF8:
CASE_TEXTFF_UTF8: {
        the::astring bytes;
        the::text::string_converter::convert_to_utf8(bytes, text);
        if (bom != TEXTFF_BOM_NO) {
            unsigned char BOM[] = { 0xEF, 0xBB, 0xBF };
            file.Write(BOM, 3);
        }
        the::astring::size_type len = bytes.size();
        return (static_cast<the::astring::size_type>(file.Write(bytes.c_str(), len)) == len);
    } break;
    case TEXTFF_UTF16:
#ifdef _WIN32
        goto CASE_TEXTFF_UNICODE;
#else /* _WIN32 */
        THE_TRACE(THE_TRCCHL_DEFAULT, THE_TRCLVL_ERROR, "Unsupported text file format %d\n", format);
#endif /* _WIN32 */
        break;
    case TEXTFF_UTF16_BE:
        THE_TRACE(THE_TRCCHL_DEFAULT, THE_TRCLVL_ERROR, "Unsupported text file format %d\n", format);
        break;
    case TEXTFF_UTF32:
        THE_TRACE(THE_TRCCHL_DEFAULT, THE_TRCLVL_ERROR, "Unsupported text file format %d\n", format);
        break;
    case TEXTFF_UTF32_BE:
        THE_TRACE(THE_TRCCHL_DEFAULT, THE_TRCLVL_ERROR, "Unsupported text file format %d\n", format);
        break;
    case TEXTFF_UTF7:
        THE_TRACE(THE_TRCCHL_DEFAULT, THE_TRCLVL_ERROR, "Unsupported text file format %d\n", format);
        break;
    case TEXTFF_UTF1:
        THE_TRACE(THE_TRCCHL_DEFAULT, THE_TRCLVL_ERROR, "Unsupported text file format %d\n", format);
        break;
    case TEXTFF_UTF_EBCDIC:
        THE_TRACE(THE_TRCCHL_DEFAULT, THE_TRCLVL_ERROR, "Unsupported text file format %d\n", format);
        break;
    case TEXTFF_SCSU:
        THE_TRACE(THE_TRCCHL_DEFAULT, THE_TRCLVL_ERROR, "Unsupported text file format %d\n", format);
        break;
    case TEXTFF_BOCU1:
        THE_TRACE(THE_TRCCHL_DEFAULT, THE_TRCLVL_ERROR, "Unsupported text file format %d\n", format);
        break;
    case TEXTFF_GB18030:
        THE_TRACE(THE_TRCCHL_DEFAULT, THE_TRCLVL_ERROR, "Unsupported text file format %d\n", format);
        break;
    default:
        THE_TRACE(THE_TRCCHL_DEFAULT, THE_TRCLVL_ERROR, "Unknown text file format %d\n", format);
        break;
    }
    //the::astring tmp(text);
    //size_t len = tmp.Length();
    //return (file.Write(tmp.c_str(), len) == len);
    return false;
}
