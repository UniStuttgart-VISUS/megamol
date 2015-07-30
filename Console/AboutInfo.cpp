/*
 * AboutInfo.cpp
 *
 * Copyright (C) 2008 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */
#include "stdafx.h"
#include "AboutInfo.h"
#include "mmcore/api/MegaMolCore.h"
#include "mmcore/versioninfo.h"
#include "MegaMolViewer.h"
#include "versioninfo.h"
#include "vislib/sys/Log.h"
#include "vislib/String.h"
#include "vislib/UnsupportedOperationException.h"


/*
 * megamol::console::utility::AboutInfo::LibFileNameFormatString
 */
const char *
megamol::console::utility::AboutInfo::LibFileNameFormatString(void) {
    return 
#ifdef _WIN32
#if defined(WINVER)
#if (WINVER >= 0x0501)
        //""    // Window
#endif /* (WINVER >= 0x0501) */
#endif /* defined(WINVER) */
#else /* _WIN32 */
        "lib"    // Linux
#endif /* _WIN32 */
        "%s" // MEGAMOL_FILENAME_BITSD 
#if !defined(_WIN32) && (defined(DEBUG) || defined(_DEBUG))
	"d"
#endif
        MEGAMOL_DLL_FILENAME_EXT;
}


/*
 * megamol::console::utility::AboutInfo::PrintGreeting
 */
void megamol::console::utility::AboutInfo::PrintGreeting(void) {
    printf("\n");
    printf("    MegaMol%s Console\n", ASCIIStringTM());
    vislib::StringA cr(MEGAMOL_CONSOLE_COPYRIGHT);
    cr.Replace("\n", "\n    ");
    cr.Prepend("    ");
    printf("%s\n", cr.PeekBuffer());
}


/*
 * megamol::console::utility::AboutInfo::LogGreeting
 */
void megamol::console::utility::AboutInfo::LogGreeting(void) {
    using vislib::sys::Log;
    Log::DefaultLog.WriteMsg(Log::LEVEL_INFO, "MegaMol%s Console",
        ASCIIStringTM());
}


/*
 * megamol::console::utility::AboutInfo::PrintVersionInfo
 */
void megamol::console::utility::AboutInfo::PrintVersionInfo(void) {
    printf("%s\n", AboutInfo::consoleVersionString().PeekBuffer());
    printf("%s\n", AboutInfo::consoleCommentString().PeekBuffer());
    printf("%s\n", AboutInfo::coreVersionString(true).PeekBuffer());
    printf("%s\n", AboutInfo::coreCommentString().PeekBuffer());
}


/*
 * megamol::console::utility::AboutInfo::LogVersionInfo
 */
void megamol::console::utility::AboutInfo::LogVersionInfo(void) {
    using vislib::sys::Log;
    Log::DefaultLog.WriteMsg(Log::LEVEL_INFO, AboutInfo::consoleVersionString());
    vislib::StringA comment = AboutInfo::consoleCommentString();
    comment.Replace("\n", "; ");
    Log::DefaultLog.WriteMsg(Log::LEVEL_INFO + 100, comment.PeekBuffer());
    Log::DefaultLog.WriteMsg(Log::LEVEL_INFO, AboutInfo::coreVersionString());
    comment = AboutInfo::coreCommentString();
    comment.Replace("\n", "; ");
    Log::DefaultLog.WriteMsg(Log::LEVEL_INFO + 100, comment.PeekBuffer());
}


/*
 * megamol::console::utility::AboutInfo::LogViewerVersionInfo
 */
void megamol::console::utility::AboutInfo::LogViewerVersionInfo(void) {
    using vislib::sys::Log;
    Log::DefaultLog.WriteMsg(Log::LEVEL_INFO, AboutInfo::viewerVersionString());
    vislib::StringA comment = AboutInfo::viewerCommentString();
    comment.Replace("\n", "; ");
    Log::DefaultLog.WriteMsg(Log::LEVEL_INFO + 100, comment.PeekBuffer());
}


/*
 * megamol::console::utility::AboutInfo::Version
 */
vislib::VersionNumber megamol::console::utility::AboutInfo::Version(void) {
    return vislib::VersionNumber(MEGAMOL_CONSOLE_VERSION);
}


/*
 * megamol::console::utility::AboutInfo::AboutInfo
 */
megamol::console::utility::AboutInfo::AboutInfo(void) {
    throw vislib::UnsupportedOperationException("AboutInfo::Ctor", __FILE__, __LINE__);
}


/*
 * megamol::console::utility::AboutInfo::~AboutInfo
 */
megamol::console::utility::AboutInfo::~AboutInfo(void) {
    throw vislib::UnsupportedOperationException("AboutInfo::Dtor", __FILE__, __LINE__);
}


/*
 * megamol::console::utility::AboutInfo::consoleVersionString
 */
vislib::StringA megamol::console::utility::AboutInfo::consoleVersionString(void) {
    vislib::StringA retval = "Console: ";
    vislib::StringA str;
    vislib::VersionNumber ver = AboutInfo::Version();

    str.Format("(Ver.: %s)", ver.ToStringA().PeekBuffer());
    retval.Append(str);
#if defined(_WIN64) || defined(_LIN64)
    retval.Append(" 64 Bit ");
#else /* defined(_WIN64) || defined(_LIN64) */
    retval.Append(" 32 Bit ");
#endif /* defined(_WIN64) || defined(_LIN64) */

#ifdef _WIN32
#if defined(WINVER)
#if (WINVER >= 0x0501)
    retval.Append("Windows");
#endif /* (WINVER >= 0x0501) */
#endif /* defined(WINVER) */
#else /* _WIN32 */
    retval.Append("Linux");
#endif /* _WIN32 */

#if defined(DEBUG) || defined(_DEBUG)
    retval.Append(" [DEBUG]");
#endif

    return retval;
}


/*
 * megamol::console::utility::AboutInfo::consoleCommentString
 */
vislib::StringA megamol::console::utility::AboutInfo::consoleCommentString(void) {
    return MEGAMOL_CONSOLE_COMMENTS;
}


/*
 * megamol::console::utility::AboutInfo::coreVersionString
 */
vislib::StringA megamol::console::utility::AboutInfo::coreVersionString(bool withCopyright) {
    vislib::StringA retval;
    int wordSize = 0;
    vislib::VersionNumber ver;
    const char *systemStr = "Unknown";
    vislib::StringA str;

    ::mmcBinaryVersionInfo *vi = ::mmcGetVersionInfo();
    if (vi == nullptr) return nullptr;

    ver.Set(vi->VersionNumber[0], vi->VersionNumber[1], vi->VersionNumber[2], vi->VersionNumber[3]);
    switch (vi->HardwareArchitecture) {
        case MMC_HARCH_I86: wordSize = 32; break;
        case MMC_HARCH_X64: wordSize = 64; break;
        default: wordSize = 0;
    }
    switch (vi->SystemType) {
        case MMC_OSYSTEM_WINDOWS: systemStr = "Windows"; break;
        case MMC_OSYSTEM_LINUX: systemStr = "Linux"; break;
        default: systemStr = "Unknown"; 
    }

    retval.Format("Core \"%s\" (Ver.: %s) %d Bit %s",
        vi->NameStr, ver.ToStringA().PeekBuffer(), wordSize, systemStr);

    if ((vi->Flags & MMC_BFLAG_DEBUG)) { str.Append(" DEBUG;"); }
    if ((vi->Flags & MMC_BFLAG_DIRTY)) { str.Append(" DIRTY;"); }
    if (!str.IsEmpty()) {
        str[0] = '[';
        str[str.Length() - 1] = ']';
        str.Prepend(' ');
        retval.Append(str);
    }

    if (withCopyright) {
        retval.Append("\n");
        retval.Append(vi->CopyrightStr);
    }

    ::mmcFreeVersionInfo(vi);

    return retval;
}


/*
 * megamol::console::utility::AboutInfo::coreCommentString
 */
vislib::StringA megamol::console::utility::AboutInfo::coreCommentString(void) {
    vislib::StringA comment;

    ::mmcBinaryVersionInfo *vi = ::mmcGetVersionInfo();
    if (vi == nullptr) return nullptr;

    comment = vi->CommentStr;

    ::mmcFreeVersionInfo(vi);

    return comment;
}


/*
 * megamol::console::utility::AboutInfo::viewerVersionString
 */
vislib::StringA megamol::console::utility::AboutInfo::viewerVersionString(void) {
    vislib::StringA retval;
    int wordSize = 0;
    vislib::VersionNumber ver;
    const char *systemStr = "Unknown";
    vislib::StringA str;

    ::mmcBinaryVersionInfo *vi = ::mmvGetVersionInfo();
    if (vi == nullptr) return nullptr;

    ver.Set(vi->VersionNumber[0], vi->VersionNumber[1], vi->VersionNumber[2], vi->VersionNumber[3]);
    switch (vi->HardwareArchitecture) {
        case MMC_HARCH_I86: wordSize = 32; break;
        case MMC_HARCH_X64: wordSize = 64; break;
        default: wordSize = 0;
    }
    switch (vi->SystemType) {
        case MMC_OSYSTEM_WINDOWS: systemStr = "Windows"; break;
        case MMC_OSYSTEM_LINUX: systemStr = "Linux"; break;
        default: systemStr = "Unknown";
    }

    retval.Format("Viewer \"%s\" (Ver.: %s) %d Bit %s",
        vi->NameStr, ver.ToStringA().PeekBuffer(), wordSize, systemStr);

    if ((vi->Flags & MMC_BFLAG_DEBUG)) { str.Append(" DEBUG;"); }
    if ((vi->Flags & MMC_BFLAG_DIRTY)) { str.Append(" DIRTY;"); }
    if (!str.IsEmpty()) {
        str[0] = '[';
        str[str.Length() - 1] = ']';
        str.Prepend(' ');
        retval.Append(str);
    }

    ::mmvFreeVersionInfo(vi);

    return retval;
}


/*
 * megamol::console::utility::AboutInfo::viewerCommentString
 */
vislib::StringA megamol::console::utility::AboutInfo::viewerCommentString(void) {
    vislib::StringA comment;

    ::mmcBinaryVersionInfo *vi = ::mmvGetVersionInfo();
    if (vi == nullptr) return nullptr;

    comment = vi->CommentStr;

    ::mmvFreeVersionInfo(vi);

    return comment;
}
