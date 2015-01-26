#
# genDynIncl.pl
#
# Copyright (C) 2008 by Universitaet Stuttgart (VIS).
# Alle Rechte vorbehalten.
#

my @allFuncNames = ();
my $funcPtrTombstone;

while (<>) {
    chomp;
    # MEGAMOLVIEWER_API unsigned int MEGAMOLVIEWER_CALL(mmvGetHandleSize)(void) MEGAMOLVIEWER_INIT;
    # if (/^\s*MEGAMOLVIEWER_API\s+(.+)\s+MEGAMOLVIEWER_CALL\((\w+)\)\s*(\([^;]+);/) {
    if (/^\s*MEGAMOLVIEWER_API\s+(.+)\s+MEGAMOLVIEWER_CALL\((\w+)\)/) {
        push @allFuncNames, $2;
    }
}


open INCL, ">include/MegaMolViewerDynamic.h" || die "could not create \"MegaMolViewerDynamic.h\"";

print INCL qq{/*
 * MegaMolViewerDynamic.h
 *
 * Copyright (C) 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLVIEWER_MEGALMOLVIEWERDYNAMIC_H_INCLUDED
#define MEGAMOLVIEWER_MEGALMOLVIEWERDYNAMIC_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#ifndef MEGAMOLVIEWER_MEGALMOLVIEWER_H_INCLUDED
#error You must not include MegaMolViewerDynamic.h directly. Always include MegaMolViewer.h
#endif /* !MEGAMOLVIEWER_MEGALMOLVIEWER_H_INCLUDED */

#define MEGAMOLVIEWER_API typedef
#ifdef _WIN32
#   define MEGAMOLVIEWER_CALL(F) (__cdecl * F##_FNPTRT)
#else /* _WIN32 */
#   define MEGAMOLVIEWER_CALL(F) (* F##_FNPTRT)
#endif /* _WIN32 */

/*
 * Include static header for type definitions 
 */
#include "MegaMolViewerStatic.h"

#ifdef MEGAMOLVIEWER_SINGLEFILE
#   define MEGAMOLVIEWER_PTR(F) F##_FNPTRT F = 0;
#else /* MEGAMOLVIEWER_SINGLEFILE */
#   define MEGAMOLVIEWER_PTR(F) extern F##_FNPTRT F;
#endif /* MEGAMOLVIEWER_SINGLEFILE */

/*
 * function pointers
 */
};

foreach $f (@allFuncNames) {
    print INCL "MEGAMOLVIEWER_PTR(" . $f . ")\n";
}

print INCL qq§
/*
 * Helper functions
 */
#ifndef MEGAMOLVIEWER_SINGLEFILE
namespace megamol {
namespace viewer {

    /**
     * Answer whether or not the library is loaded.
     *
     * @return 'true' if the library is loaded, 'false' otherwise.
     */
    extern bool mmvIsLibraryLoaded(void);

    /**
     * Load the MegaMol™ Viewer library with the specified name.
     *
     * @param libfilename The name of the library to load.
     *
     * @return 'true' if the library was successfully loaded, 'false' otherwise.
     */
    extern bool mmvLoadLibraryA(const char* filename);

    /**
     * Load the MegaMol™ Viewer library with the specified name.
     *
     * @param libfilename The name of the library to load.
     *
     * @return 'true' if the library was successfully loaded, 'false' otherwise.
     */
    extern bool mmvLoadLibraryW(const wchar_t* filename);

    /**
     * Unload the MegaMol™ Viewer library. This method should only be used if
     * it is really necessary. Usually it is not, because the library will be
     * released on application exit.
     */
    extern void mmvUnloadLibrary(void);
    
} /* end namespace viewer */
} /* end namespace megamol */
#endif /* !MEGAMOLVIEWER_SINGLEFILE */

#if defined(UNICODE) || defined(_UNICODE)
#define mmvLoadLibrary mmvLoadLibraryW
#else /* defined(UNICODE) || defined(_UNICODE) */
#define mmvLoadLibrary mmvLoadLibraryA
#endif /* defined(UNICODE) || defined(_UNICODE) */

#ifdef MEGAMOLVIEWER_SINGLEFILE

#include "vislib/sys/DynamicLinkLibrary.h"
#include "vislib/functioncast.h"

namespace megamol {
namespace viewer {

    static vislib::sys::DynamicLinkLibrary __mmvLib;

    /** forward declaration */
    void mmvUnloadLibrary(void);


    /**
     * Answer whether or not the library is loaded.
     *
     * @return 'true' if the library is loaded, 'false' otherwise.
     */
    bool mmvIsLibraryLoaded(void) {
        return (__mmvLib.IsLoaded()
§;

foreach $f (@allFuncNames) {
    print INCL "            && (" . $f . " != NULL)\n";
}

print INCL qq§            );
    }


    /**
     * Load the MegaMol™ Viewer library with the specified name.
     *
     * @param libfilename The name of the library to load.
     *
     * @return 'true' if the library was successfully loaded, 'false' otherwise.
     */
    bool mmvLoadLibraryA(const char* filename) {
        try {
            if (__mmvLib.IsLoaded()) {
                __mmvLib.Free();
            }
            if (!__mmvLib.Load(filename)) {
                throw 0;
            }
§;

foreach $f (@allFuncNames) {
    print INCL "            " . $f . " = function_cast<" . $f . "_FNPTRT>(__mmvLib.GetProcAddress(\"" . $f . "\"));\n";
}

print INCL qq§
            if (mmvIsLibraryLoaded()) {
                return true;
            }
        } catch(...) {
        }
        mmvUnloadLibrary();
        return false;
    }

    /**
     * Load the MegaMol™ Viewer library with the specified name.
     *
     * @param libfilename The name of the library to load.
     *
     * @return 'true' if the library was successfully loaded, 'false' otherwise.
     */
    bool mmvLoadLibraryW(const wchar_t* filename) {
        try {
            if (__mmvLib.IsLoaded()) {
                __mmvLib.Free();
            }
            if (!__mmvLib.Load(filename)) {
                throw 0;
            }
§;

foreach $f (@allFuncNames) {
    print INCL "            " . $f . " = function_cast<" . $f . "_FNPTRT>(__mmvLib.GetProcAddress(\"" . $f . "\"));\n";
}

print INCL qq§
            if (mmvIsLibraryLoaded()) {
                return true;
            }
        } catch(...) {
        }
        mmvUnloadLibrary();
        return false;
    }
§;
if (defined $funcPtrTombstone) {
print INCL qq§

extern "C" {

    /**
     * Capture Bullshit
     */
    void __cdecl mmvCaptureBullshit(void) {
    }
    
}
§;
}
print INCL qq§

    /**
     * Unload the MegaMol™ Viewer library. This method should only be used if
     * it is really necessary. Usually it is not, because the library will be
     * released on application exit.
     */
    void mmvUnloadLibrary(void) {
        try {
            if (__mmvLib.IsLoaded()) {
                __mmvLib.Free();
            }
        } catch(...) {
        }
§;

foreach $f (@allFuncNames) {
    print INCL "        " . $f . " = NULL;\n";
    if (defined $funcPtrTombstone) {
        print INCL "        " . $f . " = function_cast<" . $f . "_FNPTRT>(function_cast<void>(mmvCaptureBullshit));\n";
    }
}

print INCL qq§    }

} /* end namespace viewer */
} /* end namespace megamol */

#endif /* MEGAMOLVIEWER_SINGLEFILE */

#undef MEGAMOLVIEWER_SINGLEFILE

#endif /* MEGAMOLVIEWER_MEGALMOLVIEWERDYNAMIC_H_INCLUDED */
§;

close INCL
