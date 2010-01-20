/*
 * MegaMolViewer.h
 *
 * Copyright (C) 2008 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLVIEWER_MEGALMOLVIEWER_H_INCLUDED
#define MEGAMOLVIEWER_MEGALMOLVIEWER_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

/*
 * MegaMol™ Viewer include header.
 * You can include this header for 'static' or 'dynamic' usage.
 *
 * For 'static' usage define 'MEGAMOLVIEWER_USESTATIC' and link your 
 * application against the import library of this library.
 *
 * For 'dynamic' usage do not define 'MEGAMOLVIEWER_USESTATIC'. The library 
 * will then be loaded manually on runtime by calling 'mmvLoadLibrary'. You
 * must define 'MEGAMOLVIEWER_SINGLEFILE' from exactly ONE file to generate
 * the required symbols in the corresponding object file.
 *
 * Do not mix 'static' and 'dynamic' usage!
 */

#if defined(MEGAMOLVIEWER_EXPORTS) && !defined(MEGAMOLVIEWER_USESTATIC)
#error You must define "MEGAMOLVIEWER_USESTATIC" if you create the library
#endif

#ifdef MEGAMOLVIEWER_USESTATIC
#   include "MegaMolViewerStatic.h"
#else /* MEGAMOLVIEWER_USESTATIC */
#   include "MegaMolViewerDynamic.h"
#endif /* MEGAMOLVIEWER_USESTATIC */

#endif /* MEGAMOLVIEWER_MEGALMOLVIEWER_H_INCLUDED */
