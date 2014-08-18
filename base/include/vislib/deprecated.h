/*
 * deprecated.h
 *
 * Copyright (C) 2009 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#ifndef VISLIB_DEPRECATED_H_INCLUDED
#define VISLIB_DEPRECATED_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#if (defined(_WIN32) && defined(_MSC_VER))
#define VLDEPRECATED __declspec(deprecated)
#else /* (defined(_WIN32) && defined(_MSC_VER)) */
#define VLDEPRECATED __attribute__((deprecated))
#endif /* (defined(_WIN32) && defined(_MSC_VER)) */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_DEPRECATED_H_INCLUDED */
