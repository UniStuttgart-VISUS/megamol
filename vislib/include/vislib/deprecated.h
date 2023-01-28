/*
 * deprecated.h
 *
 * Copyright (C) 2009 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#pragma once
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
