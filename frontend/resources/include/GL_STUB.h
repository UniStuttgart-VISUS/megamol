/*
 * GL_STUB.h
 *
 * Copyright (C) 2021 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#pragma once

#ifndef GL_STUB
#ifdef WITH_GL
#define GL_STUB(value)
#else // WITH_GL
#define GL_STUB(value) \
    { return value; }
#endif // WITH_GL
#endif // GL_STUB
