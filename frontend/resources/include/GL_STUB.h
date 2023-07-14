/*
 * GL_STUB.h
 *
 * Copyright (C) 2021 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#pragma once

#ifndef GL_STUB
#ifdef MEGAMOL_USE_OPENGL
#define GL_STUB(value)
#else // MEGAMOL_USE_OPENGL
#define GL_STUB(value) \
    { return value; }
#endif // MEGAMOL_USE_OPENGL
#endif // GL_STUB

#ifndef GL_VSTUB
#ifdef MEGAMOL_USE_OPENGL
#define GL_VSTUB()
#else // MEGAMOL_USE_OPENGL
#define GL_VSTUB() \
    { return; }
#endif // MEGAMOL_USE_OPENGL
#endif // GL_VSTUB
