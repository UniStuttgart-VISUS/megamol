/**
 * MegaMol
 * Copyright (c) 2021, MegaMol Dev Team
 * All rights reserved.
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
