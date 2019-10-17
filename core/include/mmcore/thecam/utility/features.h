/*
 * thecam/config/features.h
 *
 * Copyright (c) 2012, TheLib Team (http://www.thelib.org/license)
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * - Redistributions of source code must retain the above copyright notice,
 *   this list of conditions and the following disclaimer.
 * - Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 * - Neither the name of TheLib, TheLib Team, nor the names of its
 *   contributors may be used to endorse or promote products derived from this
 *   software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THELIB TEAM AS IS AND ANY EXPRESS OR IMPLIED
 * WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO
 * EVENT SHALL THELIB TEAM BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
 * TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
#ifndef THE_FEATURES_H_INCLUDED
#define THE_FEATURES_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#    pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#    pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */

/* ensure the definds from 'config.h' are available */
#ifndef THE_CONFIG_H_INCLUDED
#    error "Do not include 'features.h' directly. Instead include 'config.h'."
#endif /* THE_CONFIG_H_INCLUDED */


//#ifdef THE_WINDOWS
///* Enable support for DirectX 11 features in TheLib. */
//#define WITH_THE_DIRECTX11 (1)
//#endif /* THE_WINDOWS */

//#ifdef THE_WINDOWS
///* Enable support for Windows Media Foundation. */
//#define WITH_THE_MEDIA_FOUNDATION (1)
//#endif /* THE_WINDOWS */


//#if (defined(_WIN32_WINNT) && (_WIN32_WINNT >= 0x0A00))
///* Enable support for DirectX 11 features in TheLib. */
//#define WITH_THE_DIRECTX12 (1)
//#endif /* (defined(_WIN32_WINNT) && (_WIN32_WINNT >= 0x0A00)) */

/* Enable OpenGL features of TheLib. */
#define WITH_THE_OPENGL (1)

/* Enable Network Direct features if the SDK is installed. */
#ifdef HAVE_THE_NETWORK_DIRECT
#    define WITH_THE_NETWORK_DIRECT
#endif /* HAVE_THE_NETWORK_DIRECT */


#ifdef THE_WINDOWS
/* If Win7 or newer */
#    define WITH_THE_WINDOWS_7 (1)
#endif /* THE_WINDOWS */

//#define WITH_THE_VULKAN

#if defined(_WIN32) && defined(_MANAGED)
#    pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* THE_FEATURES_H_INCLUDED */
