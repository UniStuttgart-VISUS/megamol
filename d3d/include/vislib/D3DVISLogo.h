/*
 * D3DVISLogo.h
 *
 * Copyright (C) 2006 - 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 * Copyright (C) 2008 by Christoph Müller. Alle Rechte vorbehalten.
 */

#ifndef VISLIB_D3DVISLOGO_H_INCLUDED
#define VISLIB_D3DVISLOGO_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */

#include <d3d9.h>
#include <d3d10.h>

#include "vislib/AbstractVISLogo.h"
#include "vislib/d3dutils.h"


namespace vislib {
namespace graphics {
namespace d3d {


    /**
     * Utility class for rendering a VIS logo in Direct3D.
     *
     * The class uses the managed pool for storing data, therefore lost devices
     * need not to be handled.
     */
    class D3DVISLogo : public AbstractVISLogo {

    public:

        /**
         * Create a VIS logo to be rendered on a Direct3D 9 device.
         */
        D3DVISLogo(IDirect3DDevice9 *device);

        /**
         * Create a VIS logo to be rendered on a Direct3D 10 device.
         */
        D3DVISLogo(ID3D10Device *device);

        /** Dtor. */
        ~D3DVISLogo(void);

        /**
         * Create all required resources for rendering a VIS logo.
         *
         * @throws D3DException In case the resources could not be allocated.
         */
        virtual void Create(void);

        /**
         * Render the VIS logo. Create() must have been called before.
         *
         * @throws D3DException In case the VIS logo could not be rendered.
         */
        virtual void Draw(void);

        /**
         * Release all resources of the VIS logo.
         */
        virtual void Release(void);

    private:

        /** The vertex format used for the VIS logo. */
        typedef struct Vertex_t {
            FLOAT x, y, z;      // Position.
            FLOAT nx, ny, nz;   // Normal.
            DWORD diffuse;      // Diffure colour.
        } Vertex;

        /** The number of vertices the VIS logo consists of. */
        static const Vertex VERTICES[9562];

        /** Flexible vertex format for VIS logo. */
        static const DWORD FVF;

        /** Size of our vertex vertex format. */
        static const UINT VERTEX_SIZE;

        /** The Direct3D API version the object uses. */
        ApiVersion apiVersion;

        /** The device to create and render the logo on. */
        union {
            IDirect3DDevice9 *device9;
            ID3D10Device *device10;
        };

        /** The vertex buffer holding the VIS logo. */
        union {
            IDirect3DVertexBuffer9 *vb9;
            ID3D10Buffer *vb10;
        };

    };

} /* end namespace d3d */
} /* end namespace graphics */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_D3DVISLOGO_H_INCLUDED */
