/*
 * D3D9TestBoxGeometry.h
 *
 * Copyright (C) 2009 by Universitaet Stuttgart (VISUS). 
 * Alle Rechte vorbehalten.
 */
#ifndef VISLIBTEST_D3D9TESTBOXGEOMETRY_H_INCLUDED
#define VISLIBTEST_D3D9TESTBOXGEOMETRY_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#include <d3dx9shape.h>


class D3D9TestBoxGeometry {

public:

    D3D9TestBoxGeometry(void);

    ~D3D9TestBoxGeometry(void);

    void Create(IDirect3DDevice9 *device);

    void Draw(void);

    void Release(void);

private:

    LPD3DXMESH boxX;

    LPD3DXMESH boxY;

    LPD3DXMESH boxZ;
};

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIBTEST_D3D9D3DSIMPLECAMERATEST_H_INCLUDED */
