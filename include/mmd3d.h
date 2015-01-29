/*
 * mmd3d.h
 *
 * Copyright (C) 2012 by Visualisierungsinstitut der Universität Stuttgart.
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_MMD3D_H_INCLUDED
#define MEGAMOLCORE_MMD3D_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

//#define MEGAMOLCORE_WITH_DIRECT3D11

#ifdef MEGAMOLCORE_WITH_DIRECT3D11

#ifndef _WIN32
#error "Are you sure?!"
#endif /* _WIN32 */

#include <D3D11.h>
#include <D3Dcompiler.h>
#include <xnamath.h>

#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "dxgi.lib")
#pragma comment(lib, "dxerr.lib")
#pragma comment(lib, "d3dx11.lib")
#pragma comment(lib, "d3dcompiler.lib")

#define NO_STEREO_D3D9
#define NO_STEREO_D3D10
#include "nvstereo.h"

#else /* MEGAMOLCORE_WITH_DIRECT3D11 */

#define ID3D11Device void
//#define ID3D11Buffer void
//#define ID3D11DeviceContext void
//#define ID3D11RenderTargetView void
//#define ID3D11Buffer void
//#define ID3DBlob void
//#define ID3D11VertexShader void
//#define ID3D11GeometryShader void
//#define ID3D11PixelShader void
//#define ID3D11Texture1D void
//#define ID3D11Texture2D void
//#define ID3D11InputLayout void
//#define ID3D11BlendState void
//#define ID3D11DepthStencilView void
//#define ID3D11DepthStencilState void
//#define ID3D11SamplerState void
//#define ID3D11ShaderResourceView void
//#define ID3D11RasterizerState void

#define NO_STEREO_D3D9
#define NO_STEREO_D3D10
#define NO_STEREO_D3D11

#endif /* MEGAMOLCORE_WITH_DIRECT3D11 */

#endif /* MEGAMOLCORE_MMD3D_H_INCLUDED */
