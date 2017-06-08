/*
 * AbstractD3D11RenderObject.h
 *
 * Copyright (C) 2013 by Visualisierungsinstitut der Universität Stuttgart.
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_ABSTRACTD3D11RENDEROBJECT_H_INCLUDED
#define MEGAMOLCORE_ABSTRACTD3D11RENDEROBJECT_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "mmcore/mmd3d.h"

#include "mmcore/utility/ShaderSourceFactory.h"


namespace megamol {
namespace core {
namespace utility {

    /**
     * This class is intended as super-class for all classes that represent or
     * render "objects" using D3D11, e.g. bounding boxes. It provides resource
     * management for commonly required D3D11 objects like the device, the
     * immediate context, or a vertex buffer.
     */
    class MEGAMOLCORE_API AbstractD3D11RenderObject {

    public:

        virtual ~AbstractD3D11RenderObject(void);

#ifdef MEGAMOLCORE_WITH_DIRECT3D11

        virtual HRESULT Finalise(void);

        virtual HRESULT Initialise(ID3D11Device *device);

    protected:

        AbstractD3D11RenderObject(void);

        /**
         * Compile the given shader from the BTF and hand it over to the caller
         * (caller must release the blob!).
         *
         * @param outBinary  Receives the compiled binary code.
         * @param factory    The shader factory
         * @param name       The name of the shader snippet to be compiled.
         * @param entryPoint The name of the entry point function.
         * @param target     The shader target version.
         *
         * @return 
         */
        static HRESULT compileFromBtf(ID3DBlob *& outBinary, 
            ShaderSourceFactory& factory, const char *name, 
            const char *entryPoint, const char *target);

        /**
         * Convenience method for allocating a D3D buffer.
         *
         * @param outBuffer
         * @param type
         * @param size
         * @param isDynamic
         * @param data
         *
         * @return
         */
        HRESULT createBuffer(ID3D11Buffer *& outBuffer, const UINT type,
            const UINT size, const bool isDynamic = true,
            const void *data = NULL);

        /**
         * Convenience method for initialising the 'geometryShader' member from
         * a BTF file.
         *
         * @param factory
         * @param name
         * @param entryPoint
         * @param target
         *
         * @return
         */
        HRESULT createGeometryShaderFromBtf(ShaderSourceFactory& factory, 
            const char *name, const char *entryPoint, const char *target);

        /**
         * Convenience method for initialising the 'indexBuffer' member.
         *
         * @param size
         * @param isDynamic
         * @param data
         *
         * @return
         */
        HRESULT createIndexBuffer(const UINT size, const bool isDynamic = true,
            const void *data = NULL);

        /**
         * Convenience method for initialising the 'pixelShader' member from
         * a BTF file.
         *
         * @param factory
         * @param name
         * @param entryPoint
         * @param target
         *
         * @return
         */
        HRESULT createPixelShaderFromBtf(ShaderSourceFactory& factory, 
            const char *name, const char *entryPoint, const char *target);

        HRESULT createSamplerState(ID3D11SamplerState *& outSamplerState,
            const D3D11_FILTER filter);

        /**
         * Convenience method for initialising the 'vertexBuffer' member.
         *
         * @param size
         * @param isDynamic
         * @param data
         * 
         * @return
         */
        HRESULT createVertexBuffer(const UINT size, const bool isDynamic = true,
            const void *data = NULL);

        /**
         * Convenience method for initialising the 'vertexShader' and the 
         * 'inputLayout' member from a BTF file.
         *
         * @param factory
         * @param name
         * @param entryPoint
         * @param target
         * @param desc
         * @param cntDesc
         *
         * @return
         */
        HRESULT createVertexShaderAndInputLayoutFromBtf(
            ShaderSourceFactory& factory, 
            const char *name, const char *entryPoint, const char *target,
            D3D11_INPUT_ELEMENT_DESC *desc, const UINT cntDesc);

        /** The blending state. */
        ID3D11BlendState *blendState;

        /** The depth/stencil buffer state. */
        ID3D11DepthStencilState *depthStencilState;

        /** The D3D device we are rendering to. */
        ID3D11Device *device;

        /** The geometry shader used for rendering the object. */
        ID3D11GeometryShader *geometryShader;

        /** The immediate context of 'device'. */
        ID3D11DeviceContext *immediateContext;

        /** The index buffer for the stuff to be drawn. */
        ID3D11Buffer *indexBuffer;

        /** The input layout of the content of 'vertexBuffer'. */
        ID3D11InputLayout *inputLayout;

        /** The pixel shader used for rendering the object. */
        ID3D11PixelShader *pixelShader;

        /** The vertex buffer for the stuff to be draw. */
        ID3D11Buffer *vertexBuffer;

        /** The vertes shader used for rendering the object. */
        ID3D11VertexShader *vertexShader;

#endif MEGAMOLCORE_WITH_DIRECT3D11

    };

} /* end namespace utility */
} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_ABSTRACTD3D11RENDEROBJECT_H_INCLUDED */
