/*
 * AbstractD3D11RenderObject.cpp
 *
 * Copyright (C) 2013 by Visualisierungsinstitut der Universität Stuttgart.
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "AbstractD3D11RenderObject.h"

#include "vislib/assert.h"
#include "vislib/d3dutils.h"
#include "vislib/IllegalParamException.h"
#include "vislib/Log.h"
#include "vislib/StackTrace.h"


/*
 * megamol::core::utility::AbstractD3D11RenderObject::~AbstractD3D11RenderObject
 */
megamol::core::utility::AbstractD3D11RenderObject::~AbstractD3D11RenderObject(
        void) {
    VLAUTOSTACKTRACE;
#ifdef MEGAMOLCORE_WITH_DIRECT3D11
    this->Finalise();
#endif /* MEGAMOLCORE_WITH_DIRECT3D11 */
}


#ifdef MEGAMOLCORE_WITH_DIRECT3D11
/*
 * megamol::core::utility::AbstractD3D11RenderObject::Finalise
 */
HRESULT megamol::core::utility::AbstractD3D11RenderObject::Finalise(void) {
    VLAUTOSTACKTRACE;
    SAFE_RELEASE(this->blendState);
    SAFE_RELEASE(this->depthStencilState);
    SAFE_RELEASE(this->device);
    SAFE_RELEASE(this->geometryShader);
    SAFE_RELEASE(this->immediateContext);
    SAFE_RELEASE(this->indexBuffer);
    SAFE_RELEASE(this->inputLayout);
    SAFE_RELEASE(this->pixelShader);
    SAFE_RELEASE(this->vertexBuffer);
    SAFE_RELEASE(this->vertexShader);
    return S_OK;
}


/*
 * megamol::core::utility::AbstractD3D11RenderObject::Initialise
 */
HRESULT megamol::core::utility::AbstractD3D11RenderObject::Initialise(
        ID3D11Device *device) {
    VLAUTOSTACKTRACE;

    if (device == NULL) {
        throw vislib::IllegalParamException("device", __FILE__, __LINE__);
    }

    HRESULT hr = this->Finalise();

    if (SUCCEEDED(hr)) {
        device->AddRef();
        this->device = device;

        this->device->GetImmediateContext(&this->immediateContext);
    }

    return hr;
}


/* 
 * megamol::core::utility::AbstractD3D11RenderObject::AbstractD3D11RenderObject
 */
megamol::core::utility::AbstractD3D11RenderObject::AbstractD3D11RenderObject(
        void) : blendState(NULL), depthStencilState(NULL), device(NULL),
        geometryShader(NULL), immediateContext(NULL), indexBuffer(NULL),
        inputLayout(NULL), pixelShader(NULL), vertexBuffer(NULL),
        vertexShader(NULL) {
    VLAUTOSTACKTRACE;
}


/*
 * megamol::core::utility::AbstractD3D11RenderObject::compileFromBtf
 */
HRESULT megamol::core::utility::AbstractD3D11RenderObject::compileFromBtf(
        ID3DBlob *& outBinary, ShaderSourceFactory& factory, 
        const char *name, const char *entryPoint, const char *target) {
    VLAUTOSTACKTRACE;

    UINT flags = 0;                     // Shader compiler flags.
    HRESULT hr = S_OK;                  // API call result.
    ID3DBlob *msg = NULL;               // Possible error message.
    vislib::graphics::gl::ShaderSource shader;
    vislib::StringA source;             // HLSL source code.

    if (outBinary != NULL) {
        hr = E_INVALIDARG;
    }
    if ((name == NULL) || (entryPoint == NULL) || (target == NULL)) {
        hr = E_POINTER;
    }

    flags |= D3DCOMPILE_PACK_MATRIX_COLUMN_MAJOR;
    //flags |= D3DCOMPILE_PACK_MATRIX_ROW_MAJOR;
#if (defined(DEBUG) || defined(_DEBUG))
    flags |= D3DCOMPILE_DEBUG | D3DCOMPILE_SKIP_OPTIMIZATION;
#endif /* (defined(DEBUG) || defined(_DEBUG)) */
    
    if (SUCCEEDED(hr)) {
        if (!factory.MakeShaderSource(name, shader, 
                ShaderSourceFactory::FLAGS_HLSL_LINE_PRAGMAS)) {
            vislib::sys::Log::DefaultLog.WriteError(_T("Composing shader ")
                _T("source from BTF failed."));
            hr = E_FAIL;
        }
    }

    if (SUCCEEDED(hr)) {
        source = shader.WholeCode();
        hr = ::D3DCompile(source.PeekBuffer(), source.Length() * sizeof(char),
            //name, 
            "T:\\Programmcode\\MegaMol\\core-d3d11\\Shaders\\d3d11sphere.btf",
            NULL, NULL, entryPoint, target, flags, 0, &outBinary, &msg);
        if (FAILED(hr)) {
            vislib::sys::Log::DefaultLog.WriteError(_T("Compiling shader ")
                _T("failed with error code %u"), hr);
        }
    }

    if (msg != NULL) {
        vislib::sys::Log::DefaultLog.WriteWarn(_T("%hs"),
            static_cast<const char *>(msg->GetBufferPointer()));
    }
    

    SAFE_RELEASE(msg);
    return hr;
}


/*
 * megamol::core::utility::AbstractD3D11RenderObject::createBuffer
 */
HRESULT megamol::core::utility::AbstractD3D11RenderObject::createBuffer(
        ID3D11Buffer *& outBuffer, const UINT type, const UINT size,
        const bool isDynamic, const void *data) {
    VLAUTOSTACKTRACE;
    ASSERT(this->device != NULL);

    D3D11_BUFFER_DESC desc;
    HRESULT hr = S_OK;
    D3D11_SUBRESOURCE_DATA initData;

    if (outBuffer != NULL) {
        hr = E_INVALIDARG;
    }
    if ((data == NULL) && !isDynamic) {
        hr = E_INVALIDARG;
    }

    if (SUCCEEDED(hr)) {
        ::ZeroMemory(&desc, sizeof(desc));
        ::ZeroMemory(&initData, sizeof(initData));

        desc.ByteWidth = size;
        desc.BindFlags = type;

        if (isDynamic) {
            desc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
            desc.Usage = D3D11_USAGE_DYNAMIC;
        } else {
            desc.Usage = D3D11_USAGE_DEFAULT;
        }

        if (data != NULL) {
            initData.pSysMem = data;
            initData.SysMemPitch = size;

            hr = this->device->CreateBuffer(&desc, &initData, &outBuffer);
        } else {
            hr = this->device->CreateBuffer(&desc, NULL, &outBuffer);
        }
    }

    return hr;
}


/*
 * megamol::core::utility::AbstractD3D11RenderObject::createGeometryShaderFromBtf
 */
HRESULT
megamol::core::utility::AbstractD3D11RenderObject::createGeometryShaderFromBtf(
        ShaderSourceFactory& factory, const char *name, 
        const char *entryPoint, const char *target) {
    VLAUTOSTACKTRACE;
    ASSERT(this->device != NULL);

    ID3DBlob *binary = NULL;
    HRESULT hr = S_OK;

    if (this->geometryShader != NULL) {
        vislib::sys::Log::DefaultLog.WriteError(_T("Geometry shader has ")
            _T("already been initialised."));
        hr = E_POINTER;
    }

    if (SUCCEEDED(hr)) {
        hr = this->compileFromBtf(binary, factory, name, entryPoint, target);
    }
    if (SUCCEEDED(hr)) {
        hr = this->device->CreateGeometryShader(binary->GetBufferPointer(), 
            binary->GetBufferSize(), NULL, &this->geometryShader);
    }

    SAFE_RELEASE(binary);
    return hr;
}


/*
 * megamol::core::utility::AbstractD3D11RenderObject::createIndexBuffer
 */
HRESULT megamol::core::utility::AbstractD3D11RenderObject::createIndexBuffer(
        const UINT size, const bool isDynamic, const void *data) {
    VLAUTOSTACKTRACE;

    HRESULT hr = S_OK;

    if (this->indexBuffer != NULL) {
        vislib::sys::Log::DefaultLog.WriteError(_T("Index buffer has already ")
            _T("been initialised."));
        hr = E_POINTER;
    }
        
    if (SUCCEEDED(hr)) {
        hr = this->createBuffer(this->indexBuffer, D3D11_BIND_INDEX_BUFFER, 
            size, isDynamic, data);
        if (FAILED(hr)) {
            vislib::sys::Log::DefaultLog.WriteError(_T("Allocating index ")
                _T("buffer failed with error code %u"), hr);
        }
    }


    return hr;
}


/*
 * megamol::core::utility::AbstractD3D11RenderObject::createPixelShaderFromBtf
 */
HRESULT
megamol::core::utility::AbstractD3D11RenderObject::createPixelShaderFromBtf(
        ShaderSourceFactory& factory, const char *name, 
        const char *entryPoint, const char *target) {
    VLAUTOSTACKTRACE;
    ASSERT(this->device != NULL);

    ID3DBlob *binary = NULL;
    HRESULT hr = S_OK;

    if (this->pixelShader != NULL) {
        vislib::sys::Log::DefaultLog.WriteError(_T("Pixel shader has ")
            _T("already been initialised."));
        hr = E_POINTER;
    }

    if (SUCCEEDED(hr)) {
        hr = this->compileFromBtf(binary, factory, name, entryPoint, target);
    }
    if (SUCCEEDED(hr)) {
        hr = this->device->CreatePixelShader(binary->GetBufferPointer(), 
            binary->GetBufferSize(), NULL, &this->pixelShader);
    }

    SAFE_RELEASE(binary);
    return hr;
}


/*
 * megamol::core::utility::AbstractD3D11RenderObject::createSamplerState
 */
HRESULT megamol::core::utility::AbstractD3D11RenderObject::createSamplerState(
        ID3D11SamplerState *& outSamplerState, const D3D11_FILTER filter) {
    VLAUTOSTACKTRACE;
    ASSERT(this->device != NULL);

    D3D11_SAMPLER_DESC desc;
    HRESULT hr = S_OK;

    if (SUCCEEDED(hr)) {
        ::ZeroMemory(&desc, sizeof(desc));
        // Mostly default as defined in http://msdn.microsoft.com/en-us/library/windows/desktop/ff476207(v=vs.85).aspx 
        desc.AddressU = D3D11_TEXTURE_ADDRESS_CLAMP;
        desc.AddressV = D3D11_TEXTURE_ADDRESS_CLAMP;
        desc.AddressW = D3D11_TEXTURE_ADDRESS_CLAMP;
        //desc.BorderColor
        desc.ComparisonFunc = D3D11_COMPARISON_NEVER;
        desc.Filter = filter;
        desc.MaxAnisotropy = 16;
        desc.MaxLOD = FLT_MAX;
        desc.MinLOD = -FLT_MAX;
        desc.MipLODBias = 0.0f;

        hr = this->device->CreateSamplerState(&desc, &outSamplerState);
        if (FAILED(hr)) {
            vislib::sys::Log::DefaultLog.WriteError(_T("Creating sampler ")
                _T("state failed with error code %u."), hr);
        }
    }

    return hr;
}


/*
 * megamol::core::utility::AbstractD3D11RenderObject::createVertexBuffer
 */
HRESULT megamol::core::utility::AbstractD3D11RenderObject::createVertexBuffer(
        const UINT size, const bool isDynamic, const void *data) {
    VLAUTOSTACKTRACE;

    HRESULT hr = S_OK;

    if (this->vertexBuffer != NULL) {
        vislib::sys::Log::DefaultLog.WriteError(_T("Vertex buffer has already ")
            _T("been initialised."));
        hr = E_POINTER;
    }
        
    if (SUCCEEDED(hr)) {
        hr = this->createBuffer(this->vertexBuffer, D3D11_BIND_VERTEX_BUFFER,
            size, isDynamic, data);
        if (FAILED(hr)) {
            vislib::sys::Log::DefaultLog.WriteError(_T("Allocating vertex ")
                _T("buffer failed with error code %u"), hr);
        }
    }


    return hr;
}


/*
 * ...::AbstractD3D11RenderObject::createVertexShaderFromBtfAndInputLayout
 */
HRESULT megamol::core::utility::AbstractD3D11RenderObject
        ::createVertexShaderAndInputLayoutFromBtf(
        ShaderSourceFactory& factory, const char *name,
        const char *entryPoint, const char *target,
        D3D11_INPUT_ELEMENT_DESC *desc, const UINT cntDesc) {
    VLAUTOSTACKTRACE;
    ASSERT(this->device != NULL);

    ID3DBlob *binary = NULL;
    HRESULT hr = S_OK;

    if (desc == NULL) {
        hr = E_POINTER;
    }
    if (this->vertexShader != NULL) {
        vislib::sys::Log::DefaultLog.WriteError(_T("Vertex shader has ")
            _T("already been initialised."));
        hr = E_POINTER;
    }
    if (this->inputLayout != NULL) {
        vislib::sys::Log::DefaultLog.WriteError(_T("Input layout has ")
            _T("already been initialised."));
        hr = E_POINTER;
    }

    if (SUCCEEDED(hr)) {
        hr = this->compileFromBtf(binary, factory, name, entryPoint, target);
    }
    if (SUCCEEDED(hr)) {
        hr = this->device->CreateVertexShader(binary->GetBufferPointer(), 
            binary->GetBufferSize(), NULL, &this->vertexShader);
    }


    if (SUCCEEDED(hr)) {
        hr = this->device->CreateInputLayout(desc, cntDesc, 
            binary->GetBufferPointer(), binary->GetBufferSize(),
            &this->inputLayout);
        if (FAILED(hr)) {
            vislib::sys::Log::DefaultLog.WriteError(_T("Creating input ")
                _T("layout failed with error code %u"), hr);
        }
    }
    
    SAFE_RELEASE(binary);
    return hr;
}

#endif /* MEGAMOLCORE_WITH_DIRECT3D11 */
