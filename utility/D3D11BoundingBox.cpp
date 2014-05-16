/*
 * AbstractD3D11RenderObject.cpp
 *
 * Copyright (C) 2013 by Visualisierungsinstitut der Universität Stuttgart.
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "D3D11BoundingBox.h"

#ifdef MEGAMOLCORE_WITH_DIRECT3D11
#include "vislib/assert.h"
#include "vislib/d3dutils.h"
#include "vislib/IllegalParamException.h"
#include "vislib/Log.h"
#include "vislib/StackTrace.h"


/* 
 * megamol::core::utility::D3D11BoundingBox::D3D11BoundingBox
 */
megamol::core::utility::D3D11BoundingBox::D3D11BoundingBox(void) : Base(),
        cbConstants(NULL), rasteriserStateBack(NULL), 
        rasteriserStateFront(NULL) {
    VLAUTOSTACKTRACE;
}


/*
 * megamol::core::utility::D3D11BoundingBox::~D3D11BoundingBox
 */
megamol::core::utility::D3D11BoundingBox::~D3D11BoundingBox(
        void) {
    VLAUTOSTACKTRACE;
    this->Finalise();
}


/*
 * megamol::core::utility::D3D11BoundingBox::Draw
 */
HRESULT megamol::core::utility::D3D11BoundingBox::Draw(const bool frontSides) {
    VLAUTOSTACKTRACE;
    ASSERT(this->immediateContext != NULL);

    HRESULT hr = S_OK;
    UINT offsets = 0;
    UINT strides = sizeof(XMVECTOR);

    this->immediateContext->IASetInputLayout(this->inputLayout);
    this->immediateContext->IASetPrimitiveTopology(
        D3D11_PRIMITIVE_TOPOLOGY_LINELIST);
        //D3D11_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP);
        //D3D11_PRIMITIVE_TOPOLOGY_POINTLIST);
    this->immediateContext->IASetVertexBuffers(0, 1, &this->vertexBuffer, 
        &strides, &offsets);
    this->immediateContext->IASetIndexBuffer(this->indexBuffer, 
        DXGI_FORMAT_R16_UINT, 0);

    this->immediateContext->VSSetShader(this->vertexShader, NULL, 0);
    this->immediateContext->VSSetConstantBuffers(0, 1, &this->cbConstants);

    this->immediateContext->GSSetShader(NULL, NULL, 0);

    this->immediateContext->PSSetShader(this->pixelShader, NULL, 0);
    this->immediateContext->PSSetConstantBuffers(0, 1, &this->cbConstants);

    if (frontSides) {
        this->immediateContext->RSSetState(this->rasteriserStateFront);
    } else {
        this->immediateContext->RSSetState(this->rasteriserStateBack);
    }

    //this->immediateContext->DrawIndexed(36, 0, 0);
    this->immediateContext->DrawIndexed(24, 0, 0);
    return hr;
}


/*
 * megamol::core::utility::D3D11BoundingBox::Finalise
 */
HRESULT megamol::core::utility::D3D11BoundingBox::Finalise(void) {
    VLAUTOSTACKTRACE;
    HRESULT hr = Base::Finalise();

    SAFE_RELEASE(this->cbConstants);
    SAFE_RELEASE(this->rasteriserStateBack);
    SAFE_RELEASE(this->rasteriserStateFront);

    return hr;
}


/*
 * megamol::core::utility::D3D11BoundingBox::Initialise
 */
HRESULT megamol::core::utility::D3D11BoundingBox::Initialise(
        ID3D11Device *device, utility::ShaderSourceFactory& factory) {
    VLAUTOSTACKTRACE;
    HRESULT hr = Base::Initialise(device);

    D3D11_INPUT_ELEMENT_DESC ilDesc[] = {
        { "POSITION", 0, DXGI_FORMAT_R32G32B32A32_FLOAT, 0, 0, D3D11_INPUT_PER_VERTEX_DATA, 0 }
    };
    D3D11_RASTERIZER_DESC rasteriserDesc;
    float x = 1.0f / 2.0f;
    float y = 1.0f / 2.0f;
    float z = 1.0f / 2.0f;
    float w = 1.0f;

    /* Create vertex and index buffers. */
    if (SUCCEEDED(hr)) {
        //XMVECTORF32 vertices[] = {
        //    { -x,  y, -z, w },
        //    {  x,  y, -z, w },
        //    { -x, -y, -z, w },
        //    {  x, -y, -z, w },
        //    { -x,  y,  z, w },
        //    {  x,  y,  z, w },
        //    { -x, -y,  z, w },
        //    {  x, -y,  z, w }
        //};
        XMVECTORF32 vertices[] = {
            { 0.0f, 1.0f, 0.0f, 1.0f },
            { 1.0f, 1.0f, 0.0f, 1.0f },
            { 0.0f, 0.0f, 0.0f, 1.0f },
            { 1.0f, 0.0f, 0.0f, 1.0f },
            { 0.0f, 1.0f, 1.0f, 1.0f },
            { 1.0f, 1.0f, 1.0f, 1.0f },
            { 0.0f, 0.0f, 1.0f, 1.0f },
            { 1.0f, 0.0f, 1.0f, 1.0f }
        };
        hr = this->createVertexBuffer(sizeof(vertices), false, vertices);
    }
    if (SUCCEEDED(hr)) {
        //USHORT indices[] = { 
        //    0, 1, 2,   2, 1, 3,
        //    4, 0, 6,   6, 0, 2,
        //    7, 5, 6,   6, 5, 4,
        //    3, 1, 7,   7, 1, 5,
        //    4, 5, 0,   0, 5, 1,
        //    3, 7, 2,   2, 7, 6
        //};
        USHORT indices[] = { 
            0, 1,   1, 3,   3, 2,   2, 0,
            4, 5,   5, 7,   7, 6,   6, 4,

            0, 4,   1, 5,   3, 7,   2, 6
        };
        hr = this->createIndexBuffer(sizeof(indices), false, indices);
    }

    /* Create constant buffer for transformation etc. */
    if (SUCCEEDED(hr)) {
        hr = D3D11BoundingBox::createBuffer(this->cbConstants, 
            D3D11_BIND_CONSTANT_BUFFER, sizeof(Constants));
    }

    /* Create shaders. */
    if (SUCCEEDED(hr)) {
        hr = this->createVertexShaderAndInputLayoutFromBtf(factory, 
            "d3d11sphere::bbox-vertex", "Main", "vs_5_0",
            ilDesc, sizeof(ilDesc) / sizeof(*ilDesc));
    }
    if (SUCCEEDED(hr)) {
        hr = this->createPixelShaderFromBtf(factory, 
            "d3d11sphere::bbox-pixel", "Main", "ps_5_0");
    }

    /* Create rasteriser states. */
    if (SUCCEEDED(hr)) {
        ::ZeroMemory(&rasteriserDesc, sizeof(rasteriserDesc));
        rasteriserDesc.AntialiasedLineEnable = TRUE;
        rasteriserDesc.CullMode = D3D11_CULL_NONE;//D3D11_CULL_FRONT;
        rasteriserDesc.DepthClipEnable = TRUE;
        rasteriserDesc.FillMode = D3D11_FILL_SOLID;//D3D11_FILL_WIREFRAME;
        hr = this->device->CreateRasterizerState(&rasteriserDesc, 
            &this->rasteriserStateBack);
    }
    if (SUCCEEDED(hr)) {
        ::ZeroMemory(&rasteriserDesc, sizeof(rasteriserDesc));
        rasteriserDesc.AntialiasedLineEnable = TRUE;
        rasteriserDesc.CullMode = D3D11_CULL_NONE; //D3D11_CULL_BACK;
        rasteriserDesc.DepthClipEnable = TRUE;
        rasteriserDesc.FillMode = D3D11_FILL_SOLID;//D3D11_FILL_WIREFRAME;
        hr = this->device->CreateRasterizerState(&rasteriserDesc, 
            &this->rasteriserStateFront);
    }

    return hr;
}


/*
 * megamol::core::utility::D3D11BoundingBox::Update
 */
HRESULT megamol::core::utility::D3D11BoundingBox::Update(
        const XMMATRIX& viewMatrix,
        const XMMATRIX& projMatrix,
        const vislib::graphics::SceneSpaceCuboid& bbox,
        const vislib::graphics::ColourRGBAu8& colour) {
    VLAUTOSTACKTRACE;
    ASSERT(this->immediateContext != NULL);

    // http://msdn.microsoft.com/en-us/library/windows/desktop/ee418725(v=vs.85).aspx

    Constants *constants = NULL;
    HRESULT hr = S_OK;
    D3D11_MAPPED_SUBRESOURCE mappedRes;
    vislib::math::Dimension<vislib::graphics::SceneSpaceType, 3> size 
        = bbox.GetSize();
    vislib::math::Point<vislib::graphics::SceneSpaceType, 3> origin
        = bbox.GetOrigin();

    if (SUCCEEDED(hr)) {
        hr = this->immediateContext->Map(this->cbConstants, 0, 
            D3D11_MAP_WRITE_DISCARD, 0, &mappedRes);
    }
    if (SUCCEEDED(hr)) {
        constants = static_cast<Constants *>(mappedRes.pData);

        constants->Colour.x = colour.R() / 255.0f;
        constants->Colour.y = colour.G() / 255.0f;
        constants->Colour.z = colour.B() / 255.0f;
        constants->Colour.w = colour.A() / 255.0f;

        // http://social.msdn.microsoft.com/Forums/de-LU/wingameswithdirectx/thread/73696d3c-debe-4840-a062-925449f0a366
        constants->ProjMatrix = ::XMMatrixTranspose(projMatrix);

        // Note: The following transformations are not the same as the
        // ones we do for the data set. They are used to scale and translate
        // a unity bounding box in the origin to its correct size and
        // location.
        constants->ViewMatrix = ::XMMatrixScaling(size.Width(),
            size.Height(), size.Depth());
        constants->ViewMatrix *= ::XMMatrixTranslation(origin.X(),
            origin.Y(), origin.Z());
        constants->ViewMatrix *= viewMatrix;

        // http://social.msdn.microsoft.com/Forums/de-LU/wingameswithdirectx/thread/73696d3c-debe-4840-a062-925449f0a366
        constants->ViewMatrix = ::XMMatrixTranspose(constants->ViewMatrix);
        
        this->immediateContext->Unmap(this->cbConstants, 0);
    }

    return hr;
}
#endif /* MEGAMOLCORE_WITH_DIRECT3D11 */
