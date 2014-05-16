/*
 * D3D11SimpleSphereRenderer.cpp
 *
 * Copyright (C) 2012 by Visualisierungsinstitut der Universität Stuttgart.
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "D3D11SimpleSphereRenderer.h"

#include "CoreInstance.h"

#include "utility/ShaderSourceFactory.h"
#include "view/CallRender3D.h"
#include "view/CallUpdateDirect3D.h"

#include "vislib/Camera.h"
#include "vislib/COMException.h"
#include "vislib/d3dutils.h"
#include "vislib/CameraOpenGL.h"
#include "vislib/Matrix.h"
#include "vislib/ShallowMatrix.h"
#include "vislib/Trace.h"


/*
 * megamol::core::moldyn::D3D11SimpleSphereRenderer::IsAvailable
 */
bool megamol::core::moldyn::D3D11SimpleSphereRenderer::IsAvailable(void) {
    VLAUTOSTACKTRACE;
#ifdef MEGAMOLCORE_WITH_DIRECT3D11
    return true;
#else /* MEGAMOLCORE_WITH_DIRECT3D11 */
    return false;
#endif /* MEGAMOLCORE_WITH_DIRECT3D11 */
}


/*
 * megamol::core::moldyn::D3D11SimpleSphereRenderer::D3D11SimpleSphereRenderer
 */
megamol::core::moldyn::D3D11SimpleSphereRenderer::D3D11SimpleSphereRenderer(
        void) : AbstractSimpleSphereRenderer(), AbstractD3D11RenderObject(),
#ifdef MEGAMOLCORE_WITH_DIRECT3D11
    cb(NULL), srvStereoParams(NULL), ssStereoParams(NULL), stereoManager(NULL), 
    texStereoParams(NULL), texXferFunc(NULL),
#endif /* MEGAMOLCORE_WITH_DIRECT3D11 */
    updateD3D("updated3d", "Allows the renderer using D3D11 resources of a view.")
    {
    VLAUTOSTACKTRACE;

    this->updateD3D.SetCallback(view::CallUpdateDirect3D::ClassName(),
        view::CallUpdateDirect3D::FunctionName(0),
        &D3D11SimpleSphereRenderer::Update);
    this->MakeSlotAvailable(&this->updateD3D);
}


/*
 * megamol::core::moldyn::D3D11SimpleSphereRenderer::~D3D11SimpleSphereRenderer
 */
megamol::core::moldyn::D3D11SimpleSphereRenderer::~D3D11SimpleSphereRenderer(
        void) {
    VLAUTOSTACKTRACE;
#ifdef MEGAMOLCORE_WITH_DIRECT3D11
    this->finaliseD3D();
#endif /* MEGAMOLCORE_WITH_DIRECT3D11 */
}


/*
 * megamol::core::moldyn::D3D11SimpleSphereRenderer::create
 */
bool megamol::core::moldyn::D3D11SimpleSphereRenderer::create(void) {
    VLAUTOSTACKTRACE;
#ifdef MEGAMOLCORE_WITH_DIRECT3D11
    bool retval = AbstractSimpleSphereRenderer::create();
    ::NvAPI_Initialize();
    //retval = retval && this->GetCoreInstance()->ShaderSourceFactory().LoadBTF(
    //    BTF_FILE_NAME);
    return retval;
#else /* MEGAMOLCORE_WITH_DIRECT3D11 */
    return false;
#endif /* MEGAMOLCORE_WITH_DIRECT3D11 */
}


/*
 * megamol::core::moldyn::D3D11SimpleSphereRenderer::Render
 */
bool megamol::core::moldyn::D3D11SimpleSphereRenderer::Render(Call& call) {
    VLAUTOSTACKTRACE;
#ifdef MEGAMOLCORE_WITH_DIRECT3D11
    ASSERT(this->device != NULL);
    ASSERT(this->immediateContext != NULL);
    ASSERT(this->stereoManager != NULL);

    vislib::graphics::SceneSpaceCuboid bbox;    // Data set bounding box.
    BYTE *buffer = NULL;                        // Cursor for filling VB.
    D3D11_BUFFER_DESC bufferDesc;               // VB description.
    size_t cntVertices = 0;                     // Total # of vertices to emit.
    UINT cntViewports = 1;                      // Number of viewports to read.
    XMVECTOR determinant;                       // Temporary result.
    XMVECTOR eyePosition;                       // Position of the camera.
    XMVECTOR focusPosition;                     // Look-at point of the camera.
    vislib::graphics::SceneSpaceViewFrustum frustum;    // The view frustum.
    HRESULT hr = S_OK;                          // D3D API results.
    D3D11_MAPPED_SUBRESOURCE mappedData;        // Mapped buffer data.
    Constants *constants = NULL;                // Pointer to mapped constants.
    size_t reqBytes = 0;                        // Required VB size for frame.
    XMVECTOR upDirection;                       // Camera up vector.
    XMFLOAT4 vector;                            // Temporary result.
    D3D11_VIEWPORT viewport;                    // Currently active viewport.

    UINT strides = D3D11SimpleSphereRenderer::VERTEX_SIZE;
    UINT offsets = 0;

    /* Update stereo stuff. */
    this->stereoManager->UpdateStereoTexture(this->device, 
        this->texStereoParams, false);

    /* Get the render call. */
    view::CallRender3D *cr = dynamic_cast<view::CallRender3D*>(&call);
    if (cr == NULL) {
        return false;
    }

    /* Get the data. */
    float scaling = 1.0f;
    MultiParticleDataCall *c2 = this->getData(
        static_cast<unsigned int>(cr->Time()), scaling);
    if (c2 == NULL) {
        return false;
    }

    bbox = cr->GetBoundingBoxes().WorldSpaceBBox();

    /* Determine required vertex buffer size. */
    for (unsigned int i = 0; i < c2->GetParticleListCount(); ++i) {
        ASSERT(c2->AccessParticles(i).GetCount() < SIZE_MAX);
        cntVertices += static_cast<size_t>(c2->AccessParticles(i).GetCount());
    }
    reqBytes = cntVertices * D3D11SimpleSphereRenderer::VERTEX_SIZE;

    /* 
     * Check whether we can re-use an existing VB. If not, release it in order
     * to allocate one that is large enough.
     */
    if (this->vertexBuffer != NULL) {
        this->vertexBuffer->GetDesc(&bufferDesc);
        ASSERT((bufferDesc.BindFlags & D3D11_BIND_VERTEX_BUFFER) != 0);
        if (bufferDesc.ByteWidth <= reqBytes) {
            SAFE_RELEASE(this->vertexBuffer);
        }
    }
    /* If 'vb' is not NULL here, it has a sufficient size. */

    if (this->vertexBuffer == NULL) {
        hr = this->createVertexBuffer(reqBytes, true);
    }
    if (FAILED(hr)) {
        return false;
    }

    /* Update vertex buffer. */
    hr = this->immediateContext->Map(this->vertexBuffer, 0, 
        D3D11_MAP_WRITE_DISCARD, 0, &mappedData);
    if (FAILED(hr)) {
        vislib::sys::Log::DefaultLog.WriteError(_T("Mapping vertex buffer ")
            _T("failed with error code %u."), hr);
        return false;
    }

    buffer = static_cast<BYTE *>(mappedData.pData);
    for (unsigned int i = 0; i < c2->GetParticleListCount(); ++i) {
        buffer += this->coalesceParticles(buffer, 
            reqBytes - (buffer - static_cast<BYTE *>(mappedData.pData)),
            c2->AccessParticles(i));
    }

    this->immediateContext->Unmap(this->vertexBuffer, 0);

    /* Update constant buffers. */
    // TODO: Camera handling is nasty in the best case...
    vislib::graphics::gl::CameraOpenGL cam(cr->GetCameraParameters());

    /* Update the constant buffer with the new view parameters. */
    if (SUCCEEDED(hr)) {
        hr = this->immediateContext->Map(this->cb, 0, 
            D3D11_MAP_WRITE_DISCARD, 0, &mappedData);
    }
    if (SUCCEEDED(hr)) {
        constants = static_cast<Constants *>(mappedData.pData);

        //cam.ProjectionMatrix(&constants->ProjMatrix._11);
        cam.CalcViewFrustum(frustum);
        constants->ProjMatrix = ::XMMatrixPerspectiveOffCenterRH(
            frustum.GetLeftDistance(),
            frustum.GetRightDistance(),
            frustum.GetBottomDistance(),
            frustum.GetTopDistance(),
            frustum.GetNearDistance(),
            frustum.GetFarDistance());

        this->immediateContext->RSGetViewports(&cntViewports, &viewport);
        constants->Viewport.x = viewport.TopLeftX;
        constants->Viewport.y = viewport.TopLeftY;
        constants->Viewport.z = viewport.Width;
        constants->Viewport.w = viewport.Height;

        constants->CamPos.x = cr->GetCameraParameters()->EyePosition().X();
        constants->CamPos.y = cr->GetCameraParameters()->EyePosition().Y();
        constants->CamPos.z = cr->GetCameraParameters()->EyePosition().Z();
        constants->CamPos.w = 1.0f;
        eyePosition = ::XMLoadFloat4(&constants->CamPos);

        vector.x = cr->GetCameraParameters()->LookAt().X();
        vector.y = cr->GetCameraParameters()->LookAt().Y();
        vector.z = cr->GetCameraParameters()->LookAt().Z();
        vector.w = 0.0f;
        focusPosition = ::XMLoadFloat4(&vector);

        constants->CamDir.x = cr->GetCameraParameters()->EyeDirection().X();
        constants->CamDir.y = cr->GetCameraParameters()->EyeDirection().Y();
        constants->CamDir.z = cr->GetCameraParameters()->EyeDirection().Z();
        constants->CamDir.w = 0.0f;

        constants->CamUp.x = cr->GetCameraParameters()->EyeUpVector().X();
        constants->CamUp.y = cr->GetCameraParameters()->EyeUpVector().Y();
        constants->CamUp.z = cr->GetCameraParameters()->EyeUpVector().Z();
        constants->CamUp.w = 0.0f;
        upDirection = ::XMLoadFloat4(&constants->CamUp);

        constants->ViewMatrix = ::XMMatrixLookAtRH(
            eyePosition, focusPosition, upDirection);

        // Now we have everything to transform the bounding box. Do so before
        // applying the scaling (the bbox will scale correctly by itself).
        this->bboxResources.Update(constants->ViewMatrix, 
            constants->ProjMatrix, 
            bbox,
            vislib::graphics::ColourRGBAu8(128, 128, 128, 255));

        constants->ViewMatrix = ::XMMatrixScaling(scaling, scaling, 
            scaling) * constants->ViewMatrix;

        constants->ViewProjMatrix = constants->ViewMatrix
            * constants->ProjMatrix;

        //cam.ViewMatrix(&constants->WorldViewMatrix._11);
        //vislib::math::ShallowMatrix<float, 4, vislib::math::COLUMN_MAJOR> v(
        //    &constants->WorldViewMatrix._11);
        //VLTRACE(1000, "MV %f %f %f %f\n%f %f %f %f\n%f %f %f %f\n%f %f %f %f\n\n", 
        //    v(0, 0), v(0, 1), v(0, 2), v(0, 3), 
        //    v(1, 0), v(1, 1), v(1, 2), v(1, 3), 
        //    v(2, 0), v(2, 1), v(2, 2), v(2, 3), 
        //    v(3, 0), v(3, 1), v(3, 2), v(3, 3));

        determinant = ::XMMatrixDeterminant(constants->ViewMatrix);
        constants->ViewInvMatrix = ::XMMatrixInverse(&determinant,
            constants->ViewMatrix);

        //constants->WorldViewInvMatrix = constants->WorldViewMatrix;
        //vislib::math::ShallowMatrix<float, 4, vislib::math::COLUMN_MAJOR> vi(
        //    &constants->WorldViewInvMatrix._11);
        //vi.Invert();
        //VLTRACE(1000, "MVI %f %f %f %f\n%f %f %f %f\n%f %f %f %f\n%f %f %f %f\n\n", 
        //    vi(0, 0), vi(0, 1), vi(0, 2), vi(0, 3), 
        //    vi(1, 0), vi(1, 1), vi(1, 2), vi(1, 3), 
        //    vi(2, 0), vi(2, 1), vi(2, 2), vi(2, 3), 
        //    vi(3, 0), vi(3, 1), vi(3, 2), vi(3, 3));

        constants->ViewProjInvMatrix = constants->ViewProjMatrix;
        determinant = ::XMMatrixDeterminant(constants->ViewProjInvMatrix);
        constants->ViewProjInvMatrix = ::XMMatrixInverse(&determinant,
            constants->ViewProjInvMatrix);

        //vislib::math::ShallowMatrix<float, 4, vislib::math::COLUMN_MAJOR> vpi(
        //    &constants->WorldViewProjInvMatrix._11);
        //vpi = p * v;
        ////vpi = v * p;
        //VLTRACE(1000, "MVP %f %f %f %f\n%f %f %f %f\n%f %f %f %f\n%f %f %f %f\n\n", 
        //    vpi(0, 0), vpi(0, 1), vpi(0, 2), vpi(0, 3), 
        //    vpi(1, 0), vpi(1, 1), vpi(1, 2), vpi(1, 3), 
        //    vpi(2, 0), vpi(2, 1), vpi(2, 2), vpi(2, 3), 
        //    vpi(3, 0), vpi(3, 1), vpi(3, 2), vpi(3, 3));
        //vpi.Invert();

        // http://social.msdn.microsoft.com/Forums/de-LU/wingameswithdirectx/thread/73696d3c-debe-4840-a062-925449f0a366
        constants->ProjMatrix = ::XMMatrixTranspose(
            constants->ProjMatrix);
        constants->ViewMatrix = ::XMMatrixTranspose(
            constants->ViewMatrix);
        constants->ViewInvMatrix = ::XMMatrixTranspose(
            constants->ViewInvMatrix);
        constants->ViewProjMatrix = ::XMMatrixTranspose(
            constants->ViewProjMatrix);
        constants->ViewProjInvMatrix = ::XMMatrixTranspose(
            constants->ViewProjInvMatrix);

        this->immediateContext->Unmap(this->cb, 0);
    }

    if (SUCCEEDED(hr)) {   
        this->bboxResources.Draw(true);
    }

    /* Render */
    if (SUCCEEDED(hr)) {
        this->immediateContext->IASetInputLayout(this->inputLayout);
        this->immediateContext->IASetPrimitiveTopology(
            D3D11_PRIMITIVE_TOPOLOGY_POINTLIST);
        this->immediateContext->IASetVertexBuffers(0, 1, &this->vertexBuffer,
            &strides, &offsets);

        this->immediateContext->VSSetShader(this->vertexShader, NULL, 0);
        this->immediateContext->VSSetConstantBuffers(0, 1, &this->cb);

        this->immediateContext->GSSetShader(this->geometryShader, NULL, 0);
        this->immediateContext->GSSetSamplers(8, 1, &this->ssStereoParams);
        this->immediateContext->GSSetShaderResources(8, 1, 
            &this->srvStereoParams);
        this->immediateContext->GSSetConstantBuffers(0, 1, &this->cb);

        this->immediateContext->PSSetShader(this->pixelShader, NULL, 0);
        this->immediateContext->PSSetSamplers(8, 1, &this->ssStereoParams);
        this->immediateContext->PSSetShaderResources(8, 1, 
            &this->srvStereoParams);
        this->immediateContext->PSSetConstantBuffers(0, 1, &this->cb);

        this->immediateContext->RSSetState(NULL);

        this->immediateContext->Draw(cntVertices, 0);
    
        this->bboxResources.Draw(false);
    }

    c2->Unlock();

    return true;
#else  /* MEGAMOLCORE_WITH_DIRECT3D11 */
    return false;
#endif /* MEGAMOLCORE_WITH_DIRECT3D11 */
}


/*
 * megamol::core::moldyn::D3D11SimpleSphereRenderer::Update
 */
bool megamol::core::moldyn::D3D11SimpleSphereRenderer::Update(Call& call) {
    VLAUTOSTACKTRACE;
#ifdef MEGAMOLCORE_WITH_DIRECT3D11
    using view::CallUpdateDirect3D;
    try {
        CallUpdateDirect3D& cu = dynamic_cast<CallUpdateDirect3D&>(call);
        this->finaliseD3D();
        this->initialiseD3D(cu.PeekDevice());
    } catch (vislib::Exception e) {
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
            "Preparing Direct3D resources for rendering simple spheres failed: "
            "%s\n", e.GetMsgA());
        return false;
    }
#endif /* MEGAMOLCORE_WITH_DIRECT3D11 */
    return true;
}


///*
// * megamol::core::moldyn::D3D11SimpleSphereRenderer::layoutVertices
// */
//void megamol::core::moldyn::D3D11SimpleSphereRenderer::layoutVertices(
//        BYTE *data, const SIZE_T size, 
//        MultiParticleDataCall::Particles& particles) {
//    VLAUTOSTACKTRACE;
//
//    ASSERT(data != NULL);
//    ASSERT(size >= particles.GetCount() * VERTEX_SIZE);
//
//    float minC = 0.0f, maxC = 0.0f;
//    unsigned int colTabSize = 0;
//
//        switch (parts.GetVertexDataType()) {
//            case MultiParticleDataCall::Particles::VERTDATA_NONE:
//                ::ZeroMemory(b, VERTEX_SIZE * parts.GetCount());
//                break;
//            case MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZ:
//    //            glEnableClientState(GL_VERTEX_ARRAY);
//    //            glUniform4fARB(this->sphereShader.ParameterLocation("inConsts1"), parts.GetGlobalRadius(), minC, maxC, float(colTabSize));
//    //            glVertexPointer(3, GL_FLOAT, parts.GetVertexDataStride(), parts.GetVertexData());
//    //            break;
//    //        case MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZR:
//    //            glEnableClientState(GL_VERTEX_ARRAY);
//    //            glUniform4fARB(this->sphereShader.ParameterLocation("inConsts1"), -1.0f, minC, maxC, float(colTabSize));
//    //            glVertexPointer(4, GL_FLOAT, parts.GetVertexDataStride(), parts.GetVertexData());
//    //            break;
//    //        default:
//    //            continue;
//       }
//
//    //    // colour

//    }
//
//}


#ifdef MEGAMOLCORE_WITH_DIRECT3D11
/*
 * megamol::core::moldyn::D3D11SimpleSphereRenderer::BTF_FILE_NAME
 */
const char *megamol::core::moldyn::D3D11SimpleSphereRenderer::BTF_FILE_NAME
    = "d3d11sphere";


/*
 * megamol::core::moldyn::D3D11SimpleSphereRenderer::VERTEX_SIZE
 */
const size_t megamol::core::moldyn::D3D11SimpleSphereRenderer::VERTEX_SIZE
    = sizeof(XMFLOAT4) + sizeof(XMFLOAT4);


/*
 * megamol::core::moldyn::D3D11SimpleSphereRenderer::coalesceParticles
 */
size_t megamol::core::moldyn::D3D11SimpleSphereRenderer::coalesceParticles(
        BYTE *buffer, const size_t cntBuffer,
        MultiParticleDataCall::Particles& particles) {
    VLAUTOSTACKTRACE;
    ASSERT(buffer != NULL);
    ASSERT(particles.GetCount() < SIZE_MAX);

    BYTE *b = NULL;             // Cursor through 'buffer'.
    size_t cntParticles = static_cast<size_t>(particles.GetCount());
    float grey[] = { 0.5f, 0.5f, 0.5f, 1.0f };
    const BYTE *p = NULL;       // Cursor through particle data.
    size_t particleSize = 0;
    size_t particleStride = 0;
    size_t reqBytes = cntParticles * D3D11SimpleSphereRenderer::VERTEX_SIZE;
    
    if (reqBytes > cntBuffer) {
        cntParticles = reqBytes / D3D11SimpleSphereRenderer::VERTEX_SIZE;
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_WARN,
            "Coalescing %u particles into a buffer of %u Bytes is not "
            "possible. At least %u Bytes will be required for all particles. "
            "The data set will be truncated to %u particles.\n", 
            static_cast<size_t>(particles.GetCount()), cntBuffer, reqBytes,
            cntParticles);
        // Adjust # of required bytes, as this is the return value:
        reqBytes = cntParticles * D3D11SimpleSphereRenderer::VERTEX_SIZE;
    }

    switch (particles.GetVertexDataType()) {
        case MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZ:
            b = buffer;
            p = static_cast<const BYTE *>(particles.GetVertexData());
            particleSize = 3 * sizeof(float);
            particleStride = particles.GetVertexDataStride();
            for (size_t i = 0; i < cntParticles; ++i) {
                ::memcpy(b, p, particleSize);
                *reinterpret_cast<float *>(b + particleSize) 
                    = particles.GetGlobalRadius();
                b += D3D11SimpleSphereRenderer::VERTEX_SIZE;
                p += particleStride;
            }
            break;

        case MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZR:
            b = buffer;
            p = static_cast<const BYTE *>(particles.GetVertexData());
            particleSize = 4 * sizeof(float);
            particleStride = particles.GetVertexDataStride();
            for (size_t i = 0; i < cntParticles; ++i) {
                ::memcpy(b, p, particleSize);
                b += D3D11SimpleSphereRenderer::VERTEX_SIZE;
                p += particleStride;
            }
            break;

        case MultiParticleDataCall::Particles::VERTDATA_SHORT_XYZ:
        case MultiParticleDataCall::Particles::VERTDATA_NONE:
        default:
            ASSERT(false);
            break;
    }

    switch (particles.GetColourDataType()) {
        case MultiParticleDataCall::Particles::COLDATA_NONE: {
            b = buffer + sizeof(XMFLOAT4);
            float colour[4];
            colour[0] = particles.GetGlobalColour()[0] / 255.0f;
            colour[1] = particles.GetGlobalColour()[1] / 255.0f;
            colour[2] = particles.GetGlobalColour()[2] / 255.0f;
            colour[3] = particles.GetGlobalColour()[3] / 255.0f;
            for (size_t i = 0; i < cntParticles; ++i) {
                ::memcpy(b, colour, sizeof(colour));
                b += D3D11SimpleSphereRenderer::VERTEX_SIZE;
            }
            } break;

        //case MultiParticleDataCall::Particles::COLDATA_UINT8_RGB:
        //    glEnableClientState(GL_COLOR_ARRAY);
        //    glColorPointer(3, GL_UNSIGNED_BYTE, parts.GetColourDataStride(), parts.GetColourData());
        //    break;
        //case MultiParticleDataCall::Particles::COLDATA_UINT8_RGBA:
        //    glEnableClientState(GL_COLOR_ARRAY);
        //    glColorPointer(4, GL_UNSIGNED_BYTE, parts.GetColourDataStride(), parts.GetColourData());
        //    break;
        case MultiParticleDataCall::Particles::COLDATA_FLOAT_RGB:
            b = buffer + sizeof(XMFLOAT4);
            p = static_cast<const BYTE *>(particles.GetColourData());
            particleStride = particles.GetColourDataStride();
            particleSize = 3 * sizeof(float);
            for (size_t i = 0; i < cntParticles; ++i) {
                ::memcpy(b, p, particleSize);
                *reinterpret_cast<float *>(b + particleSize) = 1.0f;
                b += D3D11SimpleSphereRenderer::VERTEX_SIZE;
                p += particleStride;
            }
            break;
        //case MultiParticleDataCall::Particles::COLDATA_FLOAT_RGBA:
        //    glEnableClientState(GL_COLOR_ARRAY);
        //    glColorPointer(4, GL_FLOAT, parts.GetColourDataStride(), parts.GetColourData());
        //    break;
        //case MultiParticleDataCall::Particles::COLDATA_FLOAT_I: {
        //    glEnableVertexAttribArrayARB(cial);
        //    glVertexAttribPointerARB(cial, 1, GL_FLOAT, GL_FALSE, parts.GetColourDataStride(), parts.GetColourData());

        //    glEnable(GL_TEXTURE_1D);

        //    view::CallGetTransferFunction *cgtf = this->getTFSlot.CallAs<view::CallGetTransferFunction>();
        //    if ((cgtf != NULL) && ((*cgtf)())) {
        //        glBindTexture(GL_TEXTURE_1D, cgtf->OpenGLTexture());
        //        colTabSize = cgtf->TextureSize();
        //    } else {
        //        glBindTexture(GL_TEXTURE_1D, this->greyTF);
        //        colTabSize = 2;
        //    }

        //    glUniform1iARB(this->sphereShader.ParameterLocation("colTab"), 0);
        //    minC = parts.GetMinColourIndexValue();
        //    maxC = parts.GetMaxColourIndexValue();
        //    glColor3ub(127, 127, 127);
        //} break;
        default:
            b = buffer + sizeof(XMFLOAT4);
            for (size_t i = 0; i < cntParticles; ++i) {
                ::memcpy(b, &grey, sizeof(grey));
                b += D3D11SimpleSphereRenderer::VERTEX_SIZE;
            }
            break;
    }

    return reqBytes;
}



/*
 * megamol::core::moldyn::D3D11SimpleSphereRenderer::finaliseD3D
 */
void megamol::core::moldyn::D3D11SimpleSphereRenderer::finaliseD3D(void) {
    VLAUTOSTACKTRACE;
    VLTRACE(VISLIB_TRCELVL_INFO, "Releasing D3D resources in "
        "D3D11SimpleSphereRenderer...\n");
    this->bboxResources.Finalise();
    SAFE_RELEASE(this->cb);
    SAFE_RELEASE(this->inputLayout);
    SAFE_RELEASE(this->srvStereoParams);
    SAFE_RELEASE(this->ssStereoParams);
    SAFE_DELETE(this->stereoManager);
    SAFE_RELEASE(this->texStereoParams);
    SAFE_RELEASE(this->texXferFunc);
    this->Release();
}


/*
 * megamol::core::moldyn::D3D11SimpleSphereRenderer::initialiseD3D
 */
HRESULT megamol::core::moldyn::D3D11SimpleSphereRenderer::initialiseD3D(
        ID3D11Device *device) {
    VLAUTOSTACKTRACE;
    using nv::stereo::ParamTextureManagerD3D11;
    using utility::ShaderSourceFactory;
    using vislib::sys::Log;

    ASSERT(device != NULL);
    ASSERT(this->device == NULL);

    HRESULT hr = S_OK;                      // API call results.
    D3D11_INPUT_ELEMENT_DESC ilDesc[2] = {  // The input layout.
        { "POSITION", 0, DXGI_FORMAT_R32G32B32A32_FLOAT, 0, 0, D3D11_INPUT_PER_VERTEX_DATA, 0 },
        { "COLOR", 0, DXGI_FORMAT_R32G32B32A32_FLOAT, 0, 16, D3D11_INPUT_PER_VERTEX_DATA, 0 },
    };
    ShaderSourceFactory shaderFactory       // Factory for shaders from BTFs.
        = this->GetCoreInstance()->ShaderSourceFactory();

    hr = AbstractD3D11RenderObject::Initialise(device);
    ASSERT(this->device != NULL);
    ASSERT(this->immediateContext != NULL);

    
    //D3D11_BLEND_DESC blendDesc;
    //D3D11_BUFFER_DESC bufferDesc;
    //D3D11_DEPTH_STENCIL_DESC dsDesc;
    //HRESULT hr = S_OK;
    D3D11_SHADER_RESOURCE_VIEW_DESC srvDesc;
    D3D11_SAMPLER_DESC ssDesc;
    D3D11_TEXTURE2D_DESC texDesc;

    if (SUCCEEDED(hr)) {
        hr = this->bboxResources.Initialise(this->device, shaderFactory);
    }

    /* Compile vertex shader and create input layout. */
    if (SUCCEEDED(hr)) {
        hr = this->createVertexShaderAndInputLayoutFromBtf(shaderFactory,
            "d3d11sphere::sphere-vertex", "Main", "vs_5_0",
            ilDesc, sizeof(ilDesc) / sizeof(*ilDesc));
    }

    /* Compile geometry shader. */
    if (SUCCEEDED(hr)) {
        hr = this->createGeometryShaderFromBtf(shaderFactory,
            "d3d11sphere::sphere-geometry", "Main", "gs_5_0");
    }

    /* Compile pixel shader. */
    if (SUCCEEDED(hr)) {
        hr = this->createPixelShaderFromBtf(shaderFactory, 
            "d3d11sphere::sphere-pixel", "Main", "ps_4_0");
    }

    /* Allocate constant buffers. */
    if (SUCCEEDED(hr)) {
        hr = this->createBuffer(this->cb, D3D11_BIND_CONSTANT_BUFFER, 
            sizeof(Constants), true);
        if (FAILED(hr)) {
            vislib::sys::Log::DefaultLog.WriteError(_T("Allocating constant ")
                _T("buffer for failed with error code %u"), hr);
        }
    }

    /* Allocate blend and depth/stencil state objects. */
    //::ZeroMemory(&dsDesc, sizeof(dsDesc));
    //dsDesc.DepthEnable = TRUE;
    ////dsDesc.DepthFunc = D3D11_COMPARISON_GREATER;
    //hr = this->device->CreateDepthStencilState(&dsDesc, &this->dsState);
    //if (FAILED(hr)) {
    //    throw vislib::sys::COMException(hr, __FILE__, __LINE__);
    //}
    //this->immediateContext->OMSetDepthStencilState(this->dsState, 0);

    //::ZeroMemory(&blendDesc, sizeof(blendDesc));
    //blendDesc.


    /* Allocate resources for rendering the bounding box. */

    /* Allocate NV stereo resources. */
    if (SUCCEEDED(hr)) {
        ::ZeroMemory(&texDesc, sizeof(texDesc));
        texDesc.Width = ParamTextureManagerD3D11::Parms::StereoTexWidth;
        texDesc.Height = ParamTextureManagerD3D11::Parms::StereoTexHeight;
        texDesc.MipLevels = 1;
        texDesc.ArraySize = 1;
        texDesc.Format = ParamTextureManagerD3D11::Parms::StereoTexFormat;
        texDesc.SampleDesc.Count = 1;
        texDesc.SampleDesc.Quality = 0;
        texDesc.Usage = D3D11_USAGE_DYNAMIC;
        texDesc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
        texDesc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
        texDesc.MiscFlags = 0;

        hr = this->device->CreateTexture2D(&texDesc, NULL, 
            &this->texStereoParams);
        if (FAILED(hr)) {
            Log::DefaultLog.WriteError(_T("Creating texture for stereo ")
                _T("parameter with error code %d."), hr);
        }
    }

    if (SUCCEEDED(hr)) {
        ::ZeroMemory(&srvDesc, sizeof(srvDesc));
        srvDesc.Format = texDesc.Format;
        srvDesc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
        srvDesc.Texture2D.MipLevels = 1;
        srvDesc.Texture2D.MostDetailedMip = 0;
        srvDesc.Texture2DArray.MostDetailedMip = 0;
        srvDesc.Texture2DArray.MipLevels = 1;
        srvDesc.Texture2DArray.FirstArraySlice = 0;
        srvDesc.Texture2DArray.ArraySize = texDesc.ArraySize;

        hr = this->device->CreateShaderResourceView(this->texStereoParams, 
            &srvDesc, &this->srvStereoParams);
        if (FAILED(hr)) {
            Log::DefaultLog.WriteError(_T("Creating shader resource view for ")
                _T("stereo parameter texture failed with error code %d."), hr);
        }
    }

    if (SUCCEEDED(hr)) {
        hr = this->createSamplerState(this->ssStereoParams,
            D3D11_FILTER_MIN_MAG_MIP_POINT);
    }

    if (SUCCEEDED(hr)) {
        this->stereoManager = new nv::stereo::ParamTextureManagerD3D11();
        this->stereoManager->Init(this->device);
    }

    return hr;
}

#endif /* MEGAMOLCORE_WITH_DIRECT3D11 */
