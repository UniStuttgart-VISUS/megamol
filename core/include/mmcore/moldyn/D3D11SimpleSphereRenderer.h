/*
 * D3D11SimpleSphereRenderer.h
 *
 * Copyright (C) 2012 by Visualisierungsinstitut der Universität Stuttgart.
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_D3D11SIMPLESPHERERENDERER_H_INCLUDED
#define MEGAMOLCORE_D3D11SIMPLESPHERERENDERER_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "mmcore/mmd3d.h"

#include "mmcore/moldyn/AbstractSphereRenderer.h"

#include "mmcore/utility/AbstractD3D11RenderObject.h"
#include "mmcore/utility/D3D11BoundingBox.h"

#include "vislib/Array.h"



namespace megamol {
namespace core {
namespace moldyn {

    class D3D11SimpleSphereRenderer : public AbstractSphereRenderer,
        protected utility::AbstractD3D11RenderObject {

    public:

        /**
         * Answer the name of this module.
         *
         * @return The name of this module.
         */
        static inline const char *ClassName(void) {
            return "D3D11SimpleSphereRenderer";
        }

        /**
         * Answer a human readable description of this module.
         *
         * @return A human readable description of this module.
         */
        static inline const char *Description(void) {
            return "Direct3D 11 renderer for sphere glyphs.";
        }

        /**
         * Answers whether this module is available on the current system.
         *
         * @return 'true' if the module is available, 'false' otherwise.
         */
        static bool IsAvailable(void);

        /**
         * Initialises a new instance.
         */
        D3D11SimpleSphereRenderer(void);

        /**
         * Cleans up the instance.
         */
        virtual ~D3D11SimpleSphereRenderer(void);

    protected:

        /**
         * Implementation of 'Create'.
         *
         * @return 'true' on success, 'false' otherwise.
         */
        virtual bool create(void);

        /**
         * The render callback.
         *
         * @param call The calling call.
         *
         * @return The return value of the function.
         */
        virtual bool Render(Call& call);

        /**
         * The Direct3D resource updatding callback.
         *
         * @param call The calling call.
         *
         * @return The return value of the function.
         */
        virtual bool Update(Call& call);

    private:

#ifdef MEGAMOLCORE_WITH_DIRECT3D11
        /** Constant buffer for data changing every frame. */
#pragma pack(push, 16)
        typedef struct Constants_t {
            DirectX::XMMATRIX ProjMatrix;
            DirectX::XMMATRIX ViewMatrix;
            DirectX::XMMATRIX ViewInvMatrix;
            DirectX::XMMATRIX ViewProjMatrix;
            DirectX::XMMATRIX ViewProjInvMatrix;
            DirectX::XMFLOAT4 Viewport;
            DirectX::XMFLOAT4 CamPos;
            DirectX::XMFLOAT4 CamDir;
            DirectX::XMFLOAT4 CamUp;
        } Constants;
#pragma pack(pop)

        /** Name of the BTF containing the HLSL shader source. */
        static const char *BTF_FILE_NAME;

        /** Size of a single vertex on the GPU. */
        static const size_t VERTEX_SIZE;

        size_t coalesceParticles(BYTE *buffer, const size_t cntBuffer,
            MultiParticleDataCall::Particles& particles);

        /**
         * Release all D3D resources.
         */
        void finaliseD3D(void);

        /**
         * Prepare all D3D resources on the device.
         *
         * @param device
         */
        HRESULT initialiseD3D(ID3D11Device *device);

        /** The resources for rendering the bounding box. */
        utility::D3D11BoundingBox bboxResources;

        /** Constant buffer for variables changing every frame. */
        ID3D11Buffer *cb;

        /** Resource view for 'texStereoParams'. */
        ID3D11ShaderResourceView *srvStereoParams;

        /** Sampler state for sampling 'texStereoParams'. */
        ID3D11SamplerState *ssStereoParams;

        /** Manages the NVIDIA stereo parameter texture. */
        nv::stereo::ParamTextureManagerD3D11 *stereoManager;

        /** Parameters with NVIDIA stereo parameters. */
        ID3D11Texture2D *texStereoParams;

        /** The texture for the colour transfer function. */
        ID3D11Texture1D *texXferFunc;

#endif /* MEGAMOLCORE_WITH_DIRECT3D11 */

        /** The render callee slot */
        CalleeSlot updateD3D;

    };

} /* end namespace moldyn */
} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_SIMPLESPHERERENDERER_H_INCLUDED */

