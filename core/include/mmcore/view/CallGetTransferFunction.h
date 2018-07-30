/*
 * CallGetTransferFunction.h
 *
 * Copyright (C) 2009 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_CALLGETTRANSFERFUNCTION_H_INCLUDED
#define MEGAMOLCORE_CALLGETTRANSFERFUNCTION_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "mmcore/api/MegaMolCore.h"
#include "mmcore/Call.h"
#include "mmcore/factories/CallAutoDescription.h"
#include "vislib/graphics/gl/IncludeAllGL.h"


namespace megamol {
namespace core {
namespace view {


    /**
     * Base class of rendering graph calls
     */
    class MEGAMOLCORE_API CallGetTransferFunction : public Call {
    public:

        /** possible texture formats */
        enum TextureFormat {
            TEXTURE_FORMAT_RGB = GL_RGB,
            TEXTURE_FORMAT_RGBA = GL_RGBA
        };

        /**
         * Answer the name of the objects of this description.
         *
         * @return The name of the objects of this description.
         */
        static const char *ClassName(void) {
            return "CallGetTransferFunction";
        }

        /**
         * Gets a human readable description of the module.
         *
         * @return A human readable description of the module.
         */
        static const char *Description(void) {
            return "Call for a 1D transfer function";
        }

        /**
         * Answer the number of functions used for this call.
         *
         * @return The number of functions used for this call.
         */
        static unsigned int FunctionCount(void) {
            return 1;
        }

        /**
         * Answer the name of the function used for this call.
         *
         * @param idx The index of the function to return it's name.
         *
         * @return The name of the requested function.
         */
        static const char * FunctionName(unsigned int idx) {
            switch (idx) {
                case 0: return "GetTexture";
                default: return NULL;
            }
        }

        /** Ctor. */
        CallGetTransferFunction(void);

        /** Dtor. */
        virtual ~CallGetTransferFunction(void);

        /**
         * Answer the OpenGL texture object id of the transfer function 1D
         * texture.
         *
         * @return The OpenGL texture object id
         */
        inline unsigned int OpenGLTexture(void) const {
            return this->texID;
        }

        /**
         * Answer the size of the 1D texture in texel.
         *
         * @return The size of the texture
         */
        inline unsigned int TextureSize(void) const {
            return this->texSize;
        }

        /**
         * Answer the OpenGL format of the texture
         *
         * @return The OpenGL format of the texture
         */
        inline TextureFormat OpenGLTextureFormat(void) const {
            return this->texFormat;
        }

        /**
         * Answer the OpenGL texture data. This is always an RGBA float color
         * array, regardless the TextureFormat returned. If TextureFormat is
         * RGB the A values stored, are simply meaningless. Thus, this pointer
         * always points to TextureSize*4 floats.
         *
         * @return The OpenGL texture data
         */
        inline float const* GetTextureData(void) const {
            return this->texData;
        }

        /**
         * Sets the 1D texture information
         *
         * @param id The OpenGL texture object id
         * @param size The size of the texture
         * @param format The texture format
         */
        inline void SetTexture(unsigned int id, unsigned int size,
                TextureFormat format = TEXTURE_FORMAT_RGB) {
            this->texID = id;
            this->texSize = size;
            this->texFormat = format;
            this->texData = nullptr;
            if (this->texSize == 0) {
                this->texSize = 1;
            }
        }

        /**
         * Sets the 1D texture information
         *
         * @param id The OpenGL texture object id
         * @param size The size of the texture
         * @param tex The float RGBA texture data, i.e. size*4 floats holding
         *            the RGBA color data. The data is not copied. The caller
         *            is responsible for keeping the memory alive.
         * @param format The texture format
         */
        inline void SetTexture(unsigned int id, unsigned int size, float const* tex,
            TextureFormat format = TEXTURE_FORMAT_RGB) {
            this->texID = id;
            this->texSize = size;
            this->texFormat = format;
            this->texData = tex;
            if (this->texSize == 0) {
                this->texSize = 1;
            }
        }

        /**
         * Copies a color from the transfer function..
         *
         * @param index The n-th color to copy.
         * @param color A pointer to copy the color to.
         * @param colorSize The size of the color in bytes.
         */
        inline void CopyColor(size_t index, float* color, size_t colorSize) {
            assert(index > 0 && index < this->texSize && "Invalid index");
            assert(colorSize == 3 * sizeof(float) || colorSize == 4 * sizeof(float) && "Not a RGB(A) color");
            memcpy(color, &this->texData[index * 4], colorSize);
        }

    private:

        /** The OpenGL texture object id */
        unsigned int texID;

        /** The size of the texture in texel */
        unsigned int texSize;

        /** The texture data */
        float const* texData;

        /** The texture format */
        TextureFormat texFormat;

    };


    /** Description class typedef */
    typedef factories::CallAutoDescription<CallGetTransferFunction> CallGetTransferFunctionDescription;


} /* end namespace view */
} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_CALLGETTRANSFERFUNCTION_H_INCLUDED */
