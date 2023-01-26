/**
 * MegaMol
 * Copyright (c) 2009, MegaMol Dev Team
 * All rights reserved.
 */

#pragma once

#include <array>
#include <cassert>
#include <memory>

#include "mmcore/Call.h"
#include "mmcore/factories/CallAutoDescription.h"

namespace megamol::core::view {

/**
 * Call for accessing a transfer function.
 */
class AbstractCallGetTransferFunction : public core::Call {
public:
    /** possible texture formats */
    enum TextureFormat { TEXTURE_FORMAT_RGB, TEXTURE_FORMAT_RGBA };

    /** Ctor. */
    AbstractCallGetTransferFunction()
            : Call()
            , texSize(1)
            , texData(NULL)
            , texFormat(TEXTURE_FORMAT_RGBA)
            , range({0.0f, 1.0f})
            , range_updated(false) {}

    /** Dtor. */
    ~AbstractCallGetTransferFunction() override {}

    /**
     * Answer whether the connected transferfunction is dirty
     *
     * @return dirty flag
     */
    inline bool IsDirty() {
        return this->usedTFVersion != this->availableTFVersion;
    }

    /**
     * Sets the transferfunction dirtiness
     */
    inline void ResetDirty() {
        this->usedTFVersion = availableTFVersion;
    }

    // SET --------------------------------------------------------------------
    /// !!! NOTE: In order to propagte changes from the call to the actual tf parameter,
    ///            the callback 'GetTexture' has to be called afterwards.

    /**
     * Sets the value range (domain) of this transfer function. Values
     * outside of min/max are to be clamped.
     */
    inline void SetRange(std::array<float, 2> range) {
        if (this->range != range) {
            this->range_updated = true;
            this->range = range;
        }
    }

    // GET --------------------------------------------------------------------

    /**
     * Copies a color from the transfer function
     *
     * @param index The n-th color to copy.
     * @param color A pointer to copy the color to.
     * @param colorSize The size of the color in bytes.
     */
    inline void CopyColor(size_t index, float* color, size_t colorSize) {
        assert(index >= 0 && index < this->texSize && "Invalid index");
        assert((colorSize == 3 * sizeof(float) || colorSize == 4 * sizeof(float)) && "Not a RGB(A) color");
        memcpy(color, &this->texData[index * 4], colorSize);
    }


    /**
     * Answer the texture data. This is always an RGBA float color
     * array, regardless the TextureFormat returned. If TextureFormat is
     * RGB the A values stored, are simply meaningless. Thus, this pointer
     * always points to TextureSize*4 floats.
     *
     * @return The texture data
     */
    inline float const* GetTextureData() const {
        return this->texData;
    }

    /**
     * Answer the size of the 1D texture in texel.
     *
     * @return The size of the texture
     */
    inline unsigned int TextureSize() const {
        return this->texSize;
    }

    /**
     * Answer the format of the texture.
     *
     * @return The format of the texture
     */
    inline TextureFormat TFTextureFormat() const {
        return this->texFormat;
    }

    /**
     * Answer the value range (domain) of this transfer function. Values
     * outside of min/max are to be clamped.
     *
     * @return The (min, max) pair.
     */
    inline std::array<float, 2> Range() const {
        return this->range;
    }


    ///// CALLEE Interface Functions //////////////////////////////////////////

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
    inline void SetTexture(
        unsigned int size, float const* tex, TextureFormat format, std::array<float, 2> range, uint32_t version) {
        this->texSize = size;
        this->texFormat = format;
        this->texData = tex;
        if (this->texSize == 0) {
            this->texSize = 1;
        }
        this->range = range;
        this->availableTFVersion = version;
    }

    /**
     * Check for updated range value and consume triggered update
     */
    bool ConsumeRangeUpdate() {
        bool consume = this->range_updated;
        this->range_updated = false;
        return consume;
    }

    /**
     * Check for updated range value
     */
    bool UpdateRange() {
        return this->range_updated;
    }

protected:
    /** The size of the texture in texel */
    unsigned int texSize;

    /** The texture data */
    float const* texData;

    /** The texture format */
    TextureFormat texFormat;

    /** The range the texture lies within */
    std::array<float, 2> range;

    /** Flag indicating changed range value. */
    bool range_updated;

    uint32_t availableTFVersion = 1;
    uint32_t usedTFVersion = 0;
};

} // namespace megamol::core::view
