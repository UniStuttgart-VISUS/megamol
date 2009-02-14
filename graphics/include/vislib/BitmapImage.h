/*
 * BitmapImage.h
 *
 * Copyright (C) 2009 by Sebastian Grottel.
 * (Copyright (C) 2009 by Visualisierungsinstitut Universitaet Stuttgart.)
 * Alle Rechte vorbehalten.
 */

#ifndef VISLIB_BITMAPIMAGE_H_INCLUDED
#define VISLIB_BITMAPIMAGE_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


namespace vislib {
namespace graphics {


    /**
     * Class storing bitmap image data. The data is organized in channels.
     */
    class BitmapImage {
    public:

        /** possible channel labels */
        enum ChannelLabel {
            CHANNEL_UNDEF = 0, //< undefined channel
            CHANNEL_RED,
            CHANNEL_GREEN,
            CHANNEL_BLUE,
            CHANNEL_GRAY,
            CHANNEL_ALPHA
        };

        /** possible channel types */
        enum ChannelType {
            CHANNELTYPE_BYTE, //< use 1 byte per channel and pixel
            CHANNELTYPE_WORD, //< use 2 bytes per channel and pixel
            CHANNELTYPE_FLOAT //< use 4 bytes per channel and pixel
        };

        /**
         * Ctor. Generates a new bitmap.
         *
         * If 'data' is not NULL, the memory it points to is interpreted
         * according to the other parameters as array of scanlines where the
         * data of the channels is stored interleaved.
         *
         * @param width The width in pixels
         * @param height The height in pixels
         * @param channels The number of channels per pixel (must be larger
         *                 than zero)
         * @param type The type of the channels in this bitmap.
         * @param data The data to initialize the image with. If this is NULL,
         *             the image will be initialized with Zero
         *             (black/transparent) in all channels.
         */
        BitmapImage(unsigned int width, unsigned int height,
            unsigned int channels, ChannelType type, const void *data);

        /**
         * Copy ctor. Creates a deep copy of the source image.
         *
         * @param src The source image to create the deep copy from.
         */
        explicit BitmapImage(const BitmapImage& src);

        /** Dtor. */
        ~BitmapImage(void);

        /**
         * Answers the bytes per pixel of this image.
         *
         * @return The bytes per pixel of this image.
         */
        inline unsigned int BytesPerPixel(void) const {
            unsigned int base = 1;
            switch (this->chanType) {
                case CHANNELTYPE_WORD: base = 2; break;
                case CHANNELTYPE_FLOAT: base = 4; break;
#ifndef _WIN32
                default: break;
#endif /* !_WIN32 */
            }
            return base * this->numChans;
        }

        /**
         *  Creates a deep copy of the source image.
         *
         * @param src The source image to create the deep copy from.
         */
        void CopyFrom(const BitmapImage& src);

        /**
         * Generates a new bitmap. This overwrites all data previously stored
         * in the bitmap.
         *
         * If 'data' is not NULL, the memory it points to is interpreted
         * according to the other parameters as array of scanlines where the
         * data of the channels is stored interleaved.
         *
         * @param width The width in pixels
         * @param height The height in pixels
         * @param channels The number of channels per pixel (must be larger
         *                 than zero)
         * @param type The type of the channels in this bitmap.
         * @param data The data to initialize the image with. If this is NULL,
         *             the image will be initialized with Zero
         *             (black/transparent) in all channels.
         */
        void CreateImage(unsigned int width, unsigned int height,
            unsigned int channels, ChannelType type, const void *data);

        /**
         * Answer the number of channels
         *
         * @return The number of channels
         */
        inline unsigned int GetChannelCount(void) const {
            return this->numChans;
        }

        /**
         * Answer a channel label for one channel. If the zero-base channel
         * index 'channel' is out of range 'CHANNEL_UNDEF' is returned.
         *
         * @param channel The zero-base index of the channel to return the
         *                label of.
         *
         * @return The channel label of the requested channel.
         */
        inline ChannelLabel GetChannelLabel(unsigned int channel) const {
            return (channel >= this->numChans)
                ? CHANNEL_UNDEF : this->labels[channel];
        }

        /**
         * Answer the channel type of the image.
         *
         * @return The channel type of the image.
         */
        inline ChannelType GetChannelType(void) const {
            return this->chanType;
        }

        /**
         * Answer the height of the image.
         *
         * @return The height of the image.
         */
        inline unsigned int Height(void) const {
            return this->height;
        }

        /**
         * Peeks at the raw image data.
         *
         * @return A pointer to the raw image data stored.
         */
        inline void * PeekData(void) {
            return static_cast<void*>(this->data);
        }

        /**
         * Peeks at the raw image data.
         *
         * @return A pointer to the raw image data stored.
         */
        inline const void * PeekData(void) const {
            return static_cast<const void*>(this->data);
        }

        /**
         * Peeks at the raw image data casted to the type of the template
         * parameter (e.g. WORD or float).
         *
         * @return A pointer to the raw image data stored.
         */
        template<class T>
        inline T * PeekDataAs(void) {
            return reinterpret_cast<T*>(this->data);
        }

        /**
         * Peeks at the raw image data casted to the type of the template
         * parameter (e.g. WORD or float).
         *
         * @return A pointer to the raw image data stored.
         */
        template<class T>
        inline const T * PeekDataAs(void) const {
            return reinterpret_cast<const T*>(this->data);
        }

        /**
         * Sets the label of on channel. If the zero-based channel index
         * 'channel' is out of range, nothing will happen.
         *
         * @param channel The zero-based index of the channel to label.
         * @param label The label to be set for this channel.
         */
        inline void SetChannelLabel(unsigned int channel, ChannelLabel label) {
            if (channel < this->numChans) {
                this->labels[channel] = label;
            }
        }

        /**
         * Answer the width of the image.
         *
         * @return The width of the image.
         */
        inline unsigned int Width(void) const {
            return this->width;
        }

    private:

        /** The raw image data */
        char *data;

        /** The type of the channel data of all channels */
        ChannelType chanType;

        /** The height of the image in pixels */
        unsigned int height;

        /** The labels of the channels */
        ChannelLabel *labels;

        /** The number of channels in the image */
        unsigned int numChans;

        /** The width of the image in pixels */
        unsigned int width;

    };
    
} /* end namespace graphics */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_BITMAPIMAGE_H_INCLUDED */

