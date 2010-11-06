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

#include "vislib/types.h"


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
            CHANNEL_ALPHA,
            CHANNEL_CYAN,
            CHANNEL_MAGENTA,
            CHANNEL_YELLOW,
            CHANNEL_BLACK
        };

        /** possible channel types */
        enum ChannelType {
            CHANNELTYPE_BYTE, //< use 1 byte per channel and pixel
            CHANNELTYPE_WORD, //< use 2 bytes per channel and pixel
            CHANNELTYPE_FLOAT //< use 4 bytes per channel and pixel
        };

        /**
         * Base class for image extensions, like animation or compositing
         * layers
         */
        class Extension {
        public:

            /**
             * Ctor
             *
             * @return owner The owning image
             */
            Extension(BitmapImage& owner);

            /** Dtor */
            virtual ~Extension(void);

            /**
             * Clones this extension object when the hosting image is copied
             *
             * @param newowner The image to become the owner for the clone
             *
             * @return A clone (deep copy) of this object
             */
            virtual Extension * Clone(BitmapImage& newowner) = 0;

            /**
             * Answer the owner of the extension
             *
             * @return The owning image of the extension
             */
            inline BitmapImage& Owner(void) {
                return this->owner;
            }

            /**
             * Answer the owner of the extension
             *
             * @return The owning image of the extension
             */
            inline const BitmapImage& Owner(void) const {
                return this->owner;
            }

        private:

            /** The hosting image */
            BitmapImage& owner;

        };

        /** A BitmapImage template with Gray byte channels */
        static const BitmapImage TemplateByteGray;

        /** A BitmapImage template with RGB byte channels */
        static const BitmapImage TemplateByteGrayAlpha;

        /** A BitmapImage template with RGB byte channels */
        static const BitmapImage TemplateByteRGB;

        /** A BitmapImage template with RGBA byte channels */
        static const BitmapImage TemplateByteRGBA;

        /**
         * Ctor. Creates an empty bitmap (channel count, width, and height
         * zero).
         */
        BitmapImage(void);

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
            unsigned int channels, ChannelType type, const void *data = NULL);

        /**
         * Ctor. Generates a new bitmap.
         *
         * If 'data' is not NULL, the memory it points to is interpreted
         * according to the other parameters as array of scanlines where the
         * data of the channels is stored interleaved.
         *
         * @param width The width in pixels
         * @param height The height in pixels
         * @param tmpl A BitmapImage object which will be used as template.
         *             It's channel configuration will be used.
         * @param data The data to initialize the image with. If this is NULL,
         *             the image will be initialized with Zero
         *             (black/transparent) in all channels.
         */
        BitmapImage(unsigned int width, unsigned int height,
            const BitmapImage& tmpl, const void *data = NULL);

        /**
         * Copy ctor. Creates a deep copy of the source image.
         *
         * @param src The source image to create the deep copy from.
         * @param copyExt If true also any image extension will be copied
         */
        explicit BitmapImage(const BitmapImage& src, bool copyExt = false);

        /**
         * Ctor. Creates an empty bitmap (width and height are zero) with one
         * channel of the specified type and label. This ctor can be used to
         * create a BitmapImage object usable as template.
         *
         * @param type The type of the channels in this bitmap.
         * @param label1 The label for the first channel.
         */
        BitmapImage(ChannelType type, ChannelLabel label1);

        /**
         * Ctor. Creates an empty bitmap (width and height are zero) with two
         * channels of the specified type and label. This ctor can be used to
         * create a BitmapImage object usable as template.
         *
         * @param type The type of the channels in this bitmap.
         * @param label1 The label for the first channel.
         * @param label2 The label for the second channel.
         */
        BitmapImage(ChannelType type, ChannelLabel label1,
            ChannelLabel label2);

        /**
         * Ctor. Creates an empty bitmap (width and height are zero) with
         * three channels of the specified type and label. This ctor can be
         * used to create a BitmapImage object usable as template.
         *
         * @param type The type of the channels in this bitmap.
         * @param label1 The label for the first channel.
         * @param label2 The label for the second channel.
         * @param label3 The label for the thrid channel.
         */
        BitmapImage(ChannelType type, ChannelLabel label1,
            ChannelLabel label2, ChannelLabel label3);

        /**
         * Ctor. Creates an empty bitmap (width and height are zero) with four
         * channels of the specified type and label. This ctor can be used to
         * create a BitmapImage object usable as template.
         *
         * @param type The type of the channels in this bitmap.
         * @param label1 The label for the first channel.
         * @param label2 The label for the second channel.
         * @param label3 The label for the thrid channel.
         * @param label4 The label for the fourth channel.
         */
        BitmapImage(ChannelType type, ChannelLabel label1,
            ChannelLabel label2, ChannelLabel label3, ChannelLabel label4);

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
         * Creates a deep copy of the source image.
         *
         * @param src The source image to create the deep copy from.
         * @param copyExt If true also any image extension will be copied
         */
        void CopyFrom(const BitmapImage& src, bool copyExt = false);

        /**
         * Converts this BitmapImage to match the channel type and channel
         * labels of the template 'tmpl'.
         *
         * @param tmpl The template object defining the targeted channel type
         *             and channel labels.
         */
        void Convert(const BitmapImage& tmpl);

        /**
         * Copies the image data from the source BitmapImage 'src' to this
         * BitmapImage and converts the data to match the channel type and
         * channel labels of the template 'tmpl'.
         *
         * @param src The source BitmapImage.
         * @param tmpl The template object defining the targeted channel type
         *             and channel labels.
         */
        void ConvertFrom(const BitmapImage& src, const BitmapImage& tmpl);

        /**
         * Crops this imag to the specified rectangular region. The crop area
         * is truncated to the current size of the image if necessary.
         *
         * @param left The left position (minimum x) to crop from
         * @param top The top position (minimum y) to crop from
         * @param width The width of the crop rectangle and the new width for
         *              the image after cropping
         * @param height The height of the crop rectangle and the new height
         *               for the image after cropping
         */
        void Crop(unsigned int left, unsigned int top, unsigned int width,
            unsigned int height);

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
            unsigned int channels, ChannelType type, const void *data = NULL);

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
         * @param tmpl A BitmapImage object which will be used as template.
         *             It's channel configuration will be used.
         * @param data The data to initialize the image with. If this is NULL,
         *             the image will be initialized with Zero
         *             (black/transparent) in all channels.
         */
        void CreateImage(unsigned int width, unsigned int height,
            const BitmapImage& tmpl, const void *data = NULL);

        /**
         * Extracts a rectangular area from a source image. The extraction
         * area is truncated to the size of the source image if necessary.
         * The current image of this object will be replaced.
         *
         * @param src The source image
         * @param left The left position (minimum x) to extract from
         * @param top The top position (minimum y) to extract from
         * @param width The width of the extraction rectangle and the new
         *              width for the image after extraction
         * @param height The height of the extraction rectangle and the new
         *               height for the image after extraction
         */
        void ExtractFrom(const BitmapImage& src, unsigned int left,
            unsigned int top, unsigned int width, unsigned int height);

        /**
         * Flipps the image vertically
         */
        void FlipVertical(void);

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
         * Labels three channels with the labels "CHANNEL_RED",
         * "CHANNEL_GREEN", and "CHANNEL_BLUE".
         *
         * @param r The zero-based index of the channel to be labeled with
         *          "CHANNEL_RED". Default is channel 0.
         * @param g The zero-based index of the channel to be labeled with
         *          "CHANNEL_GREEN". Default is channel 1.
         * @param b The zero-based index of the channel to be labeled with
         *          "CHANNEL_BLUE". Default is channel 2.
         */
        inline void LabelChannelsRGB(unsigned int r = 0, unsigned int g = 1,
                unsigned int b = 2) {
            this->SetChannelLabel(r, CHANNEL_RED);
            this->SetChannelLabel(g, CHANNEL_GREEN);
            this->SetChannelLabel(b, CHANNEL_BLUE);
        }

        /**
         * Labels three channels with the labels "CHANNEL_RED",
         * "CHANNEL_GREEN", "CHANNEL_BLUE", and "CHANNEL_ALPHA".
         *
         * @param r The zero-based index of the channel to be labeled with
         *          "CHANNEL_RED". Default is channel 0.
         * @param g The zero-based index of the channel to be labeled with
         *          "CHANNEL_GREEN". Default is channel 1.
         * @param b The zero-based index of the channel to be labeled with
         *          "CHANNEL_BLUE". Default is channel 2.
         * @param a The zero-based index of the channel to be labeled with
         *          "CHANNEL_ALPHA". Default is channel 3.
         */
        inline void LabelChannelsRGBA(unsigned int r = 0, unsigned int g = 1,
                unsigned int b = 2, unsigned int a = 3) {
            this->SetChannelLabel(r, CHANNEL_RED);
            this->SetChannelLabel(g, CHANNEL_GREEN);
            this->SetChannelLabel(b, CHANNEL_BLUE);
            this->SetChannelLabel(a, CHANNEL_ALPHA);
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
         * Removes the image extension object
         */
        inline void RemoveExtension(void) {
            this->SetExtension(NULL);
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
         * Sets the image extension object
         *
         * @param ext The image extension object
         */
        void SetExtension(Extension *ext);

        /**
         * Answer the width of the image.
         *
         * @return The width of the image.
         */
        inline unsigned int Width(void) const {
            return this->width;
        }

    private:

        /**
         * Performs a crop-copy between two flat image storages of same format
         *
         * @param to The image data to copy to
         * @param from The image data to copy from
         * @param fromWidth The width in pixel of the from image data
         * @param fromHeight The height in pixel of the from image data
         * @param cropX The crop rectangle x position in the from image
         * @param cropY The crop rectangle y position in the from image
         * @param cropWidth The crop rectangle width in the from image
         * @param cropHeight The crop rectangle height in the from image
         * @param bpp The bytes per pixel
         */
        void cropCopy(char *to, char *from, unsigned int fromWidth,
            unsigned int fromHeight, unsigned int cropX, unsigned int cropY,
            unsigned int cropWidth, unsigned int cropHeight,
            unsigned int bpp);

        /** The raw image data */
        char *data;

        /** The type of the channel data of all channels */
        ChannelType chanType;

        /** The image extension */
        Extension *ext;

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

