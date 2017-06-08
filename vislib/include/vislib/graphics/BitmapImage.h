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

#include "vislib/forceinline.h"
#include "vislib/PtrArray.h"
#include "vislib/types.h"
#include "vislib/UnsupportedOperationException.h"


namespace vislib {
namespace graphics {


    /**
     * Class storing bitmap image data. The data is organized in channels.
     *
     * TODO: Add support for HSV channels
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
            virtual Extension * Clone(BitmapImage& newowner) const = 0;

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

        /** The of list of extensions */
        typedef PtrArray<Extension> ExtensionList;

        /** A BitmapImage template with Gray byte channels */
        static const BitmapImage TemplateByteGray;

        /** A BitmapImage template with Gray/Alpha byte channels */
        static const BitmapImage TemplateByteGrayAlpha;

        /** A BitmapImage template with RGB byte channels */
        static const BitmapImage TemplateByteRGB;

        /** A BitmapImage template with RGBA byte channels */
        static const BitmapImage TemplateByteRGBA;

        /** A BitmapImage template with Gray float channels */
        static const BitmapImage TemplateFloatGray;

        /** A BitmapImage template with Gray/Alpha float channels */
        static const BitmapImage TemplateFloatGrayAlpha;

        /** A BitmapImage template with RGB float channels */
        static const BitmapImage TemplateFloatRGB;

        /** A BitmapImage template with RGBA float channels */
        static const BitmapImage TemplateFloatRGBA;

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
         * Adds an image extension object
         *
         * @param ext The image extension object
         */
        inline void AddExtension(Extension *ext) {
            this->exts.Append(ext);
        }

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
         * Removes all image extension objects
         */
        inline void ClearExtensions(void) {
            this->exts.Clear();
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
         * Answers if this bitmap image has an equal channel layout (number,
         * type, and labels in the same order) as the specified template
         * 'tmpl'.
         *
         * @param tmpl The bitmap image template to compare to
         *
         * @return true if 'this' and 'tmpl' have identical channel layout
         */
        bool EqualChannelLayout(const BitmapImage& tmpl) const;

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
         * Finds the extension of the specified type
         *
         * @return The first extension object of the specified type or NULL
         *         if no extension object of this type is present
         */
        template<class Tp>
        inline Tp* FindExtension(void) {
            SIZE_T extCnt = this->exts.Count();
            for (SIZE_T i = 0; i < extCnt; i++) {
                Tp *ptr = dynamic_cast<Tp *>(this->exts[i]);
                if (ptr != NULL) return ptr;
            }
            return NULL;
        }

        /**
         * Finds the extension of the specified type
         *
         * @return The first extension object of the specified type or NULL
         *         if no extension object of this type is present
         */
        template<class Tp>
        inline const Tp* FindExtension(void) const {
            SIZE_T extCnt = this->exts.Count();
            for (SIZE_T i = 0; i < extCnt; i++) {
                const Tp *ptr = dynamic_cast<Tp *>(this->exts[i]);
                if (ptr != NULL) return ptr;
            }
            return NULL;
        }

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
         * Answer the image extension object
         *
         * @return The image extension object
         */
        inline ExtensionList& GetExtensions(void) {
            return this->exts;
        }

        /**
         * Answer the image extension object
         *
         * @return The image extension object
         */
        inline const ExtensionList& GetExtensions(void) const {
            return this->exts;
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
         * Answer whether or not an alpha channel is present
         *
         * @return true if an alpha channel is present
         */
        inline bool HasAlpha(void) const {
            return this->HasChannel(CHANNEL_ALPHA);
        }

        /**
         * Answer whether or not a channel is labeled with the specified label
         *
         * @return true if at least on channel is labeled with the specified
         *         label
         */
        bool HasChannel(ChannelLabel label) const;

        /**
         * Answer whether or not a gray channel is present
         *
         * @return true if a gray channel is present
         */
        inline bool HasGray(void) const {
            return this->HasChannel(CHANNEL_GRAY);
        }

        /**
         * Answer whether or not all three RGB channels are present
         *
         * @return true if all three RGB channels are present
         */
        inline bool HasRGB(void) const {
            return this->HasChannel(CHANNEL_RED);
            return this->HasChannel(CHANNEL_GREEN);
            return this->HasChannel(CHANNEL_BLUE);
        }

        /**
         * Inverts the values of all colour channels
         */
        void Invert(void);

        /**
         * Inverts the values of all colour channels
         *
         * @param channel The colour channel to be inverted
         */
        void Invert(unsigned int channel);

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
         * Removes the image extension object.
         *
         * @param ext The extension object to be removed. If NULL all
         *            extension objects will be removed.
         */
        inline void RemoveExtension(Extension *ext = NULL) {
            if (ext == NULL) {
                this->exts.Clear();
            } else {
                this->exts.RemoveAll(ext);
            }
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
         * Sets the image extension object by removing all currently set
         * extensions and adding the specified one
         *
         * @param ext The image extension object
         */
        inline void SetExtension(Extension *ext) {
            this->exts.Clear();
            this->exts.Append(ext);
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

        /**
         * Utility class helping with image conversions
         * @param ST the source buffer type
         */
        template<class ST>
        class Conversion {
        public:

            /**
             * Copies a single channel of a single pixel
             *
             * @param dst The destination
             * @param src The source
             */
            template<class T>
            static VISLIB_FORCEINLINE void CopyBit(T& dst, const T& src) {
                dst = src;
            }

            /**
             * Copies a single channel of a single pixel
             *
             * @param dst The destination
             * @param src The source
             */
            static VISLIB_FORCEINLINE void CopyBit(unsigned char& dst,
                    const unsigned short& src) {
                dst = static_cast<unsigned char>(src / 256);
            }

            /**
             * Copies a single channel of a single pixel
             *
             * @param dst The destination
             * @param src The source
             */
            static VISLIB_FORCEINLINE void CopyBit(unsigned char& dst,
                    const float& src) {
                dst = static_cast<unsigned char>(((src < 0.0f) ? 0
                    : ((src > 1.0f) ? 1.0f : src)) * 255.0f);
            }

            /**
             * Copies a single channel of a single pixel
             *
             * @param dst The destination
             * @param src The source
             */
            static VISLIB_FORCEINLINE void CopyBit(unsigned short& dst,
                    const unsigned char& src) {
                dst = (static_cast<unsigned short>(src) << 8)
                    + static_cast<unsigned short>(src);
            }

            /**
             * Copies a single channel of a single pixel
             *
             * @param dst The destination
             * @param src The source
             */
            static VISLIB_FORCEINLINE void CopyBit(unsigned short& dst,
                    const float& src) {
                dst = static_cast<unsigned char>(((src < 0.0f) ? 0
                    : ((src > 1.0f) ? 1.0f : src)) * 65535.0f);
            }

            /**
             * Copies a single channel of a single pixel
             *
             * @param dst The destination
             * @param src The source
             */
            static VISLIB_FORCEINLINE void CopyBit(float& dst,
                    const unsigned char& src) {
                dst = static_cast<float>(src) / 255.0f;
            }

            /**
             * Copies a single channel of a single pixel
             *
             * @param dst The destination
             * @param src The source
             */
            static VISLIB_FORCEINLINE void CopyBit(float& dst,
                    const unsigned short& src) {
                dst = static_cast<float>(src) / 65535.0f;
            }

            /**
             * Function callback definition for channel source data
             *
             * @param conv The calling conversion object
             * @param param The channel parameter
             *
             * @return The value of the current pixel for this channel
             */
            typedef float (*ChannelSourceFunc)(Conversion* conv, int param);

            /** The possible source channels */
            enum SourceChannel {
                SC_UNDEFINED = 0,
                SC_RED,
                SC_GREEN,
                SC_BLUE,
                SC_GRAY,
                SC_ALPHA,
                SC_CMYK_CYAN,
                SC_CMYK_MAGENTA,
                SC_CMYK_YELLOW,
                SC_CMYK_BLACK,
                SC_CMY_CYAN,
                SC_CMY_MAGENTA,
                SC_CMY_YELLOW,
                SC_LASTCHANNEL
            };

            /**
             * Ctor
             *
             * @param source The first pixel in the source buffer
             * @param chanCnt The number of channels in the source buffer
             */
            Conversion(ST* source, unsigned int chanCnt);

            /** Dtor */
            ~Conversion(void);

            /**
             * Add a source channel definition to the input data
             *
             * @param chan The number of the source channel
             * @param label The label of the source channel
             */
            void AddSourceChannel(unsigned int chan, ChannelLabel label);

            /**
             * Completes the initialization after adding all source channels
             */
            void FinalizeInitialization(void);

            /**
             * Calculates the destination<->source channel mapping
             *
             * @param map The channel map to be calculated
             * @param chan The destination channel requests
             * @param cnt The number of destination channels
             */
            void ChannelMapping(SourceChannel *map, ChannelLabel *chan,
                unsigned int cnt);

            /**
             * Gets the value for the specified source channel of the current
             * pixel
             *
             * @param chan The requested source channel
             *
             * @return The requested value
             */
            VISLIB_FORCEINLINE float GetValue(SourceChannel chan) {
                return this->func[chan](this, this->param[chan]);
            }

            /**
             * Advances the source pointer one pixel
             *
             * @return A reference to this
             */
            Conversion& operator++(void) {
                this->source += this->sourceChanCnt;
                return *this;
            }

        private:

            /**
             * Channel source function for constant value one
             *
             * @param conv The calling conversion object
             * @param param The channel parameter
             *
             * @return The value of the current pixel for this channel
             */
            static float constOne(Conversion *conv, int param);

            /**
             * Channel source function for constant value zero
             *
             * @param conv The calling conversion object
             * @param param The channel parameter
             *
             * @return The value of the current pixel for this channel
             */
            static float constZero(Conversion *conv, int param);

            /**
             * Channel source function directly copying from the source buffer
             *
             * @param conv The calling conversion object
             * @param param The channel parameter
             *
             * @return The value of the current pixel for this channel
             */
            static float directSource(Conversion *conv, int param);

            /**
             * Channel source function calculating gray from RGB
             *
             * @param conv The calling conversion object
             * @param param The channel parameter
             *
             * @return The value of the current pixel for this channel
             */
            static float grayFromRGB(Conversion *conv, int param);

            /**
             * Channel source function calculating RGB from CMY
             *
             * @param conv The calling conversion object
             * @param param The channel parameter
             *
             * @return The value of the current pixel for this channel
             */
            static float rgbFromCMY(Conversion *conv, int param);

            /**
             * Channel source function calculating CMY from RGB
             *
             * @param conv The calling conversion object
             * @param param The channel parameter
             *
             * @return The value of the current pixel for this channel
             */
            static float cmyFromRGB(Conversion *conv, int param);

            /**
             * Channel source function calculating CMY from CMYK
             *
             * @param conv The calling conversion object
             * @param param The channel parameter
             *
             * @return The value of the current pixel for this channel
             */
            static float cmyFromCMYK(Conversion *conv, int param);

            /**
             * Channel source function calculating CMYK from CMY
             *
             * @param conv The calling conversion object
             * @param param The channel parameter
             *
             * @return The value of the current pixel for this channel
             */
            static float cmykFromCMY(Conversion *conv, int param);

            /**
             * Forbidden copy ctor
             *
             * @param src The source object
             */
            Conversion(const Conversion& src) {
                throw vislib::UnsupportedOperationException(
                    "Conversion::CopyCtor", __FILE__, __LINE__);
            }

            /**
             * Forbidden assignment operator
             *
             * @rhs The right hand side operand
             *
             * @return A reference to this
             */
            Conversion& operator=(const Conversion& rhs) {
                if (this != &rhs) {
                    throw vislib::UnsupportedOperationException(
                        "Conversion::operator=", __FILE__, __LINE__);
                }
                return *this;
            }

            /** The current source position */
            ST* source;

            /** The number of source channels */
            unsigned int sourceChanCnt;

            /** The channel source functions */
            ChannelSourceFunc func[static_cast<int>(SC_LASTCHANNEL)];

            /** The channel source parameters */
            int param[static_cast<int>(SC_LASTCHANNEL)];

        };

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

        /**
         * Copies all pixels from one BitmapImage buffer into another one
         *
         * @param w The width of both of the buffers
         * @param h The height of both of the buffers
         * @param dst The destination buffer in correct type
         * @param src The source buffer in correct type
         * @param srcChanCnt The number of channels in the source buffer
         * @param chanMap The channel map from destination to source
         * @param chanCnt The number of channels in the channel map and in the
         *                destination buffer
         */
        template<class DT, class ST>
        void copyBits(unsigned int w, unsigned int h, DT *dst, ST *src,
            unsigned int srcChanCnt, int *chanMap, unsigned int chanCnt);

        /**
         * Performs a full image conversion from one BitmapImage buffer into another one
         *
         * @param w The width of both of the buffers
         * @param h The height of both of the buffers
         * @param dst The destination buffer in correct type
         * @param dstChan The destination channels labels
         * @param dstChanCnt The number of destination channels
         * @param src The source buffer in correct type
         * @param srcChan The source channels labels
         * @param srcChanCnt The number of source channels
         */
        template<class DT, class ST>
        void fullConvert(unsigned int w, unsigned int h,
            DT* dst, ChannelLabel *dstChan, unsigned int dstChanCnt,
            ST* src, ChannelLabel *srcChan, unsigned int srcChanCnt);

        /**
         * Inverts colour channels
         *
         * @param maxval The maximum value
         * @param chan The number of the channel to invert or UINT_MAX if all
         *             colour channels should be inverted
         */
        template<class T> void invert(T maxval, unsigned int chan);

        /** The raw image data */
        char *data;

        /** The type of the channel data of all channels */
        ChannelType chanType;

        /** The image extension */
        ExtensionList exts;

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

