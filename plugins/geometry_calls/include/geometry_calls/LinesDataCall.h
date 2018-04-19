/*
 * LinesDataCall.h
 *
 * Copyright (C) 2010-2018 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOL_GEOMETRY_CALLS_LINESDATACALL_H_INCLUDED
#define MEGAMOL_GEOMETRY_CALLS_LINESDATACALL_H_INCLUDED
#pragma once

#include "geometry_calls/geometry_calls.h"
#include "mmcore/AbstractGetData3DCall.h"
#include "mmcore/factories/CallAutoDescription.h"
#include "vislib/assert.h"
#include "vislib/graphics/ColourRGBAu8.h"
#include "vislib/forceinline.h"


namespace megamol {
namespace geocalls {


    /**
     * Call for lines data
     */
    class GEOMETRY_CALLS_API LinesDataCall : public core::AbstractGetData3DCall {
    public:

        /**
         * Class storing a list of lines.
         * All elements are stored in flat lists without any stride!
         * The graphical primitive is GL_LINES, so the number of elements must
         * be a multiple of 2, since each line segment must explicitly store
         * it's start point and end point. In a line strip the inner points
         * must be multiplied. You can use the index array to reduce the
         * memory overhead in the colour and vertex array.
         */
        class GEOMETRY_CALLS_API Lines {
        public:
            class VertexData_Detail {
            public:
                virtual float const GetXf() const = 0;
                virtual float const GetYf() const = 0;
                virtual float const GetZf() const = 0;
                virtual double const GetXd() const = 0;
                virtual double const GetYd() const = 0;
                virtual double const GetZd() const = 0;
                virtual void SetBasePtr(void const* ptr) = 0;
                virtual std::unique_ptr<VertexData_Detail> Clone() const = 0;
                virtual ~VertexData_Detail() = default;
            };

            class VertexData_None : public VertexData_Detail {
            public:
                VertexData_None() = default;

                VertexData_None(VertexData_None const& rhs) = default;

                virtual float const GetXf() const override {
                    return 0.0f;
                }

                virtual float const GetYf() const override {
                    return 0.0f;
                }

                virtual float const GetZf() const override {
                    return 0.0f;
                }

                virtual double const GetXd() const override {
                    return 0;
                }

                virtual double const GetYd() const override {
                    return 0;
                }

                virtual double const GetZd() const override {
                    return 0;
                }

                virtual void SetBasePtr(void const* ptr) override { };

                virtual std::unique_ptr<VertexData_Detail> Clone() const override {
                    return std::unique_ptr<VertexData_Detail>{new VertexData_None{*this}};
                }
            };

            template<class T>
            class VertexData_Impl : public VertexData_Detail {
            public:
                VertexData_Impl() = default;

                VertexData_Impl(VertexData_Impl const& rhs)
                    : basePtr{rhs.basePtr} { }

                virtual float const GetXf() const override {
                    return GetX<float>();
                }

                virtual double const GetXd() const override {
                    return GetX<double>();
                }

                template<class R>
                std::enable_if_t<std::is_same<T, R>::value, R> const GetX() const {
                    return this->basePtr[0];
                }

                template<class R>
                std::enable_if_t<!std::is_same<T, R>::value, R> const GetX() const {
                    return static_cast<R>(this->basePtr[0]);
                }

                virtual float const GetYf() const override {
                    return GetY<float>();
                }

                virtual double const GetYd() const override {
                    return GetY<double>();
                }

                template<class R>
                std::enable_if_t<std::is_same<T, R>::value, R> const GetY() const {
                    return this->basePtr[1];
                }

                template<class R>
                std::enable_if_t<!std::is_same<T, R>::value, R> const GetY() const {
                    return static_cast<R>(this->basePtr[1]);
                }

                virtual float const GetZf() const override {
                    return GetZ<float>();
                }

                virtual double const GetZd() const override {
                    return GetZ<double>();
                }

                template<class R>
                std::enable_if_t<std::is_same<T, R>::value, R> const GetZ() const {
                    return this->basePtr[2];
                }

                template<class R>
                std::enable_if_t<!std::is_same<T, R>::value, R> const GetZ() const {
                    return static_cast<R>(this->basePtr[2]);
                }

                virtual void SetBasePtr(void const* ptr) override {
                    this->basePtr = reinterpret_cast<T const*>(ptr);
                }

                virtual std::unique_ptr<VertexData_Detail> Clone() const override {
                    return std::unique_ptr<VertexData_Detail>{new VertexData_Impl{*this}};
                }
            private:
                T const* basePtr;
            };

            class VertexData_Base {
            public:
                VertexData_Base(std::unique_ptr<VertexData_Detail>&& impl, void const* basePtr)
                    : pimpl{std::forward<std::unique_ptr<VertexData_Detail>>(impl)} {
                    pimpl->SetBasePtr(basePtr);
                }

                VertexData_Base(VertexData_Base const& rhs) = delete;

                VertexData_Base(VertexData_Base&& rhs)
                    : pimpl{std::forward<std::unique_ptr<VertexData_Detail>>(rhs.pimpl)} { }

                VertexData_Base& operator=(VertexData_Base const& rhs) = delete;

                VertexData_Base& operator=(VertexData_Base&& rhs) {
                    pimpl = std::move(rhs.pimpl);
                    return *this;
                }

                float const GetXf() const {
                    return pimpl->GetXf();
                }

                float const GetYf() const {
                    return pimpl->GetYf();
                }

                float const GetZf() const {
                    return pimpl->GetZf();
                }

                double const GetXd() const {
                    return pimpl->GetXd();
                }

                double const GetYd() const {
                    return pimpl->GetYd();
                }

                double const GetZd() const {
                    return pimpl->GetZd();
                }
            private:
                std::unique_ptr<VertexData_Detail> pimpl;
            };

            class IndexData_Detail {
            public:
                virtual uint8_t const GetIDXu8() const = 0;
                virtual uint16_t const GetIDXu16() const = 0;
                virtual uint32_t const GetIDXu32() const = 0;
                virtual void SetBasePtr(void const* ptr) = 0;
                virtual std::unique_ptr<IndexData_Detail> Clone() const = 0;
                virtual ~IndexData_Detail() = default;
            };

            class IndexData_None : public IndexData_Detail {
            public:
                IndexData_None() = default;

                IndexData_None(IndexData_None const& rhs) = default;

                virtual uint8_t const GetIDXu8() const override {
                    return 0;
                }

                virtual uint16_t const GetIDXu16() const override {
                    return 0;
                }

                virtual uint32_t const GetIDXu32() const override {
                    return 0;
                }

                virtual void SetBasePtr(void const* ptr) override { }

                virtual std::unique_ptr<IndexData_Detail> Clone() const override {
                    return std::unique_ptr<IndexData_Detail>{new IndexData_None{*this}};
                }
            };

            template<class T>
            class IndexData_Impl : public IndexData_Detail {
            public:
                IndexData_Impl() = default;

                IndexData_Impl(IndexData_Impl const& rhs)
                    : basePtr{rhs.basePtr} { }

                virtual uint8_t const GetIDXu8() const override {
                    return GetIDX<uint8_t>();
                }

                virtual uint16_t const GetIDXu16() const override {
                    return GetIDX<uint16_t>();
                }

                virtual uint32_t const GetIDXu32() const override {
                    return GetIDX<uint32_t>();
                }

                template<class R>
                std::enable_if_t<std::is_same<T, R>::value, R> const GetIDX() const {
                    return this->basePtr[0];
                }

                template<class R>
                std::enable_if_t<!std::is_same<T, R>::value, R> const GetIDX() const {
                    return static_cast<R>(this->basePtr[0]);
                }

                virtual void SetBasePtr(void const* ptr) override {
                    this->basePtr = reinterpret_cast<T const*>(ptr);
                }

                virtual std::unique_ptr<IndexData_Detail> Clone() const override {
                    return std::unique_ptr<IndexData_Detail>{new IndexData_Impl{*this}};
                }

            private:
                T const* basePtr;
            };

            class IndexData_Base {
            public:
                IndexData_Base(std::unique_ptr<IndexData_Detail>&& impl, void const* basePtr)
                    : pimpl{std::forward<std::unique_ptr<IndexData_Detail>>(impl)} {
                    pimpl->SetBasePtr(basePtr);
                }

                IndexData_Base(IndexData_Base const& rhs) = delete;

                IndexData_Base(IndexData_Base&& rhs)
                    : pimpl{std::forward<std::unique_ptr<IndexData_Detail>>(rhs.pimpl)} {

                }

                IndexData_Base& operator=(IndexData_Base const& rhs) = delete;

                IndexData_Base& operator=(IndexData_Base&& rhs) {
                    pimpl = std::move(rhs.pimpl);
                    return *this;
                }

                uint8_t const GetIDXu8() const {
                    return pimpl->GetIDXu8();
                }

                uint16_t const GetIDXu16() const {
                    return pimpl->GetIDXu16();
                }

                uint32_t const GetIDXu32() const {
                    return pimpl->GetIDXu32();
                }

            private:
                std::unique_ptr<IndexData_Detail> pimpl;
            };

            class ColorData_Detail {
            public:
                virtual uint8_t const GetRu8() const = 0;
                virtual uint8_t const GetGu8() const = 0;
                virtual uint8_t const GetBu8() const = 0;
                virtual uint8_t const GetAu8() const = 0;
                virtual float const GetRf() const = 0;
                virtual float const GetGf() const = 0;
                virtual float const GetBf() const = 0;
                virtual float const GetAf() const = 0;
                virtual double const GetRd() const = 0;
                virtual double const GetGd() const = 0;
                virtual double const GetBd() const = 0;
                virtual double const GetAd() const = 0;
                virtual void SetBasePtr(void const* ptr) = 0;
                virtual std::unique_ptr<ColorData_Detail> Clone() const = 0;
                virtual ~ColorData_Detail() = default;
            };

            class ColorData_None : public ColorData_Detail {
            public:
                ColorData_None() = default;

                ColorData_None(ColorData_None const& rhs) = default;

                virtual uint8_t const GetRu8() const override {
                    return 0;
                }

                virtual uint8_t const GetGu8() const override {
                    return 0;
                }

                virtual uint8_t const GetBu8() const override {
                    return 0;
                }

                virtual uint8_t const GetAu8() const override {
                    return 0;
                }

                virtual float const GetRf() const override {
                    return 0.0f;
                }

                virtual float const GetGf() const override {
                    return 0.0f;
                }

                virtual float const GetBf() const override {
                    return 0.0f;
                }

                virtual float const GetAf() const override {
                    return 0.0f;
                }

                virtual double const GetRd() const override {
                    return 0.0;
                }

                virtual double const GetGd() const override {
                    return 0.0;
                }

                virtual double const GetBd() const override {
                    return 0.0;
                }
                virtual double const GetAd() const override {
                    return 0.0;
                }

                virtual void SetBasePtr(void const* ptr) override { }

                virtual std::unique_ptr<ColorData_Detail> Clone() const override {
                    return std::unique_ptr<ColorData_Detail>{new ColorData_None{*this}};
                }
            };

            template<class T, bool hasAlpha>
            class ColorData_Impl : public ColorData_Detail {
            public:
                virtual uint8_t const GetRu8() const override {
                    return GetR<uint8_t>();
                }

                virtual float const GetRf() const override {
                    return GetR<float>();
                }

                virtual double const GetRd() const override {
                    return GetR<double>();
                }

                template<class R>
                std::enable_if_t<std::is_same<T, R>::value, R> const GetR() const {
                    return this->basePtr[0];
                }

                template<class R>
                std::enable_if_t<!std::is_same<T, R>::value, R> const GetR() const {
                    return static_cast<R>(this->basePtr[0]);
                }

                virtual uint8_t const GetGu8() const override {
                    return GetG<uint8_t>();
                }

                virtual float const GetGf() const override {
                    return GetG<float>();
                }

                virtual double const GetGd() const override {
                    return GetG<double>();
                }

                template<class R>
                std::enable_if_t<std::is_same<T, R>::value, R> const GetG() const {
                    return this->basePtr[1];
                }

                template<class R>
                std::enable_if_t<!std::is_same<T, R>::value, R> const GetG() const {
                    return static_cast<R>(this->basePtr[1]);
                }

                virtual uint8_t const GetBu8() const override {
                    return GetB<uint8_t>();
                }

                virtual float const GetBf() const override {
                    return GetB<float>();
                }

                virtual double const GetBd() const override {
                    return GetB<double>();
                }

                template<class R>
                std::enable_if_t<std::is_same<T, R>::value, R> const GetB() const {
                    return this->basePtr[2];
                }

                template<class R>
                std::enable_if_t<!std::is_same<T, R>::value, R> const GetB() const {
                    return static_cast<R>(this->basePtr[2]);
                }

                virtual uint8_t const GetAu8() const override {
                    return GetA<uint8_t, hasAlpha>();
                }

                virtual float const GetAf() const override {
                    return GetA<float, hasAlpha>();
                }

                virtual double const GetAd() const override {
                    return GetA<double, hasAlpha>();
                }

                template<class R, bool hasAlpha_v>
                std::enable_if_t<std::is_same<T, R>::value && hasAlpha_v, R> const GetA() const {
                    return this->basePtr[3];
                }

                template<class R, bool hasAlpha_v>
                std::enable_if_t<!std::is_same<T, R>::value && hasAlpha_v, R> const GetA() const {
                    return static_cast<R>(this->basePtr[3]);
                }

                template<class R, bool hasAlpha_v>
                std::enable_if_t<!hasAlpha_v, R> const GetA() const {
                    return static_cast<R>(0.0);
                }

                virtual void SetBasePtr(void const* ptr) override {
                    this->basePtr = reinterpret_cast<T const*>(ptr);
                }

                virtual std::unique_ptr<ColorData_Detail> Clone() const override {
                    return std::unique_ptr<ColorData_Detail>{new ColorData_Impl{*this}};
                }
            private:
                T const* basePtr;
            };

            class ColorData_Base {
            public:
                ColorData_Base(std::unique_ptr<ColorData_Detail>&& impl, void const* basePtr)
                    : pimpl{std::forward<std::unique_ptr<ColorData_Detail>>(impl)} {
                    pimpl->SetBasePtr(basePtr);
                }

                ColorData_Base(ColorData_Base const& rhs) = delete;

                ColorData_Base(ColorData_Base&& rhs)
                    : pimpl{std::forward<std::unique_ptr<ColorData_Detail>>(rhs.pimpl)} {

                }

                ColorData_Base& operator=(ColorData_Base const& rhs) = delete;

                ColorData_Base& operator=(ColorData_Base&& rhs) {
                    pimpl = std::move(rhs.pimpl);
                    return *this;
                }

                uint8_t const GetRu8() const {
                    return pimpl->GetRu8();
                }
                uint8_t const GetGu8() const {
                    return pimpl->GetGu8();
                }
                uint8_t const GetBu8() const {
                    return pimpl->GetBu8();
                }
                uint8_t const GetAu8() const {
                    return pimpl->GetAu8();
                }
                float const GetRf() const {
                    return pimpl->GetRf();
                }
                float const GetGf() const {
                    return pimpl->GetGf();
                }
                float const GetBf() const {
                    return pimpl->GetBf();
                }
                float const GetAf() const {
                    return pimpl->GetAf();
                }
                double const GetRd() const {
                    return pimpl->GetRd();
                }
                double const GetGd() const {
                    return pimpl->GetGd();
                }
                double const GetBd() const {
                    return pimpl->GetBd();
                }
                double const GetAd() const {
                    return pimpl->GetAd();
                }
            private:
                std::unique_ptr<ColorData_Detail> pimpl;
            };

            /** Struct holding pointers into data streams for a specific vertex */
            struct vertex_t {
                vertex_t(VertexData_Base&& v, ColorData_Base&& c)
                    : vert{std::forward<VertexData_Base>(v)}
                    , col{std::forward<ColorData_Base>(c)} { }

                vertex_t(vertex_t const& rhs) = delete;

                vertex_t(vertex_t&& rhs)
                    : vert{std::move(rhs.vert)}
                    , col{std::move(rhs.col)} { }

                vertex_t& operator=(vertex_t const& rhs) = delete;

                vertex_t& operator=(vertex_t&& rhs) {
                    vert = std::move(rhs.vert);
                    col = std::move(rhs.col);
                    return *this;
                }

                VertexData_Base vert;

                ColorData_Base col;
            };

            /** Struct holding a pointer into index stream */
            struct index_t {
                index_t(IndexData_Base&& i)
                    : idx{std::forward<IndexData_Base>(i)} { }

                index_t(index_t const& rhs) = delete;

                index_t(index_t&& rhs)
                    : idx{std::move(rhs.idx)} { }

                index_t& operator=(index_t const& rhs) = delete;

                index_t& operator=(index_t&& rhs) {
                    idx = std::move(rhs.idx);
                    return *this;
                }

                IndexData_Base idx;
            };

            /** The possible colour data types */
            enum ColourDataType {
                CDT_NONE = 0,
                CDT_BYTE_RGB = 1,
                CDT_BYTE_RGBA = 2,
                CDT_FLOAT_RGB = 3,
                CDT_FLOAT_RGBA = 4,
                CDT_DOUBLE_RGB = 5,
                CDT_DOUBLE_RGBA = 6
            };

            /** Possible data types */
            enum DataType {
                DT_NONE = 0,
                DT_BYTE = 1, // UINT8
                DT_UINT16 = 2,
                DT_UINT32 = 3,
                DT_FLOAT = 4,
                DT_DOUBLE = 5
            };

            /** Possible color strides */
            size_t const ColorDataStride[7] = {
                0, 3, 4, 12, 16, 24, 32
            };

            /** Possible data strides */
            size_t const DataStride[6] = {
                0, 1, 2, 4, 4, 8
            };

            /**
             * Ctor
             */
            Lines(void);

            /**
             * CCtor
             */
            Lines(Lines const& rhs);

            /**
             * Dtor
             */
            ~Lines(void);

            /**
             * Removes all data
             */
            inline void Clear() {
                // TODO: Why are only specific data pointers nulled here?
                this->count = 0;
                this->vrtDT = DT_NONE;
                this->vrt.dataFloat = nullptr;
                this->colDT = CDT_NONE;
                this->col.dataByte = nullptr;
                this->idxDT = DT_NONE;
                this->idx.dataByte = nullptr;
                this->globCol.Set(0, 0, 0, 0);
            }

            /**
             * Answer the number of elements. If 'IndexArray' is NULL this
             * is the number of vertex (and colour) entries stored. If
             * 'IndexArray' is not NULL this is the number of index entries
             * stored.
             *
             * @return The number of elements
             */
            inline unsigned int Count(void) const {
                return this->count;
            }

            /**
             * The colour data type
             *
             * @return The colour data type
             */
            inline ColourDataType ColourArrayType(void) const {
                return this->colDT;
            }

            /**
             * Answer the colour array. This can be NULL if the global colour
             * data should be used
             *
             * @return The colour array
             */
            inline const unsigned char *ColourArrayByte(void) const {
                ASSERT((this->colDT == CDT_BYTE_RGB) || (this->colDT == CDT_BYTE_RGBA));
                return this->col.dataByte;
            }

            /**
             * Answer the colour array. This can be NULL if the global colour
             * data should be used
             *
             * @return The colour array
             */
            inline const float *ColourArrayFloat(void) const {
                ASSERT((this->colDT == CDT_FLOAT_RGB) || (this->colDT == CDT_FLOAT_RGBA));
                return this->col.dataFloat;
            }

            /**
             * Answer the colour array. This can be NULL if the global colour
             * data should be used
             *
             * @return The colour array
             */
            inline const double *ColourArrayDouble(void) const {
                ASSERT((this->colDT == CDT_DOUBLE_RGB) || (this->colDT == CDT_DOUBLE_RGBA));
                return this->col.dataDouble;
            }

            /**
             * Answer the global colour value
             *
             * @return The global colour value
             */
            inline const vislib::graphics::ColourRGBAu8& GlobalColour(void) const {
                return this->globCol;
            }

            /**
            * Answer the ID of this line
            *
            * @return The ID
            */
            inline const size_t ID(void) const {
                return this->id;
            }

            /**
             * The data type of the index array
             *
             * @return The data type of the index array
             */
            inline DataType IndexArrayDataType(void) const {
                return this->idxDT;
            }

            /**
             * Answer the index array. This can be NULL.
             *
             * @return The index array
             */
            inline const unsigned char *IndexArrayByte(void) const {
                ASSERT((this->idx.dataByte == NULL) || (this->idxDT == DT_BYTE));
                return this->idx.dataByte;
            }

            /**
             * Answer the index array. This can be NULL.
             *
             * @return The index array
             */
            inline const unsigned short *IndexArrayUInt16(void) const {
                ASSERT((this->idx.dataByte == NULL) || (this->idxDT == DT_UINT16));
                return this->idx.dataUInt16;
            }

            /**
             * Answer the index array. This can be NULL.
             *
             * @return The index array
             */
            inline const unsigned int *IndexArrayUInt32(void) const {
                ASSERT((this->idx.dataByte == NULL) || (this->idxDT == DT_UINT32));
                return this->idx.dataUInt32;
            }

            /**
             * Sets the data for this object. Ownership to all memory all
             * pointers point to will not be take by this object. The owner
             * must ensure that these pointers remain valid as long as they
             * are used. None of the pointers may be NULL; Use the proper
             * version of this method instead.
             *
             * @param cnt The number of elements
             * @param vert The vertex array (XYZ-Float)
             * @param col The global colour to be used for all lines
             */
            template<class Tp>
            inline void Set(unsigned int cnt, Tp vert, vislib::graphics::ColourRGBAu8 col) {
                ASSERT(vert != NULL);
                this->count = cnt;
                this->setVrtData(vert);
                this->colDT = CDT_NONE;
                this->col.dataByte = NULL;
                this->idxDT = DT_NONE;
                this->idx.dataByte = NULL;
                this->globCol = col;
            }

            /**
             * Sets the data for this object. Ownership to all memory all
             * pointers point to will not be take by this object. The owner
             * must ensure that these pointers remain valid as long as they
             * are used. None of the pointers may be NULL; Use the proper
             * version of this method instead.
             *
             * @param cnt The number of elements
             * @param idx The index array (UInt32)
             * @param vert The vertex array (XYZ-Float)
             * @param col The global colour to be used for all lines
             */
            template<class Tp1, class Tp2>
            inline void Set(unsigned int cnt, Tp1 idx, Tp2 vert, vislib::graphics::ColourRGBAu8 col) {
                ASSERT(idx != NULL);
                ASSERT(vert != NULL);
                this->count = cnt;
                this->setVrtData(vert);
                this->setIdxData(idx);
                this->colDT = CDT_NONE;
                this->col.dataByte = NULL;
                this->globCol = col;
            }

            /**
             * Sets the data for this object. Ownership to all memory all
             * pointers point to will not be take by this object. The owner
             * must ensure that these pointers remain valid as long as they
             * are used. None of the pointers may be NULL; Use the proper
             * version of this method instead.
             *
             * @param cnt The number of elements
             * @param vert The vertex array (XYZ-Float)
             * @param col The colour array (use same number of entries as the
             *            vertex array)
             * @param withAlpha Flag if the colour array contains RGBA(true)
             *                  or RGB(false) values
             */
            template<class Tp1, class Tp2>
            inline void Set(unsigned int cnt, Tp1 vert, Tp2 col, bool withAlpha) {
                ASSERT(vert != NULL);
                ASSERT(col != NULL);
                this->count = cnt;
                this->setVrtData(vert);
                this->setColData(col, withAlpha);
                this->idxDT = DT_NONE;
                this->idx.dataUInt32 = NULL;
                this->globCol.Set(0, 0, 0, 255);
            }

            /**
             * Sets the data for this object. Ownership to all memory all
             * pointers point to will not be take by this object. The owner
             * must ensure that these pointers remain valid as long as they
             * are used. None of the pointers may be NULL; Use the proper
             * version of this method instead.
             *
             * @param cnt The number of elements
             * @param idx The index array (UInt32)
             * @param vert The vertex array (XYZ-Float)
             * @param col The colour array (use same number of entries as the
             *            vertex array)
             * @param withAlpha Flag if the colour array contains RGBA(true)
             *                  or RGB(false) values
             */
            template<class Tp1, class Tp2, class Tp3>
            inline void Set(unsigned int cnt, Tp1 idx, Tp2 vert, Tp3 col, bool withAlpha) {
                ASSERT(idx != NULL);
                ASSERT(vert != NULL);
                ASSERT(col != NULL);
                this->count = cnt;
                this->setVrtData(vert);
                this->setColData(col, withAlpha);
                this->setIdxData(idx);
                this->globCol.Set(0, 0, 0, 255);
            }

            /**
             * Sets the list ID
             *
             * @param ID the list ID
             */
            inline void SetID(size_t ID) {
                this->id = ID;
            }

            /**
             * Answer the data type of the vertex array
             *
             * @return The data type of the vertex array
             */
            inline DataType VertexArrayDataType(void) const {
                return this->vrtDT;
            }

            /**
             * Answer the vertex array (XYZ-Float)
             *
             * @return The vertex array
             */
            inline const float *VertexArrayFloat(void) const {
                ASSERT(this->vrtDT == DT_FLOAT);
                return this->vrt.dataFloat;
            }

            /**
             * Answer the vertex array (XYZ-Double)
             *
             * @return The vertex array
             */
            inline const double *VertexArrayDouble(void) const {
                ASSERT(this->vrtDT == DT_DOUBLE);
                return this->vrt.dataDouble;
            }

            /**
             * Access vertex at index without range check.
             * 
             * @param idx Index of vertex in the streams.
             * 
             * @return Struct of pointers to positions of the vertex in the streams.
             */
            vertex_t operator[](size_t idx) const noexcept {
                return vertex_t{
                    VertexData_Base{this->vertexAccessor->Clone(),
                    static_cast<char const*>(this->vertPtr) + idx * this->DataStride[this->vrtDT] * 3},
                    ColorData_Base{this->colorAccessor->Clone(),
                    static_cast<char const*>(this->colPtr) + idx * this->ColorDataStride[this->colDT]}
                };
            }

            /**
             * Access index array at index without range check.
             *
             * @param idx Index of element in index array.
             *
             * @return Struct of pointer to position in index array.
             */
            index_t GetIdx(size_t idx) const noexcept {
                return index_t{
                    IndexData_Base{this->indexAccessor->Clone(),
                    static_cast<char const*>(this->idxPtr) + idx * this->DataStride[this->idxDT]}
                };
            }

        private:

            /**
             * Sets the colour data
             *
             * @param the data pointer
             * @param withAlpha Flag if data contains alpha information
             */
            inline void setColData(unsigned char *data, bool withAlpha) {
                if (withAlpha) {
                    this->colDT = CDT_BYTE_RGBA;
                    this->colorAccessor.reset(new ColorData_Impl<uint8_t, true>{ });
                } else {
                    this->colDT = CDT_BYTE_RGB;
                    this->colorAccessor.reset(new ColorData_Impl<uint8_t, false>{ });
                }
                this->col.dataByte = data;
                this->colPtr = data;
            }

            /**
             * Sets the colour data
             *
             * @param the data pointer
             * @param withAlpha Flag if data contains alpha information
             */
            inline void setColData(const unsigned char *data, bool withAlpha) {
                if (withAlpha) {
                    this->colDT = CDT_BYTE_RGBA;
                    this->colorAccessor.reset(new ColorData_Impl<uint8_t, true>{ });
                } else {
                    this->colDT = CDT_BYTE_RGB;
                    this->colorAccessor.reset(new ColorData_Impl<uint8_t, false>{ });
                }
                this->col.dataByte = data;
                this->colPtr = data;
            }

            /**
             * Sets the colour data
             *
             * @param the data pointer
             * @param withAlpha Flag if data contains alpha information
             */
            inline void setColData(float *data, bool withAlpha) {
                if (withAlpha) {
                    this->colDT = CDT_FLOAT_RGBA;
                    this->colorAccessor.reset(new ColorData_Impl<float, true>{ });
                } else {
                    this->colDT = CDT_FLOAT_RGB;
                    this->colorAccessor.reset(new ColorData_Impl<float, false>{ });
                }
                this->col.dataFloat = data;
                this->colPtr = data;
            }

            /**
             * Sets the colour data
             *
             * @param the data pointer
             * @param withAlpha Flag if data contains alpha information
             */
            inline void setColData(const float *data, bool withAlpha) {
                if (withAlpha) {
                    this->colDT = CDT_FLOAT_RGBA;
                    this->colorAccessor.reset(new ColorData_Impl<float, true>{ });
                } else {
                    this->colDT = CDT_FLOAT_RGB;
                    this->colorAccessor.reset(new ColorData_Impl<float, false>{ });
                }
                this->col.dataFloat = data;
                this->colPtr = data;
            }

            /**
             * Sets the colour data
             *
             * @param the data pointer
             * @param withAlpha Flag if data contains alpha information
             */
            inline void setColData(double *data, bool withAlpha) {
                if (withAlpha) {
                    this->colDT = CDT_DOUBLE_RGBA;
                    this->colorAccessor.reset(new ColorData_Impl<double, true>{ });
                } else {
                    this->colDT = CDT_DOUBLE_RGB;
                    this->colorAccessor.reset(new ColorData_Impl<double, false>{ });
                }
                this->col.dataDouble = data;
                this->colPtr = data;
            }

            /**
             * Sets the colour data
             *
             * @param the data pointer
             * @param withAlpha Flag if data contains alpha information
             */
            inline void setColData(const double *data, bool withAlpha) {
                if (withAlpha) {
                    this->colDT = CDT_DOUBLE_RGBA;
                    this->colorAccessor.reset(new ColorData_Impl<double, true>{ });
                } else {
                    this->colDT = CDT_DOUBLE_RGB;
                    this->colorAccessor.reset(new ColorData_Impl<double, false>{ });
                }
                this->col.dataDouble = data;
                this->colPtr = data;
            }

            /**
             * Sets the colour data
             *
             * @param the data pointer
             * @param withAlpha Flag if data contains alpha information
             */
            template<class Tp>
            inline void setColData(Tp data, bool withAlpha) {
                ASSERT(data == nullptr);
                this->colDT = CDT_NONE;
                this->col.dataByte = nullptr;
                this->colorAccessor.reset(new ColorData_None{ });
                this->colPtr = nullptr;
            }

            /**
             * Sets the index data
             *
             * @param the data pointer
             */
            inline void setIdxData(unsigned char *data) {
                this->idxDT = DT_BYTE;
                this->idx.dataByte = data;
                this->indexAccessor.reset(new IndexData_Impl<uint8_t>{ });
                this->idxPtr = data;
            }

            /**
             * Sets the index data
             *
             * @param the data pointer
             */
            inline void setIdxData(const unsigned char *data) {
                this->idxDT = DT_BYTE;
                this->idx.dataByte = data;
                this->indexAccessor.reset(new IndexData_Impl<uint8_t>{ });
                this->idxPtr = data;
            }

            /**
             * Sets the index data
             *
             * @param the data pointer
             */
            inline void setIdxData(unsigned short *data) {
                this->idxDT = DT_UINT16;
                this->idx.dataUInt16 = data;
                this->indexAccessor.reset(new IndexData_Impl<uint16_t>{ });
                this->idxPtr = data;
            }

            /**
             * Sets the index data
             *
             * @param the data pointer
             */
            inline void setIdxData(const unsigned short *data) {
                this->idxDT = DT_UINT16;
                this->idx.dataUInt16 = data;
                this->indexAccessor.reset(new IndexData_Impl<uint16_t>{ });
                this->idxPtr = data;
            }

            /**
             * Sets the index data
             *
             * @param the data pointer
             */
            inline void setIdxData(unsigned int *data) {
                this->idxDT = DT_UINT32;
                this->idx.dataUInt32 = data;
                this->indexAccessor.reset(new IndexData_Impl<uint32_t>{ });
                this->idxPtr = data;
            }

            /**
             * Sets the index data
             *
             * @param the data pointer
             */
            inline void setIdxData(const unsigned int *data) {
                this->idxDT = DT_UINT32;
                this->idx.dataUInt32 = data;
                this->indexAccessor.reset(new IndexData_Impl<uint8_t>{ });
                this->idxPtr = data;
            }

            /**
             * Sets the index data
             *
             * @param the data pointer
             */
            template<class Tp>
            inline void setIdxData(const Tp data) {
                ASSERT(data == nullptr);
                this->idxDT = DT_NONE;
                this->idx.dataUInt32 = nullptr;
                this->indexAccessor.reset(new IndexData_None{ });
                this->idxPtr = nullptr;
            }

            /**
             * Sets the index data
             *
             * @param the data pointer
             */
            inline void setVrtData(float *data) {
                this->vrtDT = DT_FLOAT;
                this->vrt.dataFloat = data;
                this->vertexAccessor.reset(new VertexData_Impl<float>{});
                this->vertPtr = data;
            }

            /**
             * Sets the index data
             *
             * @param the data pointer
             */
            inline void setVrtData(const float *data) {
                this->vrtDT = DT_FLOAT;
                this->vrt.dataFloat = data;
                this->vertexAccessor.reset(new VertexData_Impl<float>{ });
                this->vertPtr = data;
            }

            /**
             * Sets the index data
             *
             * @param the data pointer
             */
            inline void setVrtData(double *data) {
                this->vrtDT = DT_DOUBLE;
                this->vrt.dataDouble = data;
                this->vertexAccessor.reset(new VertexData_Impl<double>{ });
                this->vertPtr = data;
            }

            /**
             * Sets the index data
             *
             * @param the data pointer
             */
            inline void setVrtData(const double *data) {
                this->vrtDT = DT_DOUBLE;
                this->vrt.dataDouble = data;
                this->vertexAccessor.reset(new VertexData_Impl<double>{ });
                this->vertPtr = data;
            }

            /**
             * Sets the index data
             *
             * @param the data pointer
             */
            template<class Tp>
            inline void setVrtData(Tp data) {
                ASSERT(data == nullptr);
                this->vrtDT = DT_NONE;
                this->vrt.dataDouble = nullptr;
                this->vertexAccessor.reset(new VertexData_None{ });
                this->vertPtr = nullptr;
            }

            /** The colour data type */
            ColourDataType colDT;

            /** The colour array */
            union _col_t {
                const unsigned char *dataByte;
                const float *dataFloat;
                const double *dataDouble;
            } col;

            /** The number of elements */
            unsigned int count;

#ifdef _WIN32
#pragma warning (disable: 4251)
#endif /* _WIN32 */
            /** The global colour */
            vislib::graphics::ColourRGBAu8 globCol;
#ifdef _WIN32
#pragma warning (default: 4251)
#endif /* _WIN32 */

            /** The index array data type */
            DataType idxDT;

            /** The index array (1xunsigned int*) */
            union _idx_t {
                const unsigned char *dataByte;
                const unsigned short *dataUInt16;
                const unsigned int *dataUInt32;
            } idx;

            /** The vertex array data type */
            DataType vrtDT;

            /** The vertex array (XYZ-Float*) */
            union _vrt_t {
                const float *dataFloat;
                const double *dataDouble;
            } vrt;

            /** The line ID */
            size_t id;

            /** Polymorphic vertex data accessor */
            std::unique_ptr<VertexData_Detail> vertexAccessor;

            /** Polymorphic color data accessor */
            std::unique_ptr<ColorData_Detail> colorAccessor;

            /** Polymorphic index data accessor */
            std::unique_ptr<IndexData_Detail> indexAccessor;

            /** Vertex data pointer for dispatch */
            void const* vertPtr;

            /** Color data pointer for dispatch */
            void const* colPtr;

            /** Index data pointer for dispatch */
            void const* idxPtr;

        };

        /**
         * Answer the name of the objects of this description.
         *
         * @return The name of the objects of this description.
         */
        static const char *ClassName(void) {
            return "LinesDataCall";
        }

        /**
         * Gets a human readable description of the module.
         *
         * @return A human readable description of the module.
         */
        static const char *Description(void) {
            return "Call to get lines data";
        }

        /**
         * Answer the number of functions used for this call.
         *
         * @return The number of functions used for this call.
         */
        static unsigned int FunctionCount(void) {
            return core::AbstractGetData3DCall::FunctionCount();
        }

        /**
         * Answer the name of the function used for this call.
         *
         * @param idx The index of the function to return it's name.
         *
         * @return The name of the requested function.
         */
        static const char * FunctionName(unsigned int idx) {
            return core::AbstractGetData3DCall::FunctionName(idx);
        }

        /** Ctor. */
        LinesDataCall(void);

        /** Dtor. */
        virtual ~LinesDataCall(void);

        /**
         * Answer the size of the lines lists
         *
         * @return The size of the lines lists
         */
        VISLIB_FORCEINLINE unsigned int Count(void) const {
            return this->count;
        }

        /**
         * Answer the lines list. Might be NULL! Do not delete the returned
         * memory.
         *
         * @return The lines list
         */
        VISLIB_FORCEINLINE const Lines* GetLines(void) const {
            return this->lines;
        }

        /**
         * Sets the data. The object will not take ownership of the memory
         * 'lines' points to. The caller is responsible for keeping the data
         * valid as long as it is used.
         *
         * @param count The number of lines stored in 'lines'
         * @param lines Pointer to a flat array of lines.
         * @param time The point in time for which these lines are meant.
         */
        void SetData(unsigned int count, const Lines *lines, const float time = 0.0f);

        /**
        * Sets the time the lines are called for.
        *
        * @param time The new time value.
        */
        VISLIB_FORCEINLINE void SetTime(const float time) {
            this->time = time;
        }

        /**
        * Answers the time the lines are called for.
        *
        * @return The time for which the lines are needed.
        */
        VISLIB_FORCEINLINE const float Time(void) const {
            return this->time;
        }


        /**
         * Assignment operator.
         * Makes a deep copy of all members. While for data these are only
         * pointers, the pointer to the unlocker object is also copied.
         *
         * @param rhs The right hand side operand
         *
         * @return A reference to this
         */
        LinesDataCall& operator=(const LinesDataCall& rhs);

    private:

        /** The call time. */
        float time;

        /** Number of curves */
        unsigned int count;

        /** Cubic bézier curves */
        const Lines *lines;

    };

    /** Description class typedef */
    typedef megamol::core::factories::CallAutoDescription<LinesDataCall> LinesDataCallDescription;


} /* end namespace geocalls */
} /* end namespace megamol */

#endif /* MEGAMOL_GEOMETRY_CALLS_LINESDATACALL_H_INCLUDED */
