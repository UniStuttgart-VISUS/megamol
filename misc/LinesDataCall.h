/*
 * LinesDataCall.h
 *
 * Copyright (C) 2010 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_LINESDATACALL_H_INCLUDED
#define MEGAMOLCORE_LINESDATACALL_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "api/MegaMolCore.std.h"
#include "AbstractGetData3DCall.h"
#include "CallAutoDescription.h"
#include "vislib/assert.h"
#include "vislib/ColourRGBAu8.h"
#include "vislib/forceinline.h"


namespace megamol {
namespace core {
namespace misc {


    /**
     * Call for lines data
     */
    class MEGAMOLCORE_API LinesDataCall : public AbstractGetData3DCall {
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
        class MEGAMOLCORE_API Lines {
        public:

            /**
             * Ctor
             */
            Lines(void);

            /**
             * Dtor
             */
            ~Lines(void);

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
             * Answer the colour array. This can be NULL if the global colour
             * data should be used
             *
             * @return The colour array
             */
            inline const void *ColourArray(void) const {
                return this->colArray;
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
             * Answer whether or not the colour data contains an alpha channel
             *
             * @return True if the colour data is RGBA, False if the colour
             *         data is RGB
             */
            inline bool HasColourAlpha(void) const {
                return this->colHasAlpha;
            }

            /**
             * Answer the index array. This can be NULL.
             *
             * @return The index array
             */
            inline const unsigned int *IndexArray(void) const {
                return this->idxArray;
            }

            /**
             * Answer whether or not the colour data is stored as floats.
             *
             * @return True if the colour data is stored as 'float*', false if
             *         the colour data is stored as 'unsigned char*'.
             */
            inline bool IsFloatColour(void) const {
                return this->useFloatCol;
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
            inline void Set(unsigned int cnt, const float *vert,
                    vislib::graphics::ColourRGBAu8 col) {
                ASSERT(vert != NULL);
                this->count = cnt;
                this->vertArray = vert;
                this->colArray = NULL;
                this->idxArray = NULL;
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
            inline void Set(unsigned int cnt, const unsigned int *idx,
                    const float *vert, vislib::graphics::ColourRGBAu8 col) {
                ASSERT(idx != NULL);
                ASSERT(vert != NULL);
                this->count = cnt;
                this->vertArray = vert;
                this->colArray = NULL;
                this->idxArray = idx;
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
            inline void Set(unsigned int cnt, const float *vert,
                    const float *col, bool withAlpha) {
                ASSERT(vert != NULL);
                ASSERT(col != NULL);
                this->count = cnt;
                this->vertArray = vert;
                this->colArray = static_cast<const void*>(col);
                this->useFloatCol = true;
                this->colHasAlpha = withAlpha;
                this->idxArray = NULL;
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
            inline void Set(unsigned int cnt, const unsigned int *idx,
                    const float *vert, const float *col, bool withAlpha) {
                ASSERT(idx != NULL);
                ASSERT(vert != NULL);
                ASSERT(col != NULL);
                this->count = cnt;
                this->vertArray = vert;
                this->colArray = static_cast<const void *>(col);
                this->useFloatCol = true;
                this->colHasAlpha = withAlpha;
                this->idxArray = idx;
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
             * @param vert The vertex array (XYZ-Float)
             * @param col The colour array (use same number of entries as the
             *            vertex array)
             * @param withAlpha Flag if the colour array contains RGBA(true)
             *                  or RGB(false) values
             */
            inline void Set(unsigned int cnt, const float *vert,
                    const unsigned char *col, bool withAlpha) {
                ASSERT(vert != NULL);
                ASSERT(col != NULL);
                this->count = cnt;
                this->vertArray = vert;
                this->colArray = static_cast<const void *>(col);
                this->useFloatCol = false;
                this->colHasAlpha = withAlpha;
                this->idxArray = NULL;
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
            inline void Set(unsigned int cnt, const unsigned int *idx,
                    const float *vert, const unsigned char *col,
                    bool withAlpha) {
                ASSERT(idx != NULL);
                ASSERT(vert != NULL);
                ASSERT(col != NULL);
                this->count = cnt;
                this->vertArray = vert;
                this->colArray = static_cast<const void*>(col);
                this->useFloatCol = false;
                this->colHasAlpha = withAlpha;
                this->idxArray = idx;
                this->globCol.Set(0, 0, 0, 255);
            }

            /**
             * Answer the vertex array (XYZ-Float)
             *
             * @return The vertex array
             */
            inline const float *VertexArray(void) const {
                return this->vertArray;
            }

        private:

            /** The colour array */
            const void *colArray;

            /** Flag if the colour data contains an alpha channel */
            bool colHasAlpha;

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

            /** The index array (1xunsigned int*) */
            const unsigned int *idxArray;

            /** Flag if the colour data is stored as float array */
            bool useFloatCol;

            /** The vertex array (XYZ-Float*) */
            const float *vertArray;

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
            return AbstractGetData3DCall::FunctionCount();
        }

        /**
         * Answer the name of the function used for this call.
         *
         * @param idx The index of the function to return it's name.
         *
         * @return The name of the requested function.
         */
        static const char * FunctionName(unsigned int idx) {
            return AbstractGetData3DCall::FunctionName(idx);
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
         */
        void SetData(unsigned int count, const Lines *lines);

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

        /** Number of curves */
        unsigned int count;

        /** Cubic bézier curves */
        const Lines *lines;

    };

    /** Description class typedef */
    typedef CallAutoDescription<LinesDataCall> LinesDataCallDescription;


} /* end namespace misc */
} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_LINESDATACALL_H_INCLUDED */
