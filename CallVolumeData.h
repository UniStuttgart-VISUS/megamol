/*
 * CallVolumeData.h
 *
 * Copyright (C) 2011 by Universitaet Stuttgart (VISUS). 
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_CALLVOLUMEDATA_H_INCLUDED
#define MEGAMOLCORE_CALLVOLUMEDATA_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "api/MegaMolCore.std.h"
#include <climits>
#include "AbstractGetData3DCall.h"
//#include "CallAutoDescription.h"
#include "vislib/Array.h"
#include "vislib/assert.h"
#include "vislib/String.h"


namespace megamol {
namespace core {


    /**
     * Class holding all data of a volume data set
     */
    class MEGAMOLCORE_API CallVolumeData : public AbstractGetData3DCall {
    public:
    
        /**
         * Answer the name of the objects of this description.
         *
         * @return The name of the objects of this description.
         */
        static const char *ClassName(void) {
            return "CallVolumeData";
        }

        /**
         * Gets a human readable description of the module.
         *
         * @return A human readable description of the module.
         */
        static const char *Description(void) {
            return "Call to get data of a volume data set";
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

        /** Possible data types */
        enum DataType {
            TYPE_BYTE,
            TYPE_FLOAT,
            TYPE_DOUBLE
        };

        /**
         * Attribute data
         */
        class Data {
        public:

            /**
             * Ctor
             *
             * @param name The attribute name
             * @param type The attribute data type
             * @param data The attribute data pointer
             */
            Data(const char* name = "", DataType type = TYPE_FLOAT, void* data = NULL)
                    : name(name), type(type), data(data) {
                // intentionally empty
            }

            /**
             * Copy Ctor
             *
             * @param src The object to clone from
             */
            Data(const Data& src) : name(src.name), type(src.type), data(src.data) {
                // intentionally empty
            }

            /** Dtor */
            ~Data(void) {
                this->data = NULL; // DO NOT DELETE
            }

            /**
             * Gets the bytes data pointer
             *
             * @return The bytes data pointer
             */
            inline const BYTE* Bytes(void) const {
                return static_cast<const BYTE*>(this->data);
            }

            /**
             * Gets the doubles data pointer
             *
             * @return The doubles data pointer
             */
            inline const double* Doubles(void) const {
                return static_cast<const double*>(this->data);
            }

            /**
             * Gets the floats data pointer
             *
             * @return The floats data pointer
             */
            inline const float* Floats(void) const {
                return static_cast<const float*>(this->data);
            }

            /**
             * Gets the name of the attribute
             *
             * @return The name of the attribute
             */
            inline const vislib::StringA& Name(void) const {
                return this->name;
            }

            /**
             * Gets the data pointer
             *
             * @return The data pointer
             */
            inline const void* RawData(void) const {
                return this->data;
            }

            /**
             * Sets the data pointer
             *
             * @param data The new data pointer
             */
            inline void SetData(const void *data) {
                this->data = data;
            }

            /**
             * Sets the name of the attribute
             *
             * @param name The new name
             */
            inline void SetName(const vislib::StringA& name) {
                this->name = name;
            }

            /**
             * Sets the data type
             *
             * @param type The new data type
             */
            inline void SetType(DataType type) {
                this->type = type;
            }

            /**
             * Gets the data type
             *
             * @return The data type
             */
            inline DataType Type(void) const {
                return this->type;
            }

            /**
             * Assignment operator
             *
             * @param rhs The right hand side operand
             *
             * @return A reference to this
             */
            Data& operator=(const Data& rhs) {
                this->name = rhs.name;
                this->type = rhs.type;
                this->data = rhs.data;
                return *this;
            }

            /**
             * Test for equality
             *
             * @param rhs The right hand side operand
             *
             * @return True if this and rhs are equal
             */
            bool operator==(const Data& rhs) const {
                return this->name.Equals(rhs.name)
                    && (this->type == rhs.type)
                    && (this->data == rhs.data);
            }

        private:

            /** The name of the attribute */
            vislib::StringA name;

            /** The attribute's data type */
            DataType type;

            /** The data itself (pointer only. Memory is not owned by this object!) */
            const void* data;

        };

        /** Ctor */
        CallVolumeData(void);

        /** Dtor */
        virtual ~CallVolumeData(void);

        /**
         * Gets the idx-th attribute
         *
         * @param idx The zero-based index
         *
         * @return The requested attribute
         */
        inline Data& Attribute(unsigned int idx) {
            return this->attributes[idx];
        }

        /**
         * Gets the idx-th attribute
         *
         * @param idx The zero-based index
         *
         * @return The requested attribute
         */
        inline const Data& Attribute(unsigned int idx) const {
            return this->attributes[idx];
        }

        /**
         * Returns the number of attributes
         *
         * @return The number of attributes
         */
        inline unsigned int AttributeCount(void) const {
            return static_cast<unsigned int>(this->attributes.Count());
        }

        /**
         * Gets the attribute index of the first attribute with the name 'name'
         *
         * @param name The name of the attribute to search for
         *
         * @return The index of the first attribute with name 'name' or UNIT_MAX if no attribute with this name exists.
         */
        inline unsigned int FindAttribute(const vislib::StringA& name) const {
            for (SIZE_T i = 0; i < this->attributes.Count(); i++) {
                if (this->attributes[i].Name().Equals(name)) {
                    return static_cast<unsigned int>(i);
                }
            }
            return UINT_MAX;
        }

        /**
         * Sets the number of samples in all directions
         *
         * @param x The number of samples in x direction
         * @param y The number of samples in y direction
         * @param z The number of samples in z direction
         */
        inline void SetSize(unsigned int x, unsigned int y, unsigned int z) {
            this->xSize = x;
            this->ySize = y;
            this->zSize = z;
        }

        /**
         * Sets the number of attributes
         *
         * @param cnt The new number of attributes
         */
        inline void SetAttributeCount(unsigned int cnt) {
            this->attributes.SetCount(cnt);
        }

        /**
         * Gets the number of samples in x direction
         *
         * @return The number of samples in x direction
         */
        inline unsigned int XSize(void) const {
            return this->xSize;
        }

        /**
         * Gets the number of samples in y direction
         *
         * @return The number of samples in y direction
         */
        inline unsigned int YSize(void) const {
            return this->ySize;
        }

        /**
         * Gets the number of samples in z direction
         *
         * @return The number of samples in z direction
         */
        inline unsigned int ZSize(void) const {
            return this->zSize;
        }

        /**
         * Assignment operator
         *
         * @param rhs The right-hand side operand
         *
         * @return A reference to this
         */
        CallVolumeData& operator=(const CallVolumeData& rhs);

    private:

        /** The number of samples in x direction */
        unsigned int xSize;

        /** The number of samples in y direction */
        unsigned int ySize;

        /** The number of samples in z direction */
        unsigned int zSize;

#ifdef _WIN32
#pragma warning(disable:4251)
#endif /* _WIN32 */

        /** The array of attributes */
        vislib::Array<Data> attributes;

#ifdef _WIN32
#pragma warning(default:4251)
#endif /* _WIN32 */

    };

    /** Description class typedef */
    typedef CallAutoDescription<CallVolumeData>
        CallVolumeDataDescription;

} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_CALLVOLUMEDATA_H_INCLUDED */
