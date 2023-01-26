/*
 * MMSPDHeader.h
 *
 * Copyright (C) 2011-2015 by MegaMol Team
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_MMSPDHEADER_H_INCLUDED
#define MEGAMOLCORE_MMSPDHEADER_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "vislib/Array.h"
#include "vislib/String.h"
#include "vislib/math/Cuboid.h"
#include "vislib/types.h"


namespace megamol {
namespace moldyn {
namespace io {


/**
 * MMSPD file header data
 */
class MMSPDHeader {
public:
    /**
     * Class for particle definition data fields
     */
    class Field {
    public:
        /** Possible type identifiers */
        enum TypeID { TYPE_BYTE, TYPE_FLOAT, TYPE_DOUBLE };

        /** Ctor */
        Field();

        /** Dtor */
        ~Field();

        /**
         * Gets the name of the field
         *
         * @return The name of the field
         */
        inline const vislib::StringA& GetName() const {
            return this->name;
        }

        /**
         * Gets the type of the field
         *
         * @return The type of the field
         */
        inline TypeID GetType() const {
            return this->type;
        }

        /**
         * Sets the name for the field
         *
         * @param name The new name for the field
         */
        inline void SetName(const vislib::StringA& name) {
            this->name = name;
        }

        /**
         * Sets the type for the field
         *
         * @param type The new type for the field
         */
        inline void SetType(TypeID type) {
            this->type = type;
        }

        /**
         * Test for equality
         *
         * @param rhs The right-hand side operand
         *
         * @return True if 'this' and 'rhs' are equal
         */
        bool operator==(const Field& rhs) const;

    private:
#ifdef _WIN32
#pragma warning(disable : 4251)
#endif /* _WIN32 */
        /** The name of the field */
        vislib::StringA name;
#ifdef _WIN32
#pragma warning(default : 4251)
#endif /* _WIN32 */

        /** The type of the field */
        TypeID type;
    };

    /**
     * Class for const particle definition data fields
     */
    class ConstField : public Field {
    public:
        /** Ctor */
        ConstField();

        /** Dtor */
        ~ConstField();

        /**
         * Gets the value of the field casted as float
         *
         * @return the value of the field casted as float
         */
        inline float GetAsFloat() const {
            switch (this->GetType()) {
            case TYPE_BYTE:
                return static_cast<float>(this->data.valByte) / 255.0f;
            case TYPE_FLOAT:
                return this->data.valFloat;
            case TYPE_DOUBLE:
                return static_cast<float>(this->data.valDouble);
            }
            return 0.0f;
        }

        /**
         * Gets the value of the field as byte.
         * The caller must ensure the compatibility of the field type setting.
         *
         * @return The value of the field as byte
         */
        inline BYTE GetByte() const {
            return this->data.valByte;
        }

        /**
         * Gets the value of the field as float.
         * The caller must ensure the compatibility of the field type setting.
         *
         * @return The value of the field as float
         */
        inline float GetFloat() const {
            return this->data.valFloat;
        }

        /**
         * Gets the value of the field as double.
         * The caller must ensure the compatibility of the field type setting.
         *
         * @return The value of the field as double
         */
        inline double GetDouble() const {
            return this->data.valDouble;
        }

        /**
         * Sets the byte value of the field.
         * The caller must ensure the compatibility of the field type setting.
         *
         * @param val The new value for the field
         */
        inline void SetByte(BYTE val) {
            this->data.valByte = val;
        }

        /**
         * Sets the float value of the field
         * The caller must ensure the compatibility of the field type setting.
         *
         * @param val The new value for the field
         */
        inline void SetFloat(float val) {
            this->data.valFloat = val;
        }

        /**
         * Sets the double value of the field
         * The caller must ensure the compatibility of the field type setting.
         *
         * @param val The new value for the field
         */
        inline void SetDouble(double val) {
            this->data.valDouble = val;
        }

        /**
         * Test for equality
         *
         * @param rhs The right-hand side operand
         *
         * @return True if 'this' and 'rhs' are equal
         */
        bool operator==(const ConstField& rhs) const;

    private:
        /** The data of the const field */
        union data_t {
            BYTE valByte;
            float valFloat;
            double valDouble;
        } data;
    };

    /**
     * Class of particle type definitions
     */
    class TypeDefinition {
    public:
        /** Ctor */
        TypeDefinition();

        /** Dtor */
        ~TypeDefinition();

        /**
         * Accesses the array of constant fields
         *
         * @return The array of constant fields
         */
        inline vislib::Array<ConstField>& ConstFields() {
            return this->constFields;
        }

        /**
         * Accesses the array of fields
         *
         * @return The array of fields
         */
        inline vislib::Array<Field>& Fields() {
            return this->fields;
        }

        /**
         * Gets the base type
         *
         * @return The base type
         */
        inline const vislib::StringA& GetBaseType() const {
            return this->baseType;
        }

        /**
         * Gets the array of constant fields
         *
         * @return The array of constant fields
         */
        inline const vislib::Array<ConstField>& GetConstFields() const {
            return this->constFields;
        }

        /**
         * Answer the size of the variable data fields of particles of this size.
         * Add 8 if hasIDs is true.
         * Add 4 if typeCount is larger than 1.
         *
         * @return The size of the variable data fields
         */
        inline unsigned int GetDataSize() const {
            unsigned int s = 0;
            for (SIZE_T i = 0; i < this->fields.Count(); i++) {
                switch (this->fields[i].GetType()) {
                case Field::TYPE_BYTE:
                    s += 1;
                    break;
                case Field::TYPE_FLOAT:
                    s += 4;
                    break;
                case Field::TYPE_DOUBLE:
                    s += 8;
                    break;
                }
            }
            return s;
        }

        /**
         * Gets the array of fields
         *
         * @return The array of fields
         */
        inline const vislib::Array<Field>& GetFields() const {
            return this->fields;
        }

        /**
         * Sets the base type
         *
         * @param basetype The new base type
         */
        inline void SetBaseType(const vislib::StringA& basetype) {
            this->baseType = basetype;
        }

        /**
         * Test for equality
         *
         * @param rhs The right-hand side operand
         *
         * @return True if 'this' and 'rhs' are equal
         */
        bool operator==(const TypeDefinition& rhs) const;

    private:
#ifdef _WIN32
#pragma warning(disable : 4251)
#endif /* _WIN32 */
        /** The base type identifier */
        vislib::StringA baseType;
#ifdef _WIN32
#pragma warning(default : 4251)
#endif /* _WIN32 */

#ifdef _WIN32
#pragma warning(disable : 4251)
#endif /* _WIN32 */
        /** The constant data fields */
        vislib::Array<ConstField> constFields;
#ifdef _WIN32
#pragma warning(default : 4251)
#endif /* _WIN32 */

#ifdef _WIN32
#pragma warning(disable : 4251)
#endif /* _WIN32 */
        /** The variable data fields */
        vislib::Array<Field> fields;
#ifdef _WIN32
#pragma warning(default : 4251)
#endif /* _WIN32 */
    };

    /** Ctor */
    MMSPDHeader();

    /** Dtor */
    virtual ~MMSPDHeader();

    /**
     * Accesses the bounding box
     *
     * @return The bounding box
     */
    inline vislib::math::Cuboid<double>& BoundingBox() {
        return this->bbox;
    }

    /**
     * Gets the bounding box
     *
     * @return The bounding box
     */
    inline const vislib::math::Cuboid<double>& GetBoundingBox() const {
        return this->bbox;
    }

    /**
     * Gets the number of particles
     *
     * @return The number of particles
     */
    inline UINT64 GetParticleCount() const {
        return this->particleCount;
    }

    /**
     * Gets the number of time frames
     *
     * @return The number of time frames
     */
    inline UINT32 GetTimeCount() const {
        return this->timeCount;
    }

    /**
     * Gets the particle type definitions
     *
     * @return The particle type definitions
     */
    inline const vislib::Array<TypeDefinition>& GetTypes() const {
        return this->types;
    }

    /**
     * Gets the flag whether or not particle IDs are explicitly stored
     *
     * @return True if particle IDs are explicitly stored
     */
    inline bool HasIDs() const {
        return this->hasIDs;
    }

    /**
     * Sets the bounding box
     *
     * @param bbox The new bounding box
     */
    inline void SetBoundingBox(const vislib::math::Cuboid<double>& bbox) {
        this->bbox = bbox;
    }

    /**
     * Sets the flag whether or not particle IDs are explicitly stored
     *
     * @param hasIDs True if the particles IDs are explicitly stored
     */
    inline void SetHasIDs(bool hasIDs) {
        this->hasIDs = hasIDs;
    }

    /**
     * Sets the number of particles
     *
     * @param cnt The number of particles
     */
    inline void SetParticleCount(UINT64 cnt) {
        this->particleCount = cnt;
    }

    /**
     * Sets the number of time frames
     *
     * @param cnt The number of time frames
     */
    inline void SetTimeCount(UINT32 cnt) {
        this->timeCount = cnt;
    }

    /**
     * Accesses the particle type definitions
     *
     * @return The particle type definitions
     */
    inline vislib::Array<TypeDefinition>& Types() {
        return this->types;
    }

private:
    /** Flag whether or not particle IDs are explicitly stored */
    bool hasIDs;

#ifdef _WIN32
#pragma warning(disable : 4251)
#endif /* _WIN32 */
    /** The particles bounding box */
    vislib::math::Cuboid<double> bbox;
#ifdef _WIN32
#pragma warning(default : 4251)
#endif /* _WIN32 */

    /** The number of time frames */
    UINT32 timeCount;

#ifdef _WIN32
#pragma warning(disable : 4251)
#endif /* _WIN32 */
    /** The defined particle types */
    vislib::Array<TypeDefinition> types;
#ifdef _WIN32
#pragma warning(default : 4251)
#endif /* _WIN32 */

    /** The number of particles */
    UINT64 particleCount;
};


} /* end namespace io */
} /* end namespace moldyn */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_MMSPDHEADER_H_INCLUDED */
