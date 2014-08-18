/*
 * testvector.cpp
 *
 * Copyright (C) 2007 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#include "testhelper.h"

#include "vislib/Vector.h"
#include "vislib/Array.h"
#include "vislib/FastMap.h"
#include "vislib/ForceDirected.h"

#define OUTDIMS 1

class myObject {
public:
    float x,y;
    float z;
    float Distance(myObject &other) {
        float tmp = (this->x - other.x) * (this->x - other.x);
        tmp += (this->y - other.y) * (this->y - other.y);
        //tmp += (this->z - other.z) * (this->z - other.z);
        return sqrt(tmp);
    }
    inline bool operator ==(const myObject &rhs) {
        //return this->x == rhs.x && this->y == rhs.y && this->z == rhs.z;
        return this->x == rhs.x && this->y == rhs.y;
    }

    float Weight(myObject &other) {
        return 1.0;
    }

    float TotalWeight(void) {
        return 1.0;
    }
};

typedef vislib::Array<myObject> sourceType;
typedef vislib::Array<vislib::math::Vector<float, OUTDIMS> > destType;

void TestFastMap(void) {
    sourceType source;
    myObject a, b, c;
    a.x = 1.0;
    a.y = 1.0;
    //a.z = 2.0;

    b.x = 4.0;
    b.y = 1.0;
    //b.z = 7.0;

    c.x = 3.0;
    c.y = 2.0;
    //c.z = 6.0;

    assert(source.Count() == 0);

    source.Add(a);
    source.Add(b);
    source.Add(c);

    assert(source.Count() == 3);

    destType destination(source.Count());
    destination.SetCount(source.Count());

    vislib::math::FastMap<myObject, float, OUTDIMS> fm(source, destination);

    for (SIZE_T i = 0; i < source.Count(); i++) {
        printf("item %u: (", static_cast<unsigned int>(i));
        for (unsigned int j = 0; j < OUTDIMS; j++) {
            printf("%f", destination[i][j]);
            if (j != OUTDIMS - 1) {
                printf(", ");
            }
        }
        printf(")\n");
    }

}

class myOtherObject {
public:
    float x;

    float Distance(myObject &other) {
        float tmp = (this->x - other.x) * (this->x - other.x);
        //tmp += (this->y - other.y) * (this->y - other.y);
        //tmp += (this->z - other.z) * (this->z - other.z);
        return sqrt(tmp);
    }

    inline bool operator ==(const myOtherObject &rhs) {
        //return this->x == rhs.x && this->y == rhs.y && this->z == rhs.z;
        //return this->x == rhs.x && this->y == rhs.y;
        return this->x == rhs.x;
    }

    float Weight(myOtherObject &other) {
        return 1.0;
    }

    float TotalWeight(void) {
        return 1.0;
    }
};

typedef vislib::Array<myOtherObject> otherSourceType;
typedef vislib::Array<vislib::math::Vector<float, 1> > otherDestType;

void TestForceDirected(void) {
    otherSourceType source;
    myOtherObject a, b, c;
    a.x = 1.0;
    //a.y = 1.0;
    //a.z = 2.0;

    b.x = 4.0;
    //b.y = 1.0;
    //b.z = 7.0;

    c.x = 3.0;
    //c.y = 2.0;
    //c.z = 6.0;

    assert(source.Count() == 0);

    source.Add(a);
    source.Add(b);
    source.Add(c);

    assert(source.Count() == 3);

    otherDestType destination(source.Count());
    destination.SetCount(source.Count());

    vislib::math::ForceDirected<myOtherObject, float, 1> fm(source, destination, 20, 0.1f, 3, 1, 0.0001f, 0.9f);

    for (SIZE_T loop = 0; loop < 20; loop++) {
        fm.SingleStep();
        for (SIZE_T i = 0; i < source.Count(); i++) {
            printf("item %u: (", static_cast<unsigned int>(i));
            for (unsigned int j = 0; j < 1; j++) {
                printf("%f", destination[i][j]);
                if (j != 1 - 1) {
                    printf(", ");
                }
            }
            printf(")\n");
        }
    }
}
