/*
 * testmultisz.cpp
 *
 * Copyright (C) 2006 - 2007 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#include "testmultisz.h"

#include <iostream>

#include "vislib/MultiSz.h"

#include "testhelper.h"


void TestMultiSz(void) {
    using namespace std;
    using namespace vislib;

    MultiSzA multiSz;

    AssertTrue("MultiSz is intially empty.", multiSz.IsEmpty());
    AssertEqual("Empty MultiSz has zero entries.", multiSz.Count(), SIZE_T(0));
    AssertEqual("Empty MultiSz has length zero.", multiSz.Length(), SIZE_T(0));
    multiSz.Clear();
    AssertEqual("Clearing empty MultiSz has no effect.", multiSz.Count(), SIZE_T(0));

    const char *strings[] = { "Horst", "Hugo", "Heinz", "Hans" };
    MultiSzA multiSz2(strings, 4);
    AssertFalse("MultiSz is not empty.", multiSz2.IsEmpty());
    AssertEqual("MultiSz has 4 entries.", multiSz2.Count(), SIZE_T(4));
    AssertEqual("MultiSz has length 23.", multiSz2.Length(), SIZE_T(23));
    AssertEqual("First element (peek) is \"Horst\".", ::strcmp(multiSz2.PeekAt(0), "Horst"), 0);
    AssertEqual("First element (get) is \"Horst\".", ::strcmp(multiSz2.GetAt(0).PeekBuffer(), "Horst"), 0);
    AssertEqual("First element (array) is \"Horst\".", ::strcmp(multiSz2[0].PeekBuffer(), "Horst"), 0);
    AssertEqual("Second element (peek) is \"Hugo\".", ::strcmp(multiSz2.PeekAt(1), "Hugo"), 0);
    AssertEqual("Second element (get) is \"Hugo\".", ::strcmp(multiSz2.GetAt(1).PeekBuffer(), "Hugo"), 0);
    AssertEqual("Second element (array) is \"Hugo\".", ::strcmp(multiSz2[1].PeekBuffer(), "Hugo"), 0);
    AssertEqual("Third element (peek) is \"Heinz\".", ::strcmp(multiSz2.PeekAt(2), "Heinz"), 0);
    AssertEqual("Third element (get) is \"Heinz\".", ::strcmp(multiSz2.GetAt(2).PeekBuffer(), "Heinz"), 0);
    AssertEqual("Third element (array) is \"Heinz\".", ::strcmp(multiSz2[2].PeekBuffer(), "Heinz"), 0);
    AssertEqual("Fourth element (peek) is \"Hans\".", ::strcmp(multiSz2.PeekAt(3), "Hans"), 0);
    AssertEqual("Fourth element (get) is \"Hans\".", ::strcmp(multiSz2.GetAt(3).PeekBuffer(), "Hans"), 0);
    AssertEqual("Fourth element (array) is \"Hans\".", ::strcmp(multiSz2[3].PeekBuffer(), "Hans"), 0);

    multiSz = multiSz2;
    AssertEqual("Assignment of MultiSz", multiSz, multiSz2);
    AssertTrue("Assignment of MultiSz (equality)", multiSz == multiSz2);
    AssertFalse("Assignment of MultiSz (not inequal)", multiSz != multiSz2);
    AssertEqual("First element now is \"Horst\".", ::strcmp(multiSz[0].PeekBuffer(), "Horst"), 0);
    AssertEqual("Second element now is \"Hugo\".", ::strcmp(multiSz[1].PeekBuffer(), "Hugo"), 0);
    AssertEqual("Third element now is \"Heinz\".", ::strcmp(multiSz[2].PeekBuffer(), "Heinz"), 0);
    AssertEqual("Fourth element now is \"Hans\".", ::strcmp(multiSz[3].PeekBuffer(), "Hans"), 0);

    multiSz2.Clear();
    AssertTrue("Clear leaves empty MultiSz.", multiSz2.IsEmpty());
    AssertEqual("Clear leaves MultiSz with 0 entries.", multiSz2.Count(), SIZE_T(0));

    AssertFalse("Test fore inequality of MultiSz (equality)", multiSz == multiSz2);
    AssertTrue("Test fore inequality of MultiSz (not inequal)", multiSz != multiSz2);

    multiSz2 = multiSz.PeekBuffer();
    AssertEqual("Assignment of buffer", multiSz, multiSz2);

    MultiSzA multiSz3(multiSz);
    AssertEqual("Copy ctor", multiSz, multiSz3);

    multiSz3.Clear();
    AssertTrue("Empty before append.", multiSz3.IsEmpty());

    multiSz3.Append("Helga");
    AssertFalse("Not empty after append.", multiSz3.IsEmpty());
    AssertEqual("One element appended.", multiSz3.Count(), SIZE_T(1));
    AssertEqual("Element appended is \"Helga\".", ::strcmp(multiSz3[0].PeekBuffer(), "Helga"), 0);

    multiSz3.Append("Horst");
    AssertEqual("One element appended.", multiSz3.Count(), SIZE_T(2));
    AssertEqual("First element untouched.", ::strcmp(multiSz3[0].PeekBuffer(), "Helga"), 0);
    AssertEqual("Element appended is \"Horst\".", ::strcmp(multiSz3[1].PeekBuffer(), "Horst"), 0);

    multiSz3.Insert(2, "Heinz");
    AssertEqual("One element inserted.", multiSz3.Count(), SIZE_T(3));
    AssertEqual("Appended using insert.", ::strcmp(multiSz3[2].PeekBuffer(), "Heinz"), 0);
    AssertEqual("Element before untouched.", ::strcmp(multiSz3[1].PeekBuffer(), "Horst"), 0);

    multiSz3.Insert(0, "Hans");
    AssertEqual("One element inserted.", multiSz3.Count(), SIZE_T(4));
    AssertEqual("Correct element inserted.", ::strcmp(multiSz3[0].PeekBuffer(), "Hans"), 0);

    multiSz3.Insert(3, "Holger");
    AssertEqual("One element inserted.", multiSz3.Count(), SIZE_T(5));
    AssertEqual("Correct element inserted before last.", ::strcmp(multiSz3[3].PeekBuffer(), "Holger"), 0);

    AssertEqual("multiSz3[0] == \"Hans\".", ::strcmp(multiSz3[0].PeekBuffer(), "Hans"), 0);
    AssertEqual("multiSz3[1] == \"Helga\".", ::strcmp(multiSz3[1].PeekBuffer(), "Helga"), 0);
    AssertEqual("multiSz3[2] == \"Horst\".", ::strcmp(multiSz3[2].PeekBuffer(), "Horst"), 0);
    AssertEqual("multiSz3[3] == \"Holger\".", ::strcmp(multiSz3[3].PeekBuffer(), "Holger"), 0);
    AssertEqual("multiSz3[4] == \"Heinz\".", ::strcmp(multiSz3[4].PeekBuffer(), "Heinz"), 0);

    multiSz3.Clear();
    AssertTrue("Empty before insert.", multiSz3.IsEmpty());
    multiSz3.Insert(0, "Hans");
    AssertEqual("Insert into empty.", ::strcmp(multiSz3[0].PeekBuffer(), "Hans"), 0);
    
    multiSz3.Clear();
    AssertException("Insert at invalid index", multiSz3.Insert(1, "Hugo"), vislib::OutOfRangeException);
    AssertException("Array operator at invalid index", multiSz3[0], vislib::OutOfRangeException);
    AssertException("GetAt at invalid index", multiSz3.GetAt(0), vislib::OutOfRangeException);
    AssertEqual("PeekAt at invalid index", (void *) multiSz3.PeekAt(0), (void *) NULL);

    multiSz.Clear();
    AssertTrue("Empty before append.", multiSz.IsEmpty());
    multiSz.Append("Hugo");
    multiSz.Append("Hugo");
    multiSz.Append("Hugo");
    AssertEqual("Three elements appended.", multiSz.Count(), SIZE_T(3));
    multiSz.Remove("Hugo");
    AssertEqual("All elements removed.", multiSz.Count(), SIZE_T(0));
    AssertTrue("Empty after remove.", multiSz.IsEmpty());

    multiSz.Clear();
    AssertTrue("Empty before append.", multiSz.IsEmpty());
    multiSz.Append("Hugo");
    multiSz.Append("Horst");
    multiSz.Append("Heinz");
    multiSz.Append("Helga");
    multiSz.Append("Heinz");
    multiSz.Append("Hans");
    AssertEqual("Three elements appended.", multiSz.Count(), SIZE_T(6));
    
    multiSz.Remove("Heinz");
    AssertEqual("Two elements removed.", multiSz.Count(), SIZE_T(4));
    AssertEqual("multiSz[0] == \"Hugo\".", ::strcmp(multiSz[0].PeekBuffer(), "Hugo"), 0);
    AssertEqual("multiSz[1] == \"Horst\".", ::strcmp(multiSz[1].PeekBuffer(), "Horst"), 0);
    AssertEqual("multiSz[2] == \"Helga\".", ::strcmp(multiSz[2].PeekBuffer(), "Helga"), 0);
    AssertEqual("multiSz[3] == \"Hans\".", ::strcmp(multiSz[3].PeekBuffer(), "Hans"), 0);
    
    multiSz.Remove("Hans");
    AssertEqual("One element removed.", multiSz.Count(), SIZE_T(3));
    AssertEqual("multiSz[0] == \"Hugo\".", ::strcmp(multiSz[0].PeekBuffer(), "Hugo"), 0);
    AssertEqual("multiSz[1] == \"Horst\".", ::strcmp(multiSz[1].PeekBuffer(), "Horst"), 0);
    AssertEqual("multiSz[2] == \"Helga\".", ::strcmp(multiSz[2].PeekBuffer(), "Helga"), 0);

    multiSz.Remove("Horst");
    AssertEqual("One element removed.", multiSz.Count(), SIZE_T(2));
    AssertEqual("multiSz[0] == \"Hugo\".", ::strcmp(multiSz[0].PeekBuffer(), "Hugo"), 0);
    AssertEqual("multiSz[1] == \"Helga\".", ::strcmp(multiSz[1].PeekBuffer(), "Helga"), 0);

    multiSz.Remove("Hugo");
    AssertEqual("One element removed.", multiSz.Count(), SIZE_T(1));
    AssertEqual("multiSz[0] == \"Helga\".", ::strcmp(multiSz[0].PeekBuffer(), "Helga"), 0);

    multiSz.Remove("Hugo");
    AssertEqual("No element removed.", multiSz.Count(), SIZE_T(1));
    AssertEqual("multiSz[0] == \"Helga\".", ::strcmp(multiSz[0].PeekBuffer(), "Helga"), 0);

    multiSz.Remove("");
    AssertEqual("No element removed.", multiSz.Count(), SIZE_T(1));
    AssertEqual("multiSz[0] == \"Helga\".", ::strcmp(multiSz[0].PeekBuffer(), "Helga"), 0);

    multiSz.Remove(NULL);
    AssertEqual("No element removed.", multiSz.Count(), SIZE_T(1));
    AssertEqual("multiSz[0] == \"Helga\".", ::strcmp(multiSz[0].PeekBuffer(), "Helga"), 0);

    multiSz.Remove("Helga");
    AssertEqual("One element removed.", multiSz.Count(), SIZE_T(0));
    AssertTrue("Is empty now.", multiSz.IsEmpty());

}
