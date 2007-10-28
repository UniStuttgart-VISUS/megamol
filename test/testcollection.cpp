/*
 * testcollection.cpp
 *
 * Copyright (C) 2006 - 2007 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#include "testcollection.h"
#include "testhelper.h"

#include "vislib/Array.h"
#include "vislib/Heap.h"
#include "vislib/Pair.h"
#include "vislib/String.h"


void TestArray(void) {
    using vislib::Array;

    Array<int> intAry;

    ::AssertEqual("Array default capacity", intAry.Capacity(), Array<int>::DEFAULT_CAPACITY);
    ::AssertEqual("Array initially empty", intAry.Count(), SIZE_T(0));
    ::AssertTrue("IsEmpty method", intAry.IsEmpty());
    
    intAry.Append(4);
    ::AssertEqual("Appended one element", intAry.Count(), SIZE_T(1));
    ::AssertTrue("New element in array", intAry.Contains(4));
    ::AssertEqual("New element is at expected position", intAry[0], 4);

    intAry.Append(6);
    intAry.Append(43);
    ::AssertEqual("Appended two elements", intAry.Count(), SIZE_T(3));

    intAry.AssertCapacity(Array<int>::DEFAULT_CAPACITY + 5);
    ::AssertEqual("Increased capacity", intAry.Capacity(), Array<int>::DEFAULT_CAPACITY + 5);

    intAry.Trim();
    ::AssertEqual("Capacity set to current count", intAry.Count(), intAry.Capacity());

    intAry.Append(99);
    ::AssertEqual("New element appended to full array", intAry[INT(intAry.Count() - 1)], 99);

    intAry.Erase(1, 2);
    ::AssertEqual("Two elements erased", intAry.Count(), SIZE_T(2));
    ::AssertEqual("Capacity unchanged", intAry.Capacity(), SIZE_T(4));
    ::AssertEqual("First element unchanged", intAry[0], 4);
    ::AssertEqual("Last element unchanged", intAry[1], 99);

    intAry.Insert(1, 5);
    ::AssertEqual("First element unchanged", intAry[0], 4);
    ::AssertEqual("Element inserted", intAry[1], 5);
    ::AssertEqual("Last element moved", intAry[2], 99);

    intAry.Insert(0, 1);
    ::AssertEqual("Insert at begin", intAry[0], 1);

    intAry.Insert(intAry.Count() - 1, 1981);
    ::AssertEqual("Insert at end", intAry[intAry.Count() - 2], 1981);
    ::AssertEqual("End moved", intAry.Last(), 99);

    intAry.Insert(intAry.Count(), 2007);
    ::AssertEqual("Append using Insert", intAry.Last(), 2007);
    

    Array<int> intAry2(10, 42);
    
    ::AssertEqual("Initially filled array", intAry2.Count(), SIZE_T(10));

    intAry = intAry2;
    ::AssertEqual("Assignment copies all elements", intAry.Count(), SIZE_T(10));
    ::AssertTrue("Assignment copies correct elements", intAry.Contains(42));
    
    intAry.RemoveAll(43);
    ::AssertEqual("Removing non-exisiting element has no effect", intAry.Count(), SIZE_T(10));
    
    intAry.RemoveAll(42);
    ::AssertFalse("Remove element", intAry.Contains(42));
    ::AssertTrue("Remove affects all matching elements", intAry.IsEmpty());

    intAry2.Erase(intAry2.Count());
    ::AssertEqual("Erase on non-exisiting index has no effect", intAry2.Count(), SIZE_T(10));

    intAry2.Erase(3);
    ::AssertEqual("Erase one element", intAry2.Count(), SIZE_T(9));

    intAry2.Erase(2, 5);
    ::AssertEqual("Erase five elements", intAry2.Count(), SIZE_T(4));

    intAry2.Erase(2, 5);
    ::AssertEqual("Erase begining at 2", intAry2.Count(), SIZE_T(2));

    intAry2.Clear();
    ::AssertTrue("Clear array", intAry2.IsEmpty());

    intAry2.Trim();
    ::AssertEqual("Trim empty array", intAry2.Capacity(), SIZE_T(0));

    
    Array<vislib::StringA> strAry;
    strAry.Append("Horst");
    AssertTrue("Contains \"Horst\"", strAry.Contains("Horst"));
}


void TestHeap(void) {
    using vislib::Array;
    using vislib::Heap;
    using vislib::Pair;

    typedef vislib::Pair<int, char> MyPair;
    
    Heap<MyPair> heap;

    ::AssertEqual("Heap default capacity", heap.Capacity(), Array<MyPair>::DEFAULT_CAPACITY);
    ::AssertEqual("Heap initially empty", heap.Count(), SIZE_T(0));
    ::AssertTrue("IsEmpty method", heap.IsEmpty());

    heap.Add(MyPair(3, 'H'));
    ::AssertEqual("One element added", heap.Count(), SIZE_T(1));
    ::AssertFalse("IsEmpty method", heap.IsEmpty());

    heap.Add(MyPair(7, 't'));
    heap.Add(MyPair(5, 'r'));
    heap.Add(MyPair(6, 's'));
    heap.Add(MyPair(4, 'o'));
    ::AssertEqual("Four additional elements added", heap.Count(), SIZE_T(5));

    ::AssertEqual("Get element 'H'", heap.First().Value(), 'H');
    heap.RemoveFirst();
    ::AssertEqual("Get element 'o'", heap.First().Value(), 'o');
    heap.RemoveFirst();
    ::AssertEqual("Get element 'r'", heap.First().Value(), 'r');
    heap.RemoveFirst();
    ::AssertEqual("Get element 's'", heap.First().Value(), 's');
    heap.RemoveFirst();
    ::AssertEqual("Get element 't'", heap.First().Value(), 't');
    heap.RemoveFirst();
    ::AssertTrue("Heap is empty now", heap.IsEmpty());

    heap.Add(MyPair(7, 't'));
    heap.Add(MyPair(5, 'r'));
    heap.Add(MyPair(4, 'o'));
    heap.Add(MyPair(0, 'H'));
    heap.Add(MyPair(6, 's'));
    heap.Add(MyPair(4, 'o'));
    ::AssertEqual("Heap filled with duplicate key", heap.Count(), SIZE_T(6));

    ::AssertEqual("Get element 'H'", heap.First().Value(), 'H');
    heap.RemoveFirst();
    ::AssertEqual("Get element 'o'", heap.First().Value(), 'o');
    heap.RemoveFirst();
    ::AssertEqual("Get element 'o'", heap.First().Value(), 'o');
    heap.RemoveFirst();
    ::AssertEqual("Get element 'r'", heap.First().Value(), 'r');
    heap.RemoveFirst();
    ::AssertEqual("Get element 's'", heap.First().Value(), 's');
    heap.RemoveFirst();
    ::AssertEqual("Get element 't'", heap.First().Value(), 't');
    heap.RemoveFirst();
    ::AssertTrue("Heap is empty now", heap.IsEmpty());

    heap.Add(MyPair(4, 'o'));
    heap.Clear();
    ::AssertTrue("Heap is empty after Clear", heap.IsEmpty());
}
