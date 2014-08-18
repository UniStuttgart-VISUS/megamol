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
#include "vislib/Map.h"
#include "vislib/Pair.h"
#include "vislib/PtrArray.h"
#include "vislib/SingleLinkedList.h"
#include "vislib/String.h"
#include "vislib/SynchronisedArray.h"
#include "vislib/ConsoleProgressBar.h"


/*
 * TestArray
 */
void TestArray(void) {
    using vislib::Array;
    using vislib::PtrArray;

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


    PtrArray<int> intPtrAry;
    intPtrAry.Add(new int(5));
    AssertEqual("Added pointer to 5", *(intPtrAry[0]), 5);
    
    intPtrAry.Add(new int(4));
    AssertEqual("Added pointer to 4", *(intPtrAry[1]), 4);
    
    intPtrAry.Resize(1);
    AssertEqual("Removed all but first element", intPtrAry.Count(), SIZE_T(1));
    AssertEqual("First element was not changed", *(intPtrAry[0]), 5);

    intPtrAry.Clear(true);
    AssertEqual("Removed all elements", intPtrAry.Count(), SIZE_T(0));
    AssertEqual("Deallocated whole array", intPtrAry.Capacity(), SIZE_T(0));


    vislib::sys::SynchronisedArray<int> syncedIntArray;

    intAry.Clear();
    intAry.Add(1);
    intAry.Add(2);
    intAry.Add(3);
    intAry.Add(4);
    intAry.Erase(0, 1);
    AssertEqual("Array has 3 elements", intAry.Count(), SIZE_T(3));
    AssertEqual("Array[0] == 2", intAry[0], 2);
    AssertEqual("Array[1] == 3", intAry[1], 3);
    AssertEqual("Array[2] == 4", intAry[2], 4);

    intAry.Clear();
    intAry.Add(1);
    intAry.Add(2);
    intAry.Add(3);
    intAry.Add(4);
    intAry.Erase(1, 1);
    AssertEqual("Array has 3 elements", intAry.Count(), SIZE_T(3));
    AssertEqual("Array[0] == 1", intAry[0], 1);
    AssertEqual("Array[1] == 3", intAry[1], 3);
    AssertEqual("Array[2] == 4", intAry[2], 4);
}


/*
 * TestSingleLinkedList
 */
void TestSingleLinkedList(void) {
    typedef vislib::SingleLinkedList<int> IntList;

    IntList list;

    list.Append(12);
    list.Add(7);
    list.Prepend(6);

    AssertEqual("List contains 3 elements", list.Count(), SIZE_T(3));

    IntList::Iterator i = list.GetIterator();
    AssertTrue("Iterator valid", i.HasNext());
    AssertEqual("Element[1] = 6", i.Next(), 6);
    AssertTrue("Iterator valid", i.HasNext());
    AssertEqual("Element[2] = 12", i.Next(), 12);
    AssertTrue("Iterator valid", i.HasNext());
    AssertEqual("Element[3] = 7", i.Next(), 7);
    AssertFalse("Iterator at end", i.HasNext());

    vislib::ConstIterator<IntList::Iterator> ci = list.GetConstIterator();
    AssertTrue("Iterator valid", ci.HasNext());
    AssertEqual("Element[1] = 6", ci.Next(), 6);
    AssertTrue("Iterator valid", ci.HasNext());
    AssertEqual("Element[2] = 12", ci.Next(), 12);
    AssertTrue("Iterator valid", ci.HasNext());
    AssertEqual("Element[3] = 7", ci.Next(), 7);
    AssertFalse("Iterator at end", ci.HasNext());

}


/*
 * TestHeap
 */
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


/*
 * TestMap
 */
void TestMap(void) {
    using vislib::Map;
    using vislib::SingleLinkedList;

    Map<int, float> map;

    map.Set(12, 13.5f);
    ::AssertEqual("Map contains one element", map.Count(), static_cast<SIZE_T>(1));
    ::AssertEqual("Map element correct", map[12], 13.5f);
    ::AssertEqual("New Map element correct", map[11], 0.0f);
    ::AssertEqual("Map contains two elements", map.Count(), static_cast<SIZE_T>(2));
    map[11] = 1.01f;
    ::AssertEqual("New Map element correct", map[11], 1.01f);
    map.Remove(11);
    ::AssertEqual("Map contains one element", map.Count(), static_cast<SIZE_T>(1));
    map[10] = 2.0f;
    map[13] = 2.0f;
    map[0] = 2.0f;
    ::AssertEqual("Map contains four elements", map.Count(), static_cast<SIZE_T>(4));
    SingleLinkedList<int> keys = map.FindKeys(2.0f);
    ::AssertEqual("Three keys associate '2.0f'", keys.Count(), static_cast<SIZE_T>(3));
    ::AssertTrue("Key 0 present", keys.Contains(0));
    ::AssertTrue("Key 10 present", keys.Contains(10));
    ::AssertTrue("Key 13 present", keys.Contains(13));
    ::AssertTrue("Key 13 present", map.Contains(13));
    ::AssertFalse("Key 14 not present", map.Contains(14));
    ::AssertEqual<float*>("Key 14 not present", map.FindValue(14), NULL);
    ::AssertEqual<float*>("Key 14 not present", map.FindValue(14), NULL);
    ::AssertEqual("Map[10] correct", map[10], 2.0f);
    ::AssertEqual("Map[13] correct", map[13], 2.0f);
    map[10] = 4.0f;
    ::AssertEqual("Map[10] correct", map[10], 4.0f);
    ::AssertEqual("Map[13] correct", map[13], 2.0f);
    map.Clear();
    ::AssertEqual("Map is empty", map.Count(), static_cast<SIZE_T>(0));
    ::AssertTrue("Map is empty", map.IsEmpty());

}


/*
 * intSortCompare
 */
int intSortCompare(const int& lhs, const int& rhs) {
    return lhs - rhs;
}


/*
 * TestSingleLinkedListSort
 */
void TestSingleLinkedListSort(void) {
    vislib::SingleLinkedList<int> list;

    list.Add(22);
    list.Add(8);
    list.Add(21);
    list.Add(22);
    list.Add(50);
    list.Add(2);
    list.Add(1);
    list.Add(10);

    AssertEqual<int>("List filled with 8 Elements", int(list.Count()), 8);

    list.Sort(intSortCompare);

    AssertEqual<int>("List still contains 8 Elements", int(list.Count()), 8);

    vislib::SingleLinkedList<int>::Iterator iter = list.GetIterator();
    AssertTrue("Iterator before Element 1", iter.HasNext());
    AssertEqual("Element 1 = 1", iter.Next(), 1);
    AssertTrue("Iterator before Element 2", iter.HasNext());
    AssertEqual("Element 2 = 2", iter.Next(), 2);
    AssertTrue("Iterator before Element 3", iter.HasNext());
    AssertEqual("Element 3 = 8", iter.Next(), 8);
    AssertTrue("Iterator before Element 4", iter.HasNext());
    AssertEqual("Element 4 = 10", iter.Next(), 10);
    AssertTrue("Iterator before Element 5", iter.HasNext());
    AssertEqual("Element 5 = 21", iter.Next(), 21);
    AssertTrue("Iterator before Element 6", iter.HasNext());
    AssertEqual("Element 6 = 22", iter.Next(), 22);
    AssertTrue("Iterator before Element 7", iter.HasNext());
    AssertEqual("Element 7 = 22", iter.Next(), 22);
    AssertTrue("Iterator before Element 8", iter.HasNext());
    AssertEqual("Element 8 = 50", iter.Next(), 50);
    AssertFalse("Iterator at end of list", iter.HasNext());

    list.Clear();

    const SIZE_T cnt = 10000000;
    for (SIZE_T i = 0; i < cnt; i++) {
        list.Add(rand());
    }

#ifdef _WIN32
    DWORD startTick = GetTickCount();
#endif /* _WIN32 */
    AssertEqual("List filled with random Elements", list.Count(), cnt);
    list.Sort(intSortCompare);
    AssertEqual("List still contains random Elements", list.Count(), cnt);
#ifdef _WIN32
    DWORD duration = GetTickCount() - startTick;
    printf("Sorted in %u milliseconds\n", duration);
#endif /* _WIN32 */

    bool growing = true;
    int ov = -1;
    iter = list.GetIterator();
    while (iter.HasNext()) {
        int v = iter.Next();
        if (v < ov) {
            growing = false;
        }
        ov = v;
    }
    AssertTrue("List sorted Accending", growing);

}


/*
 * TestArraySort
 */
void TestArraySort(void) {
    vislib::Array<int> arr;
    vislib::sys::ConsoleProgressBar progBar;
    const unsigned int size = 10000000;
    unsigned int i;
    
    progBar.Start("Filling Array", size);
    arr.SetCount(size);
    for (i = 0; i < size; i++) {
        arr[i] = rand();
        progBar.Set(i);
    }
    progBar.Stop();

    arr.Sort(intSortCompare);

    progBar.Start("Testing Sort output", size);
    bool growing = true;
    for (i = 1; i < size; i++) {
        if (arr[i - 1] > arr[i]) {
            growing = false;
        }
        progBar.Set(i);
    }
    progBar.Stop();

    AssertTrue("Array sorted Accending", growing);

}
