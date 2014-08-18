/*
 * testpoolallocator.cpp
 *
 * Copyright (C) 200* by Universitaet Stuttgart (VISUS).
 * Alle Rechte vorbehalten.
 */

#include "testpoolallocator.h"
#include "testhelper.h"
#include "vislib/PoolAllocator.h"


/*
 * testPoolProperties
 */
template <class P> void testPoolProperties(P &pool, unsigned int pageCnt,
        unsigned int elCnt, unsigned int actElCnt) {
#if defined(DEBUG) || defined(_DEBUG)
    unsigned int c1, c2, c3;
    pool._GetCounts(c1, c2, c3);
    AssertEqual("Pool has correct number of pages", c1, pageCnt);
    AssertEqual("Pool has correct number of elements", c2, elCnt);
    AssertEqual("Pool has correct number of active elements", c3, actElCnt);
#endif /* defined(DEBUG) || defined(_DEBUG) */
}


/*
 * allocateObjects
 */
template <class T, class P> void allocateObjects(T* buffer, unsigned int size,
        P &pool, unsigned int cnt) {
    for (unsigned int i = 0; i < size; i++) {
        if (cnt == 0) break;
        if (buffer[i] == NULL) {
            buffer[i] = pool.AllocateObj();
            cnt--;
        }
    }
}


/*
 * deallocateObjects
 */
template <class T, class P> void deallocateObjects(T* buffer,
        unsigned int size, P &pool, unsigned int cnt) {
    unsigned int c = 0;
    for (unsigned int i = 0; i < size; i++) {
        if (buffer[i] != NULL) {
            c++;
        }
    }
    if (c < cnt) cnt = c;
    while (cnt > 0) {
        unsigned int idx = rand() % size;
        if (buffer[idx] != NULL) {
            pool.DeallocateObj(buffer[idx]);
            cnt--;
        }
    }
}


/*
 * testIntPoolImpl
 */
int* testIntPoolImpl(void) {
    vislib::sys::PoolAllocator<int> intPool;
    intPool.SetAllocationSize(10);

    testPoolProperties(intPool, 0, 0, 0);
    int *i = intPool.AllocateObj();
    testPoolProperties(intPool, 1, 10, 1);
    intPool.DeallocateObj(i);
    testPoolProperties(intPool, 1, 10, 0);
    intPool.Cleanup();
    testPoolProperties(intPool, 0, 0, 0);

    const unsigned int bufSize = 100;
    int *buf[bufSize];
    ZeroMemory(buf, bufSize * sizeof(int*));

    allocateObjects(buf, bufSize, intPool, 12);
    testPoolProperties(intPool, 2, 20, 12);
    allocateObjects(buf, bufSize, intPool, 8);
    testPoolProperties(intPool, 2, 20, 20);
    deallocateObjects(buf, bufSize, intPool, 19);
    testPoolProperties(intPool, 2, 20, 1);
    intPool.Cleanup();
    testPoolProperties(intPool, 1, 10, 1);
    allocateObjects(buf, bufSize, intPool, 9);
    testPoolProperties(intPool, 1, 10, 10);
    deallocateObjects(buf, bufSize, intPool, 10);
    intPool.Cleanup();
    testPoolProperties(intPool, 0, 0, 0);
    allocateObjects(buf, bufSize, intPool, 100);
    testPoolProperties(intPool, 10, 100, 100);
    deallocateObjects(buf, bufSize, intPool, 100);
    testPoolProperties(intPool, 10, 100, 0);
    allocateObjects(buf, bufSize, intPool, 100);
    testPoolProperties(intPool, 10, 100, 100);
    intPool.Cleanup();
    testPoolProperties(intPool, 10, 100, 100);
    deallocateObjects(buf, bufSize, intPool, 10);
    testPoolProperties(intPool, 10, 100, 90);
    deallocateObjects(buf, bufSize, intPool, 89);
    testPoolProperties(intPool, 10, 100, 1);
    // keep one!
    for (unsigned int i = 0; i < bufSize; i++) {
        if (buf[i] != NULL) return buf[i];
    }
    return NULL;
}


/*
 * testIntPool
 */
void testIntPool(void) {
    int *memoryleak = testIntPoolImpl();
    vislib::sys::PoolAllocator<int>::Deallocate(memoryleak);
}


/**
 * Test class
 */
class TestClass {
public:

    /** Counts how often the ctor has been called */
    static unsigned int ctorCnt;

    /** Counts how often the dtor has been called */
    static unsigned int dtorCnt;

    /** Ctor. */
    TestClass(void) : alive(true) {
        ctorCnt++;
    }

    /** Dtor */
    ~TestClass(void) {
        this->alive = false;
        dtorCnt++;
    }

    /** stupid dummy flag */
    bool alive;

};


/*
 * TestClass::ctorCnt
 */
unsigned int TestClass::ctorCnt = 0;


/*
 * TestClass::dtorCnt
 */
unsigned int TestClass::dtorCnt = 0;


/*
 * testTestClassProperties
 */
void testTestClassProperties(unsigned int ctorCnt, unsigned int dtorCnt) {
    AssertEqual("TestClass::Ctor called as many times as expected",
        TestClass::ctorCnt, ctorCnt);
    AssertEqual("TestClass::Dtor called as many times as expected",
        TestClass::dtorCnt, dtorCnt);
}


/*
 * testClassPool
 */
void testClassPool(void) {
    TestClass::ctorCnt = 0;
    TestClass::dtorCnt = 0;

    testTestClassProperties(0, 0);

    vislib::sys::PoolAllocator<TestClass> pool;
    pool.SetAllocationSize(10);

    const unsigned int bufSize = 100;
    TestClass *buf[bufSize];
    ZeroMemory(buf, bufSize * sizeof(TestClass*));

    testTestClassProperties(0, 0);

    allocateObjects(buf, bufSize, pool, 12);
    testPoolProperties(pool, 2, 20, 12);
    testTestClassProperties(12, 0);

    allocateObjects(buf, bufSize, pool, 8);
    testPoolProperties(pool, 2, 20, 20);
    testTestClassProperties(20, 0);

    deallocateObjects(buf, bufSize, pool, 19);
    testPoolProperties(pool, 2, 20, 1);
    testTestClassProperties(20, 19);

    pool.Cleanup();
    testTestClassProperties(20, 19);
    testPoolProperties(pool, 1, 10, 1);

    allocateObjects(buf, bufSize, pool, 9);
    testTestClassProperties(29, 19);
    testPoolProperties(pool, 1, 10, 10);

    deallocateObjects(buf, bufSize, pool, 10);
    testTestClassProperties(29, 29);

    pool.Cleanup();
    testPoolProperties(pool, 0, 0, 0);
    testTestClassProperties(29, 29);

    allocateObjects(buf, bufSize, pool, 100);
    testPoolProperties(pool, 10, 100, 100);
    testTestClassProperties(129, 29);

    deallocateObjects(buf, bufSize, pool, 100);
    testPoolProperties(pool, 10, 100, 0);
    testTestClassProperties(129, 129);

    allocateObjects(buf, bufSize, pool, 100);
    testPoolProperties(pool, 10, 100, 100);
    testTestClassProperties(229, 129);

    pool.Cleanup();
    testTestClassProperties(229, 129);
    testPoolProperties(pool, 10, 100, 100);

    deallocateObjects(buf, bufSize, pool, 10);
    testTestClassProperties(229, 139);
    testPoolProperties(pool, 10, 100, 90);

    deallocateObjects(buf, bufSize, pool, 90);
    testPoolProperties(pool, 10, 100, 0);
    testTestClassProperties(229, 229);
}


/*
 * TestPoolAllocator
 */
void TestPoolAllocator(void) {
    testIntPool();
    testClassPool();
}
