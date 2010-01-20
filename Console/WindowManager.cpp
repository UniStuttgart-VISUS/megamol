/*
 * WindowManager.cpp
 *
 * Copyright (C) 2008 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "WindowManager.h"


/*
 * very ugly declaration of the main-scope function
 * 'viewerRequestsAppExitCallback'
 */
void viewerRequestsAppExitCallback(void *userData, void *params);


/*
 * megamol::console::WindowManager::Instance
 */
megamol::console::WindowManager*
megamol::console::WindowManager::Instance(void) {
    static vislib::SmartPtr<megamol::console::WindowManager> inst;
    if (inst.IsNull()) {
        inst = new megamol::console::WindowManager();
    }
    return inst.operator->();
}


/*
 * megamol::console::WindowManager::WindowManager
 */
megamol::console::WindowManager::WindowManager(void) : windows() {
}


/*
 * megamol::console::WindowManager::~WindowManager
 */
megamol::console::WindowManager::~WindowManager(void) {
}


/*
 * megamol::console::WindowManager::Add
 */
void megamol::console::WindowManager::Add(
        vislib::SmartPtr<megamol::console::Window>& window) {
    this->windows.Add(window);
}


/*
 * megamol::console::WindowManager::Cleanup
 */
void megamol::console::WindowManager::Cleanup(void) {
    vislib::SingleLinkedList<vislib::SmartPtr<
        megamol::console::Window> >::Iterator iter
        = this->windows.GetIterator();
    while (iter.HasNext()) {
        vislib::SmartPtr<megamol::console::Window>& win = iter.Next();
        if (win->IsClosed()) {
            win = NULL;
            this->windows.Remove(iter);
        }
    }
}


/*
 * megamol::console::WindowManager::CloseAll
 */
void megamol::console::WindowManager::CloseAll(void) {
    this->windows.Clear();
}


/*
 * megamol::console::WindowManager::MarkAllForClosure
 */
void megamol::console::WindowManager::MarkAllForClosure(void) {
    vislib::SingleLinkedList<vislib::SmartPtr<
        megamol::console::Window> >::Iterator iter
        = this->windows.GetIterator();
    while (iter.HasNext()) {
        iter.Next()->MarkToClose();
    }
    viewerRequestsAppExitCallback(NULL, NULL);
}
