/*
 * DirectoryIterator.inl
 *
 * Copyright (C) 2007 - 2010 by Visualisierungsinstitut Universitaet Stuttgart. 
 */


#ifdef _WIN32
#include <Windows.h>
#else /* _WIN32 */
#include "vislib/UnsupportedOperationException.h"
#endif /* _WIN32 */

namespace vislib {
namespace sys {


    /*
     * DirectoryIterator<T>::DirectoryIterator
     */
    template<> DirectoryIterator<CharTraitsA>::DirectoryIterator(
            const Char* path, bool isPattern, bool showDirs) : nextItem(),
            currentItem(), omitFolders(!showDirs) {
#ifdef _WIN32
        WIN32_FIND_DATAA fd;
        StringA p = path;
        if (!isPattern) {
            if (!p.EndsWith(vislib::sys::Path::SEPARATOR_A)) {
                if (!p.EndsWith(":")) {
                    p += vislib::sys::Path::SEPARATOR_A;
                }
            }
            p += "*.*";
        }

        this->findHandle = FindFirstFileA(p, &fd);
        if (this->findHandle == INVALID_HANDLE_VALUE) {
            throw SystemException(__FILE__, __LINE__);
        }
        if (fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) {
            this->nextItem.Type = Entry::DIRECTORY;
            if (this->omitFolders || (strcmp(fd.cFileName, ".") == 0)
                    || (strcmp(fd.cFileName, "..") == 0)) {
                this->fetchNextItem();
            } else {
                this->nextItem.Path = fd.cFileName;
            }
        } else {
            this->nextItem.Type = Entry::FILE;
            this->nextItem.Path = fd.cFileName;
        }
#else /* _WIN32 */
        if (isPattern || !showDirs) {
            // TODO: Fix me
            throw vislib::UnsupportedOperationException(
                "DirectoryIterator(, true, false)", __FILE__, __LINE__);
        }
        if ((this->dirStream = opendir(path)) == NULL) {
            throw SystemException(__FILE__, __LINE__);
        }
        this->basePath = path;
        this->fetchNextItem();
#endif /* _WIN32 */
    }


    /*
     * DirectoryIterator<T>::DirectoryIterator
     */
    template<> DirectoryIterator<CharTraitsW>::DirectoryIterator(
            const Char* path, bool isPattern, bool showDirs) : nextItem(),
            currentItem(), omitFolders(!showDirs) {
#ifdef _WIN32
        WIN32_FIND_DATAW fd;
        StringW p = path;
        if (!isPattern) {
            if (!p.EndsWith(vislib::sys::Path::SEPARATOR_W)) {
                if (!p.EndsWith(L":")) {
                    p += vislib::sys::Path::SEPARATOR_W;
                }
            }
            p += L"*.*";
        }

        this->findHandle = FindFirstFileW(p, &fd);
        if (this->findHandle == INVALID_HANDLE_VALUE) {
            throw SystemException(__FILE__, __LINE__);
        }
        if (fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) {
            this->nextItem.Type = Entry::DIRECTORY;
            if (this->omitFolders || (wcscmp(fd.cFileName, L".") == 0)
                    || (wcscmp(fd.cFileName, L"..") == 0)) {
                this->fetchNextItem();
            } else {
                this->nextItem.Path = fd.cFileName;
            }
        } else {
            this->nextItem.Type = Entry::FILE;
            this->nextItem.Path = fd.cFileName;
        }
#else /* _WIN32 */
        if (isPattern || !showDirs) {
            // TODO: Fix me
            throw vislib::UnsupportedOperationException(
                "DirectoryIterator(, true, false)", __FILE__, __LINE__);
        }
        if ((this->dirStream = opendir(StringA(path).PeekBuffer())) == NULL) {
            throw SystemException(__FILE__, __LINE__);
        }
        this->basePath = path;
        this->fetchNextItem();
#endif /* _WIN32 */
    }
    

    /*
     * DirectoryIterator<T>::fetchNextItem
     */
    template<> void DirectoryIterator<CharTraitsA>::fetchNextItem(void) {
#ifdef _WIN32
        WIN32_FIND_DATAA fd;
        DWORD le;
        bool found = false;
        do {
            if (FindNextFileA(this->findHandle, &fd) == 0) {
                if ((le = GetLastError()) == ERROR_NO_MORE_FILES) {
                    this->nextItem.Path.Clear();
                    found = true;
                } else {
                    throw SystemException(le, __FILE__, __LINE__);
                }
            } else {
                if (fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) {
                    this->nextItem.Type = Entry::DIRECTORY;
                    if (!this->omitFolders && (strcmp(fd.cFileName, ".") != 0)
                            && (strcmp(fd.cFileName, "..") != 0)) {
                        found = true;
                    }
                } else {
                    this->nextItem.Type = Entry::FILE;
                    found = true;
                }
                this->nextItem.Path = fd.cFileName;
            }
        } while (!found);
#else /* _WIN32 */
        struct dirent *de;
        do {
            // BUG: Linux documentation is all lies. the errno stunt does not work at all.
            //errno = 0;
            if ((de = readdir(this->dirStream)) == NULL) {
                //if (errno == 0) {
                //	this->nextItem.Path.Clear();
                //} else {
                //	throw SystemException(errno, __FILE__, __LINE__);
                //}
            } else {
                if (vislib::sys::File::IsDirectory(this->basePath + Path::SEPARATOR_A + de->d_name)) {
                    this->nextItem.Type = Entry::DIRECTORY;
                } else {
                    this->nextItem.Type = Entry::FILE;
                }
            }
        } while (de != NULL && ((strcmp(de->d_name, ".") == 0) || (strcmp(de->d_name, "..") == 0)));
        if (de == NULL) {
            this->nextItem.Path.Clear();
        }
        else {
            this->nextItem.Path = de->d_name;
        }
#endif /* _WIN32 */
    }


    /*
     * DirectoryIterator<T>::fetchNextItem
     */
    template<> void DirectoryIterator<CharTraitsW>::fetchNextItem(void) {
#ifdef _WIN32
        WIN32_FIND_DATAW fd;
        DWORD le;
        bool found = false;
        do {
            if (FindNextFileW(this->findHandle, &fd) == 0) {
                if ((le = GetLastError()) == ERROR_NO_MORE_FILES) {
                    this->nextItem.Path.Clear();
                    found = true;
                } else {
                    throw SystemException(le, __FILE__, __LINE__);
                }
            } else {
                if (fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) {
                    this->nextItem.Type = Entry::DIRECTORY;
                    if (!this->omitFolders && (wcscmp(fd.cFileName, L".") != 0)
                            && (wcscmp(fd.cFileName, L"..") != 0)) {
                        found = true;
                    }
                } else {
                    this->nextItem.Type = Entry::FILE;
                    found = true;
                }
                this->nextItem.Path = fd.cFileName;
            }
        } while (!found);
#else /* _WIN32 */
        struct dirent *de;
        do {
            // BUG: Linux documentation is all lies. the errno stunt does not work at all.
            //errno = 0;
            if ((de = readdir(this->dirStream)) == NULL) {
                //if (errno == 0) {
                //	this->nextItem.Path.Clear();
                //} else {
                //	throw SystemException(errno, __FILE__, __LINE__);
                //}
            } else {
                if (vislib::sys::File::IsDirectory(this->basePath + Path::SEPARATOR_A + de->d_name)) {
                    this->nextItem.Type = Entry::DIRECTORY;
                } else {
                    this->nextItem.Type = Entry::FILE;
                }
            }
        } while (de != NULL && ((strcmp(de->d_name, ".") == 0) || (strcmp(de->d_name, "..") == 0)));
        if (de == NULL) {
            this->nextItem.Path.Clear();
        } else {
            this->nextItem.Path = vislib::StringW(de->d_name);
        }
#endif /* _WIN32 */
    }


} /* end namespace sys */
} /* end namespace vislib */


//////////////////////////////////////////////////////////////////////////////////
//// BEGIN OF PARTIAL TEMPLATE SPECIALISATION FOR CharTraitsA
//
//
///*
// * vislib::sys::DirectoryIterator<vislib::CharTraitsA>::DirectoryIterator
// */
//vislib::sys::DirectoryIterator<vislib::CharTraitsA>::DirectoryIterator(
//        const Char *path) {
//#ifdef _WIN32
//    WIN32_FIND_DATAA fd;
//
//    StringA p = path;
//    if (!p.EndsWith(vislib::sys::Path::SEPARATOR_A)) {
//        if (!p.EndsWith(":")) {
//            p += vislib::sys::Path::SEPARATOR_A;
//        }
//    }
//
//    this->findHandle = FindFirstFileA(p + "*.*", &fd);
//    if (this->findHandle == INVALID_HANDLE_VALUE) {
//        throw SystemException(__FILE__, __LINE__);
//    }
//
//    while ((strcmp(fd.cFileName, ".") == 0) || (strcmp(fd.cFileName, "..") == 0)) {
//        if (FindNextFileA(this->findHandle, &fd) == 0) {
//            DWORD le;
//            if ((le = GetLastError()) == ERROR_NO_MORE_FILES) {
//                this->nextItem.Path.Clear();
//            } else {
//                throw SystemException(le, __FILE__, __LINE__);
//            }
//        }
//    }
//
//    if (fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) {
//        this->nextItem.Type = Item::DIRECTORY;
//    } else {
//        this->nextItem.Type = Item::FILE;
//    }
//    this->nextItem.Path = fd.cFileName;
//#else /* _WIN32 */
//    struct dirent *de;
//
//    if ((this->dirStream = opendir(path)) == NULL) {
//        throw SystemException(__FILE__, __LINE__);
//    }
//    this->basePath = path;
//    // BUG: Linux documentation is all lies. the errno stunt does not work at all.
//    //errno = 0;
//    do {
//        if ((de = readdir(this->dirStream)) == NULL) {
//            //if (errno == 0) {
//            //	this->nextItem.Path.Clear();
//            //} else {
//            //	throw SystemException(errno, __FILE__, __LINE__);
//            //}
//        }
//    } while (de != NULL && ((strcmp(de->d_name, ".") == 0) || (strcmp(de->d_name, "..") == 0)));
//    if (de == NULL) {
//        this->nextItem.Path.Clear();
//    }	else {
//        if (vislib::sys::File::IsDirectory(this->basePath + Path::SEPARATOR_A + de->d_name)) {
//            this->nextItem.Type = Item::DIRECTORY;
//        } else {
//            this->nextItem.Type = Item::FILE;
//        }
//        this->nextItem.Path = de->d_name;
//    }
//#endif /* _WIN32 */
//}
//
//
///*
// * vislib::sys::DirectoryIterator<vislib::CharTraitsA>::~DirectoryIterator
// */
//vislib::sys::DirectoryIterator<vislib::CharTraitsA>::~DirectoryIterator(void) {
//#ifdef _WIN32
//    if (this->findHandle != INVALID_HANDLE_VALUE) {
//        FindClose(this->findHandle);
//    }
//#else /* _WIN32 */
//    if (this->dirStream != NULL) {
//        closedir(dirStream);
//    }
//#endif /* WIN32 */
//}
//
//
///*
// * vislib::sys::DirectoryIterator<vislib::CharTraitsA>::HasNext
// */
//bool vislib::sys::DirectoryIterator<vislib::CharTraitsA>::HasNext() const {
//    return !this->nextItem.Path.IsEmpty();
//}
//
//
///*
// * vislib::sys::DirectoryIterator<vislib::CharTraitsA>::Next
// */
//vislib::sys::DirectoryEntry<vislib::CharTraitsA>& 
//vislib::sys::DirectoryIterator<vislib::CharTraitsA>::Next() {
//    this->currentItem = this->nextItem;
//#ifdef _WIN32
//    if (this->currentItem.Path.IsEmpty()) {
//        throw NoSuchElementException("No next element.", __FILE__, __LINE__);
//    } else {
//        WIN32_FIND_DATAA fd;
//        DWORD le;
//
//        do {
//            if (FindNextFileA(this->findHandle, &fd) == 0) {
//                if ((le = GetLastError()) == ERROR_NO_MORE_FILES) {
//                    this->nextItem.Path.Clear();
//                } else {
//                    throw IOException(le, __FILE__, __LINE__);
//                }
//            } else {
//                if (fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) {
//                    this->nextItem.Type = Item::DIRECTORY;
//                } else {
//                    this->nextItem.Type = Item::FILE;
//                }
//                this->nextItem.Path = fd.cFileName;
//            }
//        } while ((strcmp(fd.cFileName, ".") == 0) || (strcmp(fd.cFileName, "..") == 0));
//        return this->currentItem;
//    }
//#else /* _WIN32 */
//    if (this->currentItem.Path.IsEmpty()) {
//        throw NoSuchElementException("No next element.", __FILE__, __LINE__);
//    } else {
//        struct dirent *de;
//        do {
//            // BUG: Linux documentation is all lies. the errno stunt does not work at all.
//            //errno = 0;
//            if ((de = readdir(this->dirStream)) == NULL) {
//                //if (errno == 0) {
//                //	this->nextItem.Path.Clear();
//                //} else {
//                //	throw SystemException(errno, __FILE__, __LINE__);
//                //}
//            } else {
//                if (vislib::sys::File::IsDirectory(this->basePath + Path::SEPARATOR_A + de->d_name)) {
//                    this->nextItem.Type = Item::DIRECTORY;
//                } else {
//                    this->nextItem.Type = Item::FILE;
//                }
//            }
//        } while (de != NULL && ((strcmp(de->d_name, ".") == 0) || (strcmp(de->d_name, "..") == 0)));
//        if (de == NULL) {
//            this->nextItem.Path.Clear();
//        }
//        else {
//            this->nextItem.Path = de->d_name;
//        }
//        return this->currentItem;
//    }
//#endif /* _WIN32 */
//}
//
//// END OF PARTIAL TEMPLATE SPECIALISATION FOR CharTraitsA
//////////////////////////////////////////////////////////////////////////////////
//
//////////////////////////////////////////////////////////////////////////////////
//// BEGIN OF PARTIAL TEMPLATE SPECIALISATION FOR CharTraitsW
//
///*
// * vislib::sys::DirectoryIterator<vislib::CharTraitsW>::DirectoryIterator
// */
//vislib::sys::DirectoryIterator<vislib::CharTraitsW>::DirectoryIterator(
//        const Char *path) {
//#ifdef _WIN32
//    WIN32_FIND_DATAW fd;
//
//    StringW p = path;
//    if (!p.EndsWith(vislib::sys::Path::SEPARATOR_W)) {
//        if (!p.EndsWith(L":")) {
//            p += vislib::sys::Path::SEPARATOR_W;
//        }
//    }
//
//    this->findHandle = FindFirstFileW(p + L"*.*", &fd);
//    if (this->findHandle == INVALID_HANDLE_VALUE) {
//        throw SystemException(__FILE__, __LINE__);
//    }
//
//    while ((wcscmp(fd.cFileName, L".") == 0) || (wcscmp(fd.cFileName, L"..") == 0)) {
//        if (FindNextFileW(this->findHandle, &fd) == 0) {
//            DWORD le;
//            if ((le = GetLastError()) == ERROR_NO_MORE_FILES) {
//                this->nextItem.Path.Clear();
//            } else {
//                throw SystemException(le, __FILE__, __LINE__);
//            }
//        }
//    }
//
//    if (fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) {
//        this->nextItem.Type = Item::DIRECTORY;
//    } else {
//        this->nextItem.Type = Item::FILE;
//    }
//    this->nextItem.Path = fd.cFileName;
//#else /* _WIN32 */
//    this->DICA = new DirectoryIterator<CharTraitsA>(W2A(path));
//#endif /* _WIN32 */
//}
//
//
///*
// * vislib::sys::DirectoryIterator<vislib::CharTraitsW>::~DirectoryIterator
// */
//vislib::sys::DirectoryIterator<vislib::CharTraitsW>::~DirectoryIterator(void) {
//#ifdef _WIN32
//    if (this->findHandle != INVALID_HANDLE_VALUE) {
//        FindClose(this->findHandle);
//    }
//#else /* _WIN32 */
//    delete this->DICA;
//#endif /* WIN32 */
//}
//
//
///*
// * vislib::sys::DirectoryIterator<vislib::CharTraitsW>::HasNext
// */
//bool vislib::sys::DirectoryIterator<vislib::CharTraitsW>::HasNext() const {
//#ifdef _WIN32
//    return !this->nextItem.Path.IsEmpty();
//#else /* _WIN32 */
//    return this->DICA->HasNext();
//#endif /* _WIN32 */
//}
//
//
///*
// * vislib::sys::DirectoryIterator<vislib::CharTraitsW>::Next
// */
//vislib::sys::DirectoryEntry<vislib::CharTraitsW>&
//vislib::sys::DirectoryIterator<vislib::CharTraitsW>::Next() {
//#ifdef _WIN32
//    this->currentItem = this->nextItem;
//    if (this->currentItem.Path.IsEmpty()) {
//        throw NoSuchElementException("No next element.", __FILE__, __LINE__);
//    } else {
//        WIN32_FIND_DATAW fd;
//        DWORD le;
//
//        do {
//            if (FindNextFileW(this->findHandle, &fd) == 0) {
//                if ((le = GetLastError()) == ERROR_NO_MORE_FILES) {
//                    this->nextItem.Path.Clear();
//                } else {
//                    throw SystemException(le, __FILE__, __LINE__);
//                }
//            } else {
//                if (fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) {
//                    this->nextItem.Type = Item::DIRECTORY;
//                } else {
//                    this->nextItem.Type = Item::FILE;
//                }
//                this->nextItem.Path = fd.cFileName;
//            }
//        } while ((wcscmp(fd.cFileName, L".") == 0) || (wcscmp(fd.cFileName, L"..") == 0));
//        return this->currentItem;
//    }
//#else /* _WIN32 */
//    DirectoryEntry<CharTraitsA> deTmp = this->DICA->Next();
//    this->currentItem.Path = StringW(deTmp.Path);
//    this->currentItem.Type = static_cast<DirectoryEntry<CharTraitsW>::EntryType>(deTmp.Type);
//    return this->currentItem;
//#endif /* _WIN32 */
//} 
//
//// END OF PARTIAL TEMPLATE SPECIALISATION FOR CharTraitsW
//////////////////////////////////////////////////////////////////////////////////
