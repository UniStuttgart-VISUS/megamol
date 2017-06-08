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
            DWORD le = ::GetLastError();
            if ((le == ERROR_FILE_NOT_FOUND) || (le == ERROR_PATH_NOT_FOUND)) {
                this->nextItem.Path.Clear();
            } else {
                throw SystemException(le, __FILE__, __LINE__);
            }
        } else if (fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) {
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
        if (isPattern) {
            this->basePath = vislib::sys::Path::GetDirectoryName(path);
            this->pattern = path + (this->basePath.Length() + 1);

        } else {
            this->basePath = path;
            this->pattern.Clear();

        }
        if (vislib::sys::File::Exists(this->basePath)) {
            if ((this->dirStream = opendir(this->basePath)) == NULL) {
                throw SystemException(__FILE__, __LINE__);
            }
            this->fetchNextItem();
        } else {
            this->dirStream = NULL;
            this->nextItem.Path.Clear();
        }
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
            DWORD le = ::GetLastError();
            if ((le == ERROR_FILE_NOT_FOUND) || (le == ERROR_PATH_NOT_FOUND)) {
                this->nextItem.Path.Clear();
            } else {
                throw SystemException(le, __FILE__, __LINE__);
            }
        } else if (fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) {
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
        if (isPattern) {
            this->basePath = vislib::sys::Path::GetDirectoryName(path);
            this->pattern = path + (this->basePath.Length() + 1);

        } else {
            this->basePath = path;
            this->pattern.Clear();

        }
        if (vislib::sys::File::Exists(this->basePath)) {
            if ((this->dirStream = opendir(this->basePath)) == NULL) {
                throw SystemException(__FILE__, __LINE__);
            }
            this->fetchNextItem();
        } else {
            this->dirStream = NULL;
            this->nextItem.Path.Clear();
        }
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
            if (this->findHandle == INVALID_HANDLE_VALUE) {
                this->nextItem.Path.Clear();
            } else if (FindNextFileA(this->findHandle, &fd) == 0) {
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
        struct dirent *de = NULL;
        if (this->dirStream != NULL) {
            do {
                // BUG: Linux documentation is all lies. the errno stunt does not work at all.
                //errno = 0;
                if ((de = readdir(this->dirStream)) != NULL) {
                    if (!this->pattern.IsEmpty()) {
                        if (!vislib::sys::FilenameGlobMatch(de->d_name, this->pattern.PeekBuffer())) {
                            continue; // one more time
                        }
                    }
                    if (vislib::sys::File::IsDirectory(this->basePath + Path::SEPARATOR_A + de->d_name)) {
                        this->nextItem.Type = Entry::DIRECTORY;
                        if (this->omitFolders) continue; // one more time
                        if ((strcmp(de->d_name, "..") != 0) && (strcmp(de->d_name, ".") != 0)) break;
                    } else {
                        this->nextItem.Type = Entry::FILE;
                        break;
                    }
                }
            } while (de != NULL);
        }
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
            if (this->findHandle == INVALID_HANDLE_VALUE) {
                this->nextItem.Path.Clear();
            } else if (FindNextFileW(this->findHandle, &fd) == 0) {
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
        struct dirent *de = NULL;
        if (this->dirStream != NULL) {
            do {
                // BUG: Linux documentation is all lies. the errno stunt does not work at all.
                //errno = 0;
                if ((de = readdir(this->dirStream)) != NULL) {
                    if (!this->pattern.IsEmpty()) {
                        if (!vislib::sys::FilenameGlobMatch(de->d_name, this->pattern.PeekBuffer())) {
                            continue; // one more time
                        }
                    }
                    if (vislib::sys::File::IsDirectory(this->basePath + Path::SEPARATOR_A + de->d_name)) {
                        this->nextItem.Type = Entry::DIRECTORY;
                        if (this->omitFolders) continue; // one more time
                        if ((strcmp(de->d_name, "..") != 0) && (strcmp(de->d_name, ".") != 0)) break;
                    } else {
                        this->nextItem.Type = Entry::FILE;
                        break;
                    }
                }
            } while (de != NULL);
        }
        if (de == NULL) {
            this->nextItem.Path.Clear();
        } else {
            this->nextItem.Path = vislib::StringW(de->d_name);
        }
#endif /* _WIN32 */
    }


} /* end namespace sys */
} /* end namespace vislib */
