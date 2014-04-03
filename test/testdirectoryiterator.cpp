#include "testdirectoryiterator.h"

#include "vislib/error.h"
#include "vislib/File.h"
#include "vislib/Path.h"
#include "the/exception.h"
#include "the/string.h"
#include "the/text/string_converter.h"
#include "vislib/DirectoryIterator.h"
#include "vislib/Array.h"
#include "vislib/Stack.h"
#include "testhelper.h"
#include "the/system/system_message.h"

#include <iostream>

#ifdef _WIN32
#include <direct.h>
#pragma warning ( disable : 4996 )
#define SNPRINTF _snprintf
#define FORMATINT64 "%I64u"
#else /* _WIN32 */
#include <sys/stat.h>
#include <sys/types.h>
#define SNPRINTF snprintf
#define FORMATINT64 "%lu"
#endif /* _WIN32 */

#define BUF_SIZE 4096

using namespace vislib::sys;
using namespace vislib;

static void createDir(const the::astring& fn) {
#ifdef _WIN32
    _mkdir(fn.c_str());
#else /* _WIN32 */
    //std::cout << "Making directory " << dirs[i] << std::endl;
    if (mkdir(fn.c_str(), S_IXUSR | S_IRUSR | S_IWUSR) != 0) {
        int e = GetLastError();
        std::cout << e << "  " << static_cast<const char *>(the::system::system_message(e)) << std::endl;
    }
#endif /* _WIN32 */
}

static void createDir(const the::wstring& fn) {
#ifdef _WIN32
    _wmkdir(fn.c_str());
#else /* _WIN32 */
    //std::cout << "Making directory " << the::astring(dirs[i]) << std::endl;
    if (mkdir(the::text::string_converter::to_a(fn).c_str(), S_IXUSR | S_IRUSR | S_IWUSR) != 0) {
        int e = GetLastError();
        std::cout << e << "  " << static_cast<const char *>(the::system::system_message(e)) << std::endl;
    }
#endif /* _WIN32 */
}

static void createFile(const the::astring& fn) {
    File f;
    f.Open(fn, File::WRITE_ONLY, File::SHARE_READWRITE, File::CREATE_ONLY);
    f.Close();
}

static void createFile(const the::wstring& fn) {
    File f;
    f.Open(fn, File::WRITE_ONLY, File::SHARE_READWRITE, File::CREATE_ONLY);
    f.Close();
}

static void deleteDir(const the::astring& fn) {
#ifdef _WIN32
    _rmdir(fn.c_str());
#else /* _WIN32 */
    rmdir(fn.c_str());
#endif /* _WIN32 */
}

static void deleteDir(const the::wstring& fn) {
#ifdef _WIN32
    _wrmdir(fn.c_str());
#else /* _WIN32 */
    rmdir(the::text::string_converter::to_a(fn).c_str());
#endif /* _WIN32 */
}

static void deleteFile(const the::astring& fn) {
    File::Delete(fn.c_str());
}

static void deleteFile(const the::wstring& fn) {
    File::Delete(fn.c_str());
}

static void TestDirectoryIteratorA(Array<the::astring> dirs, Array<the::astring> files) {
    int i;
    DirectoryEntry<the::astring> de;
    char buf[BUF_SIZE];
    Stack<the::astring> st;

    try {
        for (i = 0; i < static_cast<int>(dirs.Count()); i++) {
            createDir(dirs[i]);
        }
        for (i = 0; i < static_cast<int>(files.Count()); i++) {
            createFile(files[i]);
        }

        st.Push(dirs[0]);
        i = 0;
        the::astring relPath;
        while (!st.empty()) {
            relPath = st.Pop();
            DirectoryIterator<the::astring> di(relPath.c_str());
            while (di.HasNext()) {
                i++;
                de = di.Next();
                if (de.Type == DirectoryEntry<the::astring>::DIRECTORY) {
                    st.Push(relPath + Path::SEPARATOR_A + de.Path);
                }
            }
        }
        SNPRINTF(buf, static_cast<size_t>(BUF_SIZE - 1), "Found all %d entries", static_cast<int>(dirs.Count() - 1 + files.Count()));
        AssertEqual(buf, static_cast<int>(dirs.Count() - 1 + files.Count()), i);

        for (i = 0; i < static_cast<int>(files.Count()); i++) {
            deleteFile(files[i]);
        }
        for (i = static_cast<int>(dirs.Count() - 1); i >= 0; i--) {
            deleteDir(dirs[i]);
        }

    } catch (the::exception e) {
        std::cout << e.what() << std::endl;
    }
}

static void TestDirectoryIteratorW(Array<the::wstring> dirs, Array<the::wstring> files) {
    int i;
    File f;
    DirectoryEntry<the::wstring> de;
    char buf[BUF_SIZE];
    Stack<the::wstring> st;

    try {
        for (i = 0; i < static_cast<int>(dirs.Count()); i++) {
            createDir(dirs[i]);
        }
        for (i = 0; i < static_cast<int>(files.Count()); i++) {
            createFile(files[i]);
        }

        st.Push(dirs[0]);
        i = 0;
        the::wstring relPath;
        while (!st.empty()) {
            relPath = st.Pop();
            DirectoryIterator<the::wstring> di(relPath.c_str());
            while (di.HasNext()) {
                i++;
                de = di.Next();
                if (de.Type == DirectoryEntry<the::wstring>::DIRECTORY) {
                    st.Push(relPath + Path::SEPARATOR_W + de.Path);
                }
            }
        }
        SNPRINTF(buf, static_cast<size_t>(BUF_SIZE - 1), "Found all %d entries", static_cast<int>(dirs.Count() - 1 + files.Count()));
        AssertEqual(buf, static_cast<int>(dirs.Count() - 1 + files.Count()), i);

        for (i = 0; i < static_cast<int>(files.Count()); i++) {
            deleteFile(files[i]);
        }
        for (i = static_cast<int>(dirs.Count() - 1); i >= 0; i--) {
            deleteDir(dirs[i]);
        }

    } catch (the::exception e) {
        std::cout << e.what() << std::endl;
    }
}

void TestDirectoryGlobbing(const char *path) {
    if (vislib::sys::File::Exists(path)) {
        AssertTrue("Test directory root \"$path\" does not exist", false);
        fprintf(stderr, "Test aborted\n");
        return;
    } else {
        AssertTrue("Test directory root \"$path\" does not exist", true);
    }

    vislib::sys::Path::MakeDirectory(path);

    if (!vislib::sys::File::Exists(path)) {
        AssertTrue("Test directory root \"$path\" does exist", false);
        fprintf(stderr, "Test aborted\n");
        return;
    } else {
        AssertTrue("Test directory root \"$path\" does exist", true);
    }

    the::astring p(path);
    if (!the::text::string_utility::ends_with(p, vislib::sys::Path::SEPARATOR_A)) p += vislib::sys::Path::SEPARATOR_A;

    createFile(p + "hugo.txt");
    createFile(p + "herbert.txt");
    createFile(p + "kevin.txt");
    createDir(p + "heinz.dir");
    createDir(p + "adolf.dir");

    vislib::Array<the::astring> children;

    {
        children.Clear();
        DirectoryIteratorA d1((p + "*.*").c_str(), true, true);
        while (d1.HasNext()) {
            children.Add(d1.Next().Path);
            printf(" Found %s\n", children.Last().c_str());
        }
    }
    AssertTrue("hugo.txt found", children.Contains("hugo.txt"));
    AssertTrue("herbert.txt found", children.Contains("herbert.txt"));
    AssertTrue("kevin.txt found", children.Contains("kevin.txt"));
    AssertTrue("heinz.dir found", children.Contains("heinz.dir"));
    AssertTrue("adolf.dir found", children.Contains("adolf.dir"));
    AssertEqual<size_t>("5 children found", children.Count(), 5);

    {
        children.Clear();
        DirectoryIteratorA d1((p + "*.*").c_str(), true, false);
        while (d1.HasNext()) {
            children.Add(d1.Next().Path);
            printf(" Found %s\n", children.Last().c_str());
        }
    }
    AssertTrue("hugo.txt found", children.Contains("hugo.txt"));
    AssertTrue("herbert.txt found", children.Contains("herbert.txt"));
    AssertTrue("kevin.txt found", children.Contains("kevin.txt"));
    AssertFalse("heinz.txt not found", children.Contains("heinz.dir"));
    AssertFalse("adolf.txt not found", children.Contains("adolf.dir"));
    AssertEqual<size_t>("3 children found", children.Count(), 3);

    {
        children.Clear();
        DirectoryIteratorA d1((p + "h*.*").c_str(), true, true);
        while (d1.HasNext()) {
            children.Add(d1.Next().Path);
            printf(" Found %s\n", children.Last().c_str());
        }
    }
    AssertTrue("hugo.txt found", children.Contains("hugo.txt"));
    AssertTrue("herbert.txt found", children.Contains("herbert.txt"));
    AssertFalse("kevin.txt not found", children.Contains("kevin.txt"));
    AssertTrue("heinz.dir found", children.Contains("heinz.dir"));
    AssertFalse("adolf.dir not found", children.Contains("adolf.dir"));
    AssertEqual<size_t>("3 children found", children.Count(), 3);

    {
        children.Clear();
        DirectoryIteratorA d1((p + "h*.*").c_str(), true, false);
        while (d1.HasNext()) {
            children.Add(d1.Next().Path);
            printf(" Found %s\n", children.Last().c_str());
        }
    }
    AssertTrue("hugo.txt found", children.Contains("hugo.txt"));
    AssertTrue("herbert.txt found", children.Contains("herbert.txt"));
    AssertFalse("kevin.txt not found", children.Contains("kevin.txt"));
    AssertFalse("heinz.txt not found", children.Contains("heinz.dir"));
    AssertFalse("adolf.txt not found", children.Contains("adolf.dir"));
    AssertEqual<size_t>("2 children found", children.Count(), 2);

    vislib::sys::Path::DeleteDirectory(path, true);
    if (vislib::sys::File::Exists(path)) {
        AssertTrue("Test directory root \"$path\" does not exist", false);
        fprintf(stderr, "Test aborted\n");
        return;
    } else {
        AssertTrue("Test directory root \"$path\" does not exist", true);
    }
}

void TestDirectoryIterator(void) {
    Array<the::astring> dirs;
    Array<the::astring> files;
    Array<the::wstring> wdirs;
    Array<the::wstring> wfiles;

    if (vislib::sys::File::Exists("level0")) {
        AssertTrue("Test directory root \"level0\" does not exist", false);
        fprintf(stderr, "Test aborted\n");
        return;
    } else {
        AssertTrue("Test directory root \"level0\" does not exist", true);
    }

    dirs.Append("level0");
    dirs.Append(dirs[0] + Path::SEPARATOR_A + "level1");
    dirs.Append(dirs[1] + Path::SEPARATOR_A + "level2");
    files.Append(dirs[0] + Path::SEPARATOR_A + "level0.0");
    files.Append(dirs[0] + Path::SEPARATOR_A + "level0.1");
    files.Append(dirs[2] + Path::SEPARATOR_A + "level2.0");
    ::TestDirectoryIteratorA(dirs, files);

    if (vislib::sys::File::Exists(L"Überlevel0")) {
        AssertTrue("Test directory root \"{Ue}berlevel0\" does not exist", false);
        fprintf(stderr, "Test aborted\n");
        return;
    } else {
        AssertTrue("Test directory root \"{Ue}berlevel0\" does not exist", true);
    }

    wdirs.Append(L"Überlevel0");
    wdirs.Append(wdirs[0] + Path::SEPARATOR_W + L"Überlevel1");
    wdirs.Append(wdirs[1] + Path::SEPARATOR_W + L"Überlevel2");
    wfiles.Append(wdirs[0] + Path::SEPARATOR_W + L"Überlevel0.0");
    wfiles.Append(wdirs[0] + Path::SEPARATOR_W + L"Überlevel0.1");
    wfiles.Append(wdirs[2] + Path::SEPARATOR_W + L"Überlevel2.0");
    ::TestDirectoryIteratorW(wdirs, wfiles);

    // This now uses vislib::sys::Path, which is odd, since Path uses the DirectoryIterator (without globbing features)
    ::TestDirectoryGlobbing("level0");

    sys::DirectoryIteratorA failDi("C:\\does.not.*.exist.hurbelkurbel", true, false);
    AssertFalse("Inexisting file not found", failDi.HasNext());

}
