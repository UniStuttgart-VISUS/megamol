#include "testdirectoryiterator.h"

#include "vislib/error.h"
#include "vislib/File.h"
#include "vislib/Path.h"
#include "vislib/Exception.h"
#include "vislib/String.h"
#include "vislib/StringConverter.h"
#include "vislib/DirectoryIterator.h"
#include "vislib/Array.h"
#include "vislib/Stack.h"
#include "testhelper.h"
#include "vislib/SystemMessage.h"

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

static void TestDirectoryIteratorA(Array<StringA> dirs, Array<StringA> files) {
	int i;
	File f;
	DirectoryEntry<CharTraitsA> de;
	char buf[BUF_SIZE];
	Stack<StringA> st;

    try {
		for (i = 0; i < static_cast<int>(dirs.Count()); i++) {
#ifdef _WIN32
			_mkdir(dirs[i]);
#else /* _WIN32 */
			//std::cout << "Making directory " << dirs[i] << std::endl;
			if (mkdir(dirs[i], S_IXUSR | S_IRUSR | S_IWUSR) != 0) {
				int e = GetLastError();
				std::cout << e << "  " << static_cast<const char *>(SystemMessage(e)) << std::endl;
			}
#endif /* _WIN32 */
		}
		for (i = 0; i < static_cast<int>(files.Count()); i++) {
			f.Open(files[i], File::WRITE_ONLY, File::SHARE_READWRITE, File::CREATE_ONLY);
			f.Close();
		}

		st.Push(dirs[0]);
		i = 0;
		StringA relPath;
		while (!st.IsEmpty()) {
			DirectoryIterator<CharTraitsA> di(relPath = st.Pop());
			while (di.HasNext()) {
				i++;
				de = di.Next();
				if (de.Type == DirectoryEntry<CharTraitsA>::DIRECTORY) {
					st.Push(relPath + Path::SEPARATOR_A + de.Path);
				}
			}
		}
		SNPRINTF(buf, static_cast<size_t>(BUF_SIZE - 1), "Found all %d entries", static_cast<int>(dirs.Count() - 1 + files.Count()));
		AssertEqual(buf, static_cast<int>(dirs.Count() - 1 + files.Count()), i);

		for (i = 0; i < static_cast<int>(files.Count()); i++) {
			File::Delete(files[i]);
		}
		for (i = static_cast<int>(dirs.Count() - 1); i >= 0; i--) {
#ifdef _WIN32
			_rmdir(dirs[i]);
#else /* _WIN32 */
			rmdir(dirs[i]);
#endif /* _WIN32 */
		}

	} catch (vislib::Exception e) {
        std::cout << e.GetMsgA() << std::endl;
    }
}

static void TestDirectoryIteratorW(Array<StringW> dirs, Array<StringW> files) {
	int i;
	File f;
	DirectoryEntry<CharTraitsW> de;
	char buf[BUF_SIZE];
	Stack<StringW> st;

    try {
		for (i = 0; i < static_cast<int>(dirs.Count()); i++) {
#ifdef _WIN32
			_wmkdir(dirs[i]);
#else /* _WIN32 */
			//std::cout << "Making directory " << StringA(dirs[i]) << std::endl;
			if (mkdir(StringA(dirs[i]), S_IXUSR | S_IRUSR | S_IWUSR) != 0) {
				int e = GetLastError();
				std::cout << e << "  " << static_cast<const char *>(SystemMessage(e)) << std::endl;
			}
#endif /* _WIN32 */
		}
		for (i = 0; i < static_cast<int>(files.Count()); i++) {
			f.Open(files[i], File::WRITE_ONLY, File::SHARE_READWRITE, File::CREATE_ONLY);
			f.Close();
		}

		st.Push(dirs[0]);
		i = 0;
		StringW relPath;
		while (!st.IsEmpty()) {
			relPath = st.Pop();
			DirectoryIterator<CharTraitsW> di(relPath);
			while (di.HasNext()) {
				i++;
				de = di.Next();
				if (de.Type == DirectoryEntry<CharTraitsW>::DIRECTORY) {
					st.Push(relPath + Path::SEPARATOR_W + de.Path);
				}
			}
		}
		SNPRINTF(buf, static_cast<size_t>(BUF_SIZE - 1), "Found all %d entries", static_cast<int>(dirs.Count() - 1 + files.Count()));
		AssertEqual(buf, static_cast<int>(dirs.Count() - 1 + files.Count()), i);

		for (i = 0; i < static_cast<int>(files.Count()); i++) {
			File::Delete(files[i]);
		}
		for (i = static_cast<int>(dirs.Count() - 1); i >= 0; i--) {
#ifdef _WIN32
			_wrmdir(dirs[i]);
#else /* _WIN32 */
			rmdir(StringA(dirs[i]));
#endif /* _WIN32 */
		}

	} catch (vislib::Exception e) {
        std::cout << e.GetMsgA() << std::endl;
    }
}

void TestDirectoryIterator(void) {
	Array<StringA> dirs;
	Array<StringA> files;
	dirs.Append("level0");
	dirs.Append(dirs[0] + Path::SEPARATOR_A + "level1");
	dirs.Append(dirs[1] + Path::SEPARATOR_A + "level2");
	files.Append(dirs[0] + Path::SEPARATOR_A + "level0.0");
	files.Append(dirs[0] + Path::SEPARATOR_A + "level0.1");
	files.Append(dirs[2] + Path::SEPARATOR_A + "level2.0");
	::TestDirectoryIteratorA(dirs, files);

	Array<StringW> wdirs;
	Array<StringW> wfiles;
	wdirs.Append(L"Überlevel0");
	wdirs.Append(wdirs[0] + Path::SEPARATOR_W + L"Überlevel1");
	wdirs.Append(wdirs[1] + Path::SEPARATOR_W + L"Überlevel2");
	wfiles.Append(wdirs[0] + Path::SEPARATOR_W + L"Überlevel0.0");
	wfiles.Append(wdirs[0] + Path::SEPARATOR_W + L"Überlevel0.1");
	wfiles.Append(wdirs[2] + Path::SEPARATOR_W + L"Überlevel2.0");

	::TestDirectoryIteratorW(wdirs, wfiles);
}
