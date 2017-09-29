using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace MegaMolConf.Analyze {

    public class FileFinder {

        private static List<String> findStuff(System.IO.DirectoryInfo di, string extension) {
            List<String> res = new List<string>();
            System.IO.FileInfo[] files = null;
            System.IO.DirectoryInfo[] subDirs = null;
            files = di.GetFiles(extension);
            foreach (System.IO.FileInfo fi in files) {
                res.Add(fi.FullName);
            }
            subDirs = di.GetDirectories();
            foreach (System.IO.DirectoryInfo sdi in subDirs) {
                res.AddRange(findStuff(sdi, extension));
            }
            return res;
        }
        
        public static List<String> FindFiles(string path, string extension) {
            List<String> theHeaders = new List<string>();
            theHeaders = findStuff(new System.IO.DirectoryInfo(path), extension);
            return theHeaders;
        }
    }
}
