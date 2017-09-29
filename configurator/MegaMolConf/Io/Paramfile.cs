using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace MegaMolConf.Io {

    // MegaMol parameter file
    internal class Paramfile {

        public void Load(string filename) {
            string[] lines = System.IO.File.ReadAllLines(filename);
            List<KeyValuePair<string[], string>> paramsettings = new List<KeyValuePair<string[], string>>();
            bool hasIllegalLines = false;
            foreach (string line in lines) {
                if (string.IsNullOrWhiteSpace(line)) continue;
                string l = line.Trim();
                if (l[0] == '#') continue;
                if (l.Count(new Func<char, bool>((char c) => { return c == '='; })) >= 1) {
                    string[] e = l.Split(new char[] { '=' }, 2, StringSplitOptions.None);
                    System.Diagnostics.Debug.Assert(e.Length == 2);
                    paramsettings.Add(
                        new KeyValuePair<string[], string>(
                            e[0].Split(new string[]{"::"}, StringSplitOptions.RemoveEmptyEntries),
                            e[1].Trim()));
                } else {
                    hasIllegalLines = true;
                }
            }

            if (paramsettings.Count <= 0) {
                if (hasIllegalLines) {
                    throw new Exception("File seems not to be a valid MegaMol parameter file.");
                } else {
                    throw new Exception("File seems to be an empty MegaMol parameter file.");
                }
            }

            parSetPairs = paramsettings;
        }

        private List<KeyValuePair<string[], string>> parSetPairs = null;

        public bool Empty {
            get {
                return (parSetPairs == null) || parSetPairs.Count == 0;
            }
        }
        public int Count {
            get {
                if (parSetPairs == null) return 0;
                return parSetPairs.Count;
            }
        }

        public KeyValuePair<string[], string> this[int idx] {
            get {
                return parSetPairs[idx];
            }
        }

    }

}
