using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;

namespace MegaMolConf.Analyze {

    class AnalyzerData {
        private static List<Module> knownModules = new List<Module>();
        private static List<Call> knownCalls = new List<Call>();

        public static void AddModule(Module mod) {
            if (!knownModules.Contains(mod)) {
                knownModules.Add(mod);
            }
        }

        public static void AddCall(Call call) {
            if (!knownCalls.Contains(call)) {
                knownCalls.Add(call);
            }
        }

        public static int CallCount {
            get {
                return knownCalls.Count;
            }
        }

        public static void ClearCalls() {
            knownCalls.Clear();
        }

        public static void ClearModules() {
            knownModules.Clear();
        }

        public static List<string> GetClassesOfType(string baseClass) {
            List<string> res = new List<string>();
            foreach (Module m in knownModules) {
                if (m.IsA(baseClass)) {
                    res.Add(m.ClassName + " (" + m.SourceFile + ")");
                }
            }
            return res;
        }

        public static Call GetCallByClassName(string name) {
            Call ret = null;
            foreach (Call tmpCall in knownCalls) {
                if (tmpCall.ClassName != null && tmpCall.ClassName.Equals(name)) {
                    ret = tmpCall;
                    break;
                }
            }
            return ret;
        }

        public static Module GetModuleByClassName(string name) {
            Module ret = null;
            foreach (Module tmpMod in knownModules) {
                if (tmpMod.ClassName != null && tmpMod.ClassName.Equals(name)) {
                    ret = tmpMod;
                    break;
                }
            }
            return ret;
        }

        public static Call GetCallByExportedName(string name) {
            Call ret = null;
            foreach (Call tmpCall in knownCalls) {
                if (tmpCall.ExportedName.Equals(name)) {
                    ret = tmpCall;
                    break;
                }
            }
            return ret;
        }
        
        public static Module GetModuleByExportedName(string name) {
            Module ret = null;
            foreach (Module tmpMod in knownModules) {
                if (tmpMod.ExportedName != null && tmpMod.ExportedName.Equals(name)) {
                    ret = tmpMod;
                    break;
                }
            }
            return ret;
        }

        public static int ModuleCount {
            get {
                return knownModules.Count;
            }
        }

        public static void Cleanup() {
            for (int x = knownModules.Count - 1; x >= 0; x--) {
                if (knownModules[x].ClassName == null) {
                    Debug.WriteLine("removing module " + knownModules[x].ExportedName + " because it has no matched understood class");
                    knownModules.RemoveAt(x);
                }
            }
        }

        public static void Resolve() {
            bool done = true;
            //List<Module> lm = this.knownModules.FindAll(x => x.TheName.EndsWith("Observer"));
            int iter = 0;
            do {
                foreach (Module m in knownModules) {
                    if (!m.Resolved) {
                        foreach (Inheritance i in m.Inherited) {
                            if (i.parent == null) {
                                foreach (Module n in knownModules) {
                                    if (Module.ClassMatches(n.ClassName, i.name)) {
                                        i.parent = n;
                                        break;
                                    }
                                }
                            }
                        }
                    }
                    done = done && m.Resolved;
                    iter++;
                }
            } while (!done && iter < 100);
            foreach (Module m in knownModules) {
                if (!m.Resolved) {
                    Debug.WriteLine("Cannot complete resolution of " + m.ClassName);
                    Debug.WriteLine("Resolution so far: " + m.FullyResolvedName);
                }
            }
        }
    }
}
