using System;
using System.Collections.Generic;
using System.Text;
using System.Linq;
using System.Diagnostics;

namespace MegaMolConf.Analyze {

    public class Inheritance {
        public string name;
        public Module parent = null;
        public bool Resolved {
            get {
                return parent != null && parent.Resolved;
            }
        }
    }

    [DebuggerDisplay("{Type} {Name}")]
    public class ParameterSlot : IEquatable<ParameterSlot> {
        public string Name { get; set; }
        public string Type { get; set; }
        public string ExportedName { get; set; }
        public string Description { get; set; }

        public bool Equals(ParameterSlot other) {
            return Name.Equals(other.Name)
                && Type.Equals(other.Type)
                && ExportedName.Equals(other.ExportedName)
                && Description.Equals(other.Description);
        }
    }

    [DebuggerDisplay("CalleeSlot {Name}")]
    public class CalleeSlot : IEquatable<CalleeSlot> {
        public string Name { get; set; }
        public string ExportedName { get; set; }
        public string Description { get; set; }
        public List<string> Callbacks { get; set; }

        public CalleeSlot() {
            this.Callbacks = new List<string>();
        }

        public bool Equals(CalleeSlot other) {
            //return Name.Equals(other.Name)
            //    && ExportedName.Equals(other.ExportedName)
            //    && Description.Equals(other.Description);
            return this.ExportedName.Equals(other.ExportedName);
        }
    }

    [DebuggerDisplay("CallerSlot {Name}")]
    public class CallerSlot : IEquatable<CallerSlot> {
        public string Name { get; set; }
        public string ExportedName { get; set; }
        public string Description { get; set; }
        public List<Call> CompatibleCalls { get; set; }

        public CallerSlot() {
            this.CompatibleCalls = new List<Call>();
        }

        public bool Equals(CallerSlot other) {
            //return Name.Equals(other.Name)
            //    && ExportedName.Equals(other.ExportedName)
            //    && Description.Equals(other.Description);
            return this.ExportedName.Equals(other.ExportedName);
        }
    }

    [DebuggerDisplay("Call {ExportedName}")]
    public class Call : IEquatable<Call> {
        public string ClassName { get; set; }
        public string ExportedName { get; set; }
        public string Description { get; set; }
        public string SourceFile { get; set; }
        public List<string> Functions { get; set; }

        public Call() {
            this.Functions = new List<string>();
        }

        public bool Equals(Call other) {
            // this works same as MegaMol does: the exported Name is enough
            // and treated as unique
            return this.ExportedName.Equals(other.ExportedName);
        }
    }

    [DebuggerDisplay("Module {ExportedName}")]
    public class Module : IEquatable<Module> {
        //module parent;
        List<Inheritance> inherited;

        public void AddParameter(string name, string type, string exportedName, string description) {
            ParameterSlot p = new ParameterSlot() {
                Name = name,
                Type = type,
                ExportedName = exportedName,
                Description = description
            };
            if (!this.ParamSlots.Contains(p)) {
                this.ParamSlots.Add(p);
            }
        }

        //public void AddCaller(string name, string type, string exportedName, string description) {
        //    CallSlot c = new CallSlot() {
        //        Name = name,
        //        Type = type,
        //        ExportedName = exportedName,
        //        Description = description
        //    };
        //    if (!this.CallerSlots.Contains(c)) {
        //        this.CallerSlots.Add(c);
        //    }
        //}

        public List<ParameterSlot> Parameters {
            get {
                return ParamSlots;
            }
        }

        public List<CallerSlot> Callers {
            get {
                return CallerSlots;
            }
        }

        public List<CalleeSlot> Callees {
            get {
                return CalleeSlots;
            }
        }

        public List<Inheritance> Inherited {
            get { return inherited; }
            set { inherited = value; }
        }

        public string ClassName { get; set; }
        public string ExportedName { get; set; }
        public string SourceFile { get; set; }
        public string Description { get; set; }

        public List<CallerSlot> CallerSlots;
        public List<CalleeSlot> CalleeSlots;
        public List<ParameterSlot> ParamSlots;

        public bool Resolved {
            get {
                if (this.Inherited.Count == 0) {
                    return true;
                }
                bool res = true;
                foreach (Inheritance i in this.Inherited) {
                    if (!i.Resolved) {
                        res = false;
                        break;
                    }
                }
                return res;
            }
        }

        public string FullyResolvedName {
            get {
                StringBuilder sb = new StringBuilder();
                resolve(ref sb, this);
                return sb.ToString();
            }
        }

        public bool IsA(string baseClass) {
            List<Module> toDos = new List<Module>();
            toDos.Add(this);
            while (toDos.Count > 0) {
                Module n = toDos.First();
                toDos.RemoveAt(0);
                if (ClassMatches(n.ClassName, baseClass)) {
                    //res.Add(m.TheName + " (" + m.SourceFile + ")");
                    return true;
                }
                foreach (Inheritance i in n.Inherited) {
                    if (i.parent != null) {
                        toDos.Add(i.parent);
                    }
                }
            }
            return false;
        }

        private static void resolve(ref StringBuilder sb, Module m) {
            sb.Append(m.ClassName);
            if (m.Inherited.Count > 0) {
                sb.Append(": (");
            }
            for (int x = 0; x < m.Inherited.Count; x++) {
                Inheritance i = m.Inherited[x];
                if (i.parent != null) {
                    resolve(ref sb, i.parent);
                } else {
                    sb.Append(i.name + "[unknown] ");
                }
                if (x != m.Inherited.Count - 1) {
                    sb.Append(", ");
                } else {
                    sb.Append(") ");
                }
            }
        }

        public static bool ClassMatches(string theClass, string sought) {
            int idx = theClass.LastIndexOf(sought);
            if (idx == -1) {
                return false;
            } else if (idx == 0 || theClass[idx - 1] == ':') {
                if (theClass.EndsWith(sought)) {
                    return true;
                }
            }
            return false;
        }

        public Module() {
            this.inherited = new List<Inheritance>();
            this.CalleeSlots = new List<CalleeSlot>();
            this.CallerSlots = new List<CallerSlot>();
            this.ParamSlots = new List<ParameterSlot>();
        }

        public bool Equals(Module other) {
            // this works same as MegaMol does: the exported Name is enough
            // and treated as unique
            if (this.ExportedName != null & other.ExportedName != null) {
                return this.ExportedName.Equals(other.ExportedName);
            } else {
                // we cannot be sure actually
                return false;
            }
            // with all the branches, this does not work:
            //&& this.sourceFile.Equals(other.sourceFile);
        }
    }

}
