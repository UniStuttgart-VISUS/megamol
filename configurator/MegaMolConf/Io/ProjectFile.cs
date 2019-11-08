using System;
using System.Drawing;
using System.IO;
using System.Windows.Forms;

namespace MegaMolConf.Io {

    public abstract class ProjectFile {

        public class Param {
            public string Name { get; set; }
            public string Value { get; set; }
        }

        public class Module {
            public string Name { get; set; }
            public string Class { get; set; }
            public Point ConfPos { get; set; }
            public Param[] Params { get; set; }
        }

        public class Call {
            public string Class { get; set; }
            public Module FromModule { get; set; }
            public string FromSlot { get; set; }
            public Module ToModule { get; set; }
            public string ToSlot { get; set; }
        }

        public class View {
            public string Name { get; set; }
            public Module ViewModule { get; set; }
            public Module[] Modules { get; set; }
            public Param[] Params { get; set; }
            public Call[] Calls { get; set; }
        }

        public string GeneratorComment { get; set; }
        public string StartComment { get; set; }
        public View[] Views { get; set; }

        public static ProjectFile Load(string filename) {

            string ext = Path.GetExtension(filename);
            if (ext == ".lua") {
                // TODO: check for various elements. see various checks a few lines below for xmldocument
                ProjectFileLua l = new ProjectFileLua();
                l.LoadFromLua(filename);
                
                //MessageBox.Show("Loading Lua projects is not implemented yet!");
                return l;
            } else {

                // Load as ProjectFile1

                System.Xml.XmlDocument doc = new System.Xml.XmlDocument();
                doc.Load(filename);

                if (doc.DocumentElement.Name != "MegaMol") {
                    throw new Exception("Document root element must be \"MegaMol\"");
                }

                if (!doc.DocumentElement.HasAttribute("type")) {
                    throw new Exception("Root element must specify type");
                }

                if (doc.DocumentElement.Attributes["type"].Value != "project") {
                    throw new Exception("Document type must be project");
                }

                if (!doc.DocumentElement.HasAttribute("version")) {
                    throw new Exception("Root element must specify version");
                }

                if (new Version(doc.DocumentElement.Attributes["version"].Value) != new Version(1, 0)) {
                    throw new Exception("Document version must be 1.0");
                }

                ProjectFile1 p = new ProjectFile1();
                p.LoadFromXml(doc);

                return p;
            }
        }

        public abstract void Save(string filename);

    }

}
