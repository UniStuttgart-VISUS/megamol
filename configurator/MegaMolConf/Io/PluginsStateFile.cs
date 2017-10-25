using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.IO;
using System.Linq;
using System.Text;
using System.Xml.Serialization;

namespace MegaMolConf.Io {

    [XmlRoot("MegaMolConfiguratorState")]
    public class PluginsStateFile {

        public class SerializableVersion {
            [XmlAttribute(), DefaultValue(-1)]
            public int Major { get; set; }
            [XmlAttribute(), DefaultValue(-1)]
            public int Minor { get; set; }
            [XmlAttribute(), DefaultValue(-1)]
            public int Build { get; set; }
            [XmlAttribute(), DefaultValue(-1)]
            public int Revision { get; set; }

            [XmlAttribute()]
            public string Version { get; set; }

            [EditorBrowsable(EditorBrowsableState.Never)]
            public bool ShouldSerializeMajor() { return false; }

            [EditorBrowsable(EditorBrowsableState.Never)]
            public bool ShouldSerializeMinor() { return false; }

            [EditorBrowsable(EditorBrowsableState.Never)]
            public bool ShouldSerializeBuild() { return false; }

            [EditorBrowsable(EditorBrowsableState.Never)]
            public bool ShouldSerializeRevision() { return false; }

            public SerializableVersion() {
                Major = -1;
                Minor = -1;
                Build = -1;
                Revision = -1;
                Version = null;
            }
            public SerializableVersion(Version v) {
                Major = v.Major;
                Minor = v.Minor;
                Build = v.Build;
                Revision = v.Revision;
                Version = v.ToString();
            }
            public static implicit operator SerializableVersion(Version v) {
                return new SerializableVersion(v);
            }
            public static implicit operator Version(SerializableVersion v) {
                if (!String.IsNullOrWhiteSpace(v.Version)) {
                    System.Version vv;
                    if (System.Version.TryParse(v.Version, out vv)) {
                        return vv;
                    }
                }
                if (v.Major < 0) return new Version();
                if (v.Minor < 0) return new Version(v.Major.ToString());
                if (v.Build < 0) return new Version(v.Major, v.Minor);
                if (v.Revision < 0) return new Version(v.Major, v.Minor, v.Build);
                return new Version(v.Major, v.Minor, v.Build, v.Revision);
            }
        };

        public static Version MaxVersion {
            get {
                return new Version(1, 0);
            }
        }
        [XmlElement(Type = typeof(SerializableVersion))]
        public Version Version { get; set; }
        public Data.PluginFile[] Plugins { get; set; }
        public PluginsStateFile() {
            Version = MaxVersion;
            Plugins = null;
        }
        public void Save(string path) {
            if (Version < new Version(1, 0)) save_v0_1(path);
            else if (Version == new Version(1, 0)) save_v1_0(path);
            else throw new ArgumentException("Unsupproted version number");
        }
        private void save_v0_1(string path) {
            XmlSerializer s = new XmlSerializer(typeof(List<Data.PluginFile>));
            using (StreamWriter sw = new StreamWriter(path)) {
                s.Serialize(sw, new List<Data.PluginFile>(Plugins));
            }
        }
        private void save_v1_0(string path) {
            XmlSerializer s = new XmlSerializer(typeof(PluginsStateFile));
            using (StreamWriter sw = new StreamWriter(path)) {
                s.Serialize(sw, this);
            }
        }

        public void Load(string path) {
            // currently brute force!

            try {
                // v0.1
                XmlSerializer s = new XmlSerializer(typeof(List<Data.PluginFile>));
                using (StreamReader sr = new StreamReader(path)) {
                    Plugins = ((List<Data.PluginFile>)s.Deserialize(sr)).ToArray();
                }
                return;
            } catch {
                Plugins = null;
            }

            try {
                // v1.0
                XmlSerializer s = new XmlSerializer(typeof(PluginsStateFile));
                using (StreamReader sr = new StreamReader(path)) {
                    PluginsStateFile psf = (PluginsStateFile)s.Deserialize(sr);
                    Plugins = psf.Plugins;
                    Version = psf.Version;
                }
                return;

            } catch {
                Plugins = null;
                throw; // because I say so
            }
        }
    }

}
