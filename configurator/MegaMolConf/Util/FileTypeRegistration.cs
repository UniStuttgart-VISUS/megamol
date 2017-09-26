using Microsoft.Win32;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Windows.Forms;

namespace MegaMolConf.Util {

    static internal class FileTypeRegistration {

        private class Report {
            public StringBuilder Text = new StringBuilder();
            public bool HasErrors = false;

            public void AppendText(string text) {
                if (Text.Length > 0) Text.Append(Environment.NewLine);
                Text.Append(text);
            }
            public void AppendError(string text) {
                if (Text.Length > 0) Text.Append(Environment.NewLine);
                Text.Append(text);
                HasErrors = true;
            }

            public void ShowMessageBox(IWin32Window owner) {
                MessageBox.Show(owner, Text.ToString(), Application.ProductName, MessageBoxButtons.OK, HasErrors ? MessageBoxIcon.Error : MessageBoxIcon.Information);
            }
        }

        static public void Register(IWin32Window owner, string exePath) {
            Report rep = new Report();

            registerFileType(rep, ".mmprj", "MegaMol.Project",
                (RegistryKey key) => {
                    key.SetValue(null, "MegaMol™ Project File");
                    setDefaultIcon(key, "\"" + exePath + "\",-2");

                    RegistryKey shell_key = key.CreateSubKey("shell");
                    setShellCommand(shell_key, "open", "\"" + exePath + "\" \"%1\"");
                    shell_key.Close();
                });
            registerFileType(rep, ".mmplg", "MegaMol.Plugin",
                (RegistryKey key) => {
                    key.SetValue(null, "MegaMol™ Plugin");
                    setDefaultIcon(key, "\"" + exePath + "\",-3");
                });

            rep.ShowMessageBox(owner);
        }

        static public void Unregister(IWin32Window owner, string exePath) {
            Report rep = new Report();

            unregisterFileType(rep, ".mmprj", "MegaMol.Project");
            unregisterFileType(rep, ".mmplg", "MegaMol.Plugin");

            rep.ShowMessageBox(owner);
        }

        static public void RegisterDataFileType(IWin32Window owner, string ext, string desc, string iconPath, string openCmd) {
            Report rep = new Report();

            registerFileType(rep, ext, "MegaMol" + ext,
                (RegistryKey key) => {
                    key.SetValue(null, desc);
                    if (!String.IsNullOrWhiteSpace(iconPath)) setDefaultIcon(key, iconPath);
                    if (!String.IsNullOrWhiteSpace(openCmd)) {
                        RegistryKey shell_key = key.CreateSubKey("shell");
                        setShellCommand(shell_key, "open", openCmd);
                        shell_key.Close();
                    }
                });

            rep.ShowMessageBox(owner);
        }

        static public void UnregisterDataFileType(IWin32Window owner, string ext) {
            Report rep = new Report();
            unregisterFileType(rep, ext, "MegaMol" + ext);
            rep.ShowMessageBox(owner);
        }

        static private void setDefaultIcon(RegistryKey key, string iconPath) {
            RegistryKey icon_key = key.CreateSubKey("DefaultIcon");
            icon_key.SetValue(null, iconPath);
            icon_key.Close();
        }

        static private void setShellCommand(RegistryKey shell_key, string commandVerb, string commandLine) {
            RegistryKey verb_shell_key = shell_key.CreateSubKey(commandVerb);
            RegistryKey cmd_verb_shell_key = verb_shell_key.CreateSubKey("command");
            cmd_verb_shell_key.SetValue(null, commandLine);
            cmd_verb_shell_key.Close();
            verb_shell_key.Close();
        }

        /// <summary>
        /// Registers a file type in the windows registry
        /// </summary>
        /// <param name="ext">File name extension including the leading period</param>
        /// <param name="name">File type name (main registry key name)</param>
        /// <param name="setupFunc">Function called to set up keys and values within the main registry key for this file type</param>
        static private void registerFileType(Report report, string ext, string name, Action<RegistryKey> setupFunc) {
            try {
                RegistryKey ext_hcsi = Registry.ClassesRoot.CreateSubKey(ext);
                ext_hcsi.SetValue(null, name);
                ext_hcsi.Close();
                RegistryKey desc_hcsi = Registry.ClassesRoot.CreateSubKey(name);

                setupFunc(desc_hcsi);

                desc_hcsi.Close();

                report.AppendText("File type *" + ext + " registered.");
            } catch (Exception ex) {
                report.AppendError("Failed to register *" + ext + " file type:\n" + ex.ToString());
            }
        }

        /// <summary>
        /// Removes a file type registration
        /// </summary>
        /// <param name="ext">File name extension including the leading period</param>
        /// <param name="name">File type name (main registry key name)</param>
        static private void unregisterFileType(Report report, string ext, string name) {
            try {
                Registry.ClassesRoot.DeleteSubKeyTree(ext, false);
                Registry.ClassesRoot.DeleteSubKeyTree(name, false);
                report.AppendText("File type *" + ext + " unregistered.");
            } catch (Exception ex) {
                report.AppendError("Failed to unregister *" + ext + " file type:\n" + ex.ToString());
            }
        }

    }
}
