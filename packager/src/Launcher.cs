using System;
using System.Diagnostics;
using System.IO;
using System.Windows.Forms;
using System.Collections.Generic;

namespace MiRAGLauncher
{
    static class Program
    {
        [STAThread]
        static void Main(string[] args)
        {
            string baseDir = AppDomain.CurrentDomain.BaseDirectory;
            string userProfile = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
            
            // 1. Locate server.py
            string serverScript = Path.Combine(baseDir, "server.py");
            if (!File.Exists(serverScript)) {
                serverScript = Path.Combine(baseDir, "run_factory.py");
            }
            if (!File.Exists(serverScript)) {
                MessageBox.Show("Application server script not found in:\n" + baseDir, "Mi:RAG Assistant", MessageBoxButtons.OK, MessageBoxIcon.Error);
                return;
            }

            // 2. Discover candidate Python interpreters
            List<string> candidatePythons = new List<string>();

            // (a) Local .venv in current application directory
            candidatePythons.Add(Path.Combine(baseDir, ".venv", "Scripts", "python.exe"));

            // (b) User's workspace .venv paths
            candidatePythons.Add(Path.Combine(userProfile, "Mi-RAG", ".venv", "Scripts", "python.exe"));
            candidatePythons.Add(@"C:\Users\aryan\Mi-RAG\.venv\Scripts\python.exe");

            // (c) Walk up parent directories to find any local .venv
            try {
                DirectoryInfo parent = Directory.GetParent(baseDir);
                while (parent != null) {
                    string testVenv = Path.Combine(parent.FullName, ".venv", "Scripts", "python.exe");
                    if (File.Exists(testVenv)) {
                        candidatePythons.Add(testVenv);
                        break;
                    }
                    parent = parent.Parent;
                }
            } catch {}

            // (d) Standard Python install paths
            string localApp = Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData);
            candidatePythons.Add(Path.Combine(localApp, "Programs", "Python", "Python313", "python.exe"));
            candidatePythons.Add(Path.Combine(localApp, "Programs", "Python", "Python312", "python.exe"));
            candidatePythons.Add(Path.Combine(localApp, "Programs", "Python", "Python311", "python.exe"));
            candidatePythons.Add("python.exe");

            // 3. Test candidates to find an environment with required packages
            string workingPython = null;

            foreach (string py in candidatePythons) {
                if (string.IsNullOrEmpty(py)) continue;
                if (!py.Equals("python.exe", StringComparison.OrdinalIgnoreCase) && !File.Exists(py)) continue;

                if (CanImportRequirements(py, baseDir)) {
                    workingPython = py;
                    break;
                }
            }

            // 4. If a working Python environment is found, launch server.py silently
            // server.py immediately boots the FastAPI server and opens the dedicated desktop window!
            if (workingPython != null) {
                string pyDir = Path.GetDirectoryName(workingPython) ?? "";
                string pyw = Path.Combine(pyDir, "pythonw.exe");
                string exeToRun = File.Exists(pyw) ? pyw : workingPython;

                ProcessStartInfo psi = new ProcessStartInfo();
                psi.FileName = exeToRun;
                psi.Arguments = "\"" + serverScript + "\"";
                psi.WorkingDirectory = baseDir;
                psi.WindowStyle = ProcessWindowStyle.Hidden;
                psi.CreateNoWindow = true;
                psi.UseShellExecute = false;

                try {
                    Process.Start(psi);
                    return;
                } catch (Exception ex) {
                    MessageBox.Show("Failed to launch server:\n" + ex.Message, "Mi:RAG Assistant", MessageBoxButtons.OK, MessageBoxIcon.Warning);
                }
            }

            // 5. If dependencies are missing, run run.bat in a visible window to set up the environment
            string runBat = Path.Combine(baseDir, "run.bat");
            if (File.Exists(runBat)) {
                ProcessStartInfo batInfo = new ProcessStartInfo();
                batInfo.FileName = "cmd.exe";
                batInfo.Arguments = "/c \"" + runBat + "\"";
                batInfo.WorkingDirectory = baseDir;
                batInfo.WindowStyle = ProcessWindowStyle.Normal;
                batInfo.UseShellExecute = true;

                try {
                    Process.Start(batInfo);
                    return;
                } catch {}
            }

            // 6. User feedback if no Python runtime could be used
            MessageBox.Show(
                "Python runtime or dependencies not found.\n\nPlease install Python 3.11+ or run 'run.bat' in the application directory.",
                "Mi:RAG Assistant Setup",
                MessageBoxButtons.OK,
                MessageBoxIcon.Information
            );
        }

        static bool CanImportRequirements(string pythonExe, string workingDir)
        {
            try {
                ProcessStartInfo checkPsi = new ProcessStartInfo();
                checkPsi.FileName = pythonExe;
                checkPsi.Arguments = "-c \"import uvicorn, fastapi, chromadb\"";
                checkPsi.WorkingDirectory = workingDir;
                checkPsi.WindowStyle = ProcessWindowStyle.Hidden;
                checkPsi.CreateNoWindow = true;
                checkPsi.UseShellExecute = false;

                using (Process proc = Process.Start(checkPsi)) {
                    if (proc.WaitForExit(3500)) {
                        return proc.ExitCode == 0;
                    } else {
                        try { proc.Kill(); } catch {}
                    }
                }
            } catch {}
            return false;
        }
    }
}
