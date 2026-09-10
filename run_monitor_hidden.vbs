' CORE Automated Anomaly Monitor - Silent Launcher
' Runs monitor.py in the background with no visible window.
' Output goes to monitor.log in the project folder.

Dim shell
Set shell = CreateObject("WScript.Shell")

Dim cmd
' Resolve the project folder from this script's own location, so the
' launcher keeps working if the folder is moved or renamed.
Dim fso, projectDir
Set fso = CreateObject("Scripting.FileSystemObject")
projectDir = fso.GetParentFolderName(WScript.ScriptFullName)

cmd = """C:\Users\user\anaconda3\python.exe"" """ & projectDir & "\monitor.py"""

' 0 = hidden window, False = don't wait for it to finish
shell.Run cmd, 0, False

Set shell = Nothing