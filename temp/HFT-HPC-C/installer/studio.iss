; Optional Inno Setup recipe. Build/test first, then run ISCC on this file.
[Setup]
AppName=HFT HPC Studio C
AppVersion=1.0.0
DefaultDirName={localappdata}\HFT-HPC-C
DefaultGroupName=HFT HPC Studio C
OutputBaseFilename=HFT-HPC-C-Setup
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
Compression=lzma2
SolidCompression=yes
[Files]
Source: "..\windows\*"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\docs\*"; DestDir: "{app}\docs"; Flags: recursesubdirs createallsubdirs
Source: "..\configs\*"; DestDir: "{app}\configs"
Source: "..\README.md"; DestDir: "{app}"
Source: "..\LICENSE"; DestDir: "{app}"
[Icons]
Name: "{group}\HFT HPC Studio C"; Filename: "{app}\hft_studio.exe"; WorkingDir: "{app}"
