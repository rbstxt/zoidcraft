; Inno Setup script for ZoidCraft Windows x86_64 Installer
; Invoked via: iscc /DAppVersion=0.2.3 /DSourceDir=... /DOutputDir=... /DOutputBaseName=ZoidCraft-Windows-x86_64-Setup packaging/windows/zoidcraft.iss

#ifndef AppVersion
  #define AppVersion "0.2.3"
#endif
#ifndef SourceDir
  #define SourceDir "..\..\target\x86_64-pc-windows-msvc\release"
#endif
#ifndef OutputDir
  #define OutputDir "..\..\dist"
#endif
#ifndef OutputBaseName
  #define OutputBaseName "ZoidCraft-Windows-x86_64-Setup"
#endif

[Setup]
AppId={{8D6E7B42-3C8A-4D2B-9A4D-7A82E5F9B301}
AppName=ZoidCraft
AppVersion={#AppVersion}
AppVerName=ZoidCraft {#AppVersion}
AppPublisher=ZoidCraft Contributors
AppPublisherURL=https://github.com/rbstxt/zoidcraft
AppSupportURL=https://github.com/rbstxt/zoidcraft/issues
DefaultDirName={autopf}\ZoidCraft
DefaultGroupName=ZoidCraft
AllowNoIcons=yes
LicenseFile=..\..\LICENSE
OutputDir={#OutputDir}
OutputBaseFilename={#OutputBaseName}
SetupIconFile=zoidcraft.ico
UninstallDisplayIcon={app}\zoidcraft.ico
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
ChangesAssociations=yes

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"
Name: "japanese"; MessagesFile: "compiler:Languages\Japanese.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"
Name: "assoc_zcraft"; Description: "Associate .zcraft project files with ZoidCraft"; GroupDescription: "File Associations:"
Name: "assoc_pz"; Description: "Associate .pz project files with ZoidCraft"; GroupDescription: "File Associations:"

[Files]
Source: "{#SourceDir}\zoidcraft.exe"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#SourceDir}\zoidcraft-cli.exe"; DestDir: "{app}"; Flags: ignoreversion
Source: "zoidcraft.ico"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\..\LICENSE"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\..\README.md"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\..\THIRD_PARTY_LICENSES.md"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{group}\ZoidCraft"; Filename: "{app}\zoidcraft.exe"; IconFilename: "{app}\zoidcraft.ico"
Name: "{group}\Uninstall ZoidCraft"; Filename: "{uninstallexe}"
Name: "{autodesktop}\ZoidCraft"; Filename: "{app}\zoidcraft.exe"; IconFilename: "{app}\zoidcraft.ico"; Tasks: desktopicon

[Registry]
Root: HKA; Subkey: "Software\Classes\.zcraft"; ValueType: string; ValueName: ""; ValueData: "ZoidCraft.Project"; Flags: uninsdeletevalue; Tasks: assoc_zcraft
Root: HKA; Subkey: "Software\Classes\ZoidCraft.Project"; ValueType: string; ValueName: ""; ValueData: "ZoidCraft Project"; Flags: uninsdeletekey; Tasks: assoc_zcraft
Root: HKA; Subkey: "Software\Classes\ZoidCraft.Project\DefaultIcon"; ValueType: string; ValueName: ""; ValueData: "{app}\zoidcraft.ico,0"; Tasks: assoc_zcraft
Root: HKA; Subkey: "Software\Classes\ZoidCraft.Project\shell\open\command"; ValueType: string; ValueName: ""; ValueData: """{app}\zoidcraft.exe"" ""%1"""; Tasks: assoc_zcraft

Root: HKA; Subkey: "Software\Classes\.pz"; ValueType: string; ValueName: ""; ValueData: "ZoidCraft.PzProject"; Flags: uninsdeletevalue; Tasks: assoc_pz
Root: HKA; Subkey: "Software\Classes\ZoidCraft.PzProject"; ValueType: string; ValueName: ""; ValueData: "ZoidCraft .pz Project"; Flags: uninsdeletekey; Tasks: assoc_pz
Root: HKA; Subkey: "Software\Classes\ZoidCraft.PzProject\DefaultIcon"; ValueType: string; ValueName: ""; ValueData: "{app}\zoidcraft.ico,0"; Tasks: assoc_pz
Root: HKA; Subkey: "Software\Classes\ZoidCraft.PzProject\shell\open\command"; ValueType: string; ValueName: ""; ValueData: """{app}\zoidcraft.exe"" ""%1"""; Tasks: assoc_pz

[Run]
Filename: "{app}\zoidcraft.exe"; Description: "{cm:LaunchProgram,ZoidCraft}"; Flags: nowait postinstall skipifsilent
