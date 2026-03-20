; NSIS installer script for NDViewer Light

!include "MUI2.nsh"

; General settings
Name "NDViewer Light Setup"
OutFile "NDViewerLight-Setup.exe"
InstallDir "$PROGRAMFILES\NDViewer Light"
InstallDirRegKey HKLM "Software\NDViewer Light" "InstallDir"
RequestExecutionLevel admin
SetCompressor lzma

; Interface settings
!define MUI_ABORTWARNING

; Pages
!insertmacro MUI_PAGE_WELCOME
!insertmacro MUI_PAGE_DIRECTORY
!insertmacro MUI_PAGE_COMPONENTS
!insertmacro MUI_PAGE_INSTFILES
!insertmacro MUI_PAGE_FINISH

; Uninstaller pages
!insertmacro MUI_UNPAGE_CONFIRM
!insertmacro MUI_UNPAGE_INSTFILES

; Language
!insertmacro MUI_LANGUAGE "English"

; --------------------
; Installer Sections
; --------------------

Section "NDViewer Light (required)" SecMain
    SectionIn RO

    ; Install all files from the PyInstaller dist output
    SetOutPath "$INSTDIR"
    File /r "..\dist\ndviewer_light\*.*"

    ; Write uninstaller
    WriteUninstaller "$INSTDIR\Uninstall.exe"

    ; Create Start Menu shortcut
    CreateDirectory "$SMPROGRAMS\NDViewer Light"
    CreateShortcut "$SMPROGRAMS\NDViewer Light\NDViewer Light.lnk" "$INSTDIR\ndviewer_light.exe"
    CreateShortcut "$SMPROGRAMS\NDViewer Light\Uninstall.lnk" "$INSTDIR\Uninstall.exe"

    ; Register in Add/Remove Programs
    WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\NDViewer Light" \
        "DisplayName" "NDViewer Light"
    WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\NDViewer Light" \
        "UninstallString" "$\"$INSTDIR\Uninstall.exe$\""
    WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\NDViewer Light" \
        "InstallLocation" "$INSTDIR"
    WriteRegDWORD HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\NDViewer Light" \
        "NoModify" 1
    WriteRegDWORD HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\NDViewer Light" \
        "NoRepair" 1

    ; Store install directory
    WriteRegStr HKLM "Software\NDViewer Light" "InstallDir" "$INSTDIR"
SectionEnd

Section "Desktop Shortcut" SecDesktop
    CreateShortcut "$DESKTOP\NDViewer Light.lnk" "$INSTDIR\ndviewer_light.exe"
SectionEnd

; Section descriptions
!insertmacro MUI_FUNCTION_DESCRIPTION_BEGIN
    !insertmacro MUI_DESCRIPTION_TEXT ${SecMain} "Install NDViewer Light application files."
    !insertmacro MUI_DESCRIPTION_TEXT ${SecDesktop} "Create a shortcut on the Desktop."
!insertmacro MUI_FUNCTION_DESCRIPTION_END

; --------------------
; Uninstaller Section
; --------------------

Section "Uninstall"
    ; Remove files and directories
    RMDir /r "$INSTDIR"

    ; Remove Start Menu shortcuts
    RMDir /r "$SMPROGRAMS\NDViewer Light"

    ; Remove Desktop shortcut
    Delete "$DESKTOP\NDViewer Light.lnk"

    ; Remove registry entries
    DeleteRegKey HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\NDViewer Light"
    DeleteRegKey HKLM "Software\NDViewer Light"
SectionEnd
