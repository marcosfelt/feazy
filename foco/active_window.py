#!/usr/bin/env python

"""Find the currently active window."""

import logging
import sys
from time import sleep
from typing import Dict
from Foundation import NSAppleScript

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.DEBUG,
                    stream=sys.stdout)

source = """
global frontApp, frontAppName, windowTitle
set windowTitle to ""
tell application "System Events"
    set frontApp to first application process whose frontmost is true
    set frontAppName to name of frontApp
    tell process frontAppName
        try
            tell (1st window whose value of attribute "AXMain" is true)
                set windowTitle to value of attribute "AXTitle"
            end tell
        end try
    end tell
end tell
return frontAppName & "
" & windowTitle
"""

script = None


def getInfo() -> Dict[str, str]:
    # Cache compiled script
    global script
    if script is None:
        script = NSAppleScript.alloc().initWithSource_(source)

    # Call script
    result, errorinfo = script.executeAndReturnError_(None)
    if errorinfo:
        raise Exception(errorinfo)
    output = result.stringValue()

    # Ensure there's no extra newlines in the output
    assert len(output.split("\n")) == 2

    app = getApp(output)
    title = getTitle(output)

    return {"app": app, "title": title}


def getApp(info: str) -> str:
    return info.split('\n')[0]  


def getTitle(info: str) -> str:
    return info.split('\n')[1]


def get_active_window():
    """
    Get the currently active window.

    Returns
    -------
    string :
        Name of the currently active window.
    """
    import sys
    active_application = None
    active_window_name = None
    if sys.platform in ['linux', 'linux2']:
        # Alternatives: https://unix.stackexchange.com/q/38867/4784
        try:
            import wnck
        except ImportError:
            logging.info("wnck not installed")
            wnck = None
        if wnck is not None:
            screen = wnck.screen_get_default()
            screen.force_update()
            window = screen.get_active_window()
            if window is not None:
                pid = window.get_pid()
                with open("/proc/{pid}/cmdline".format(pid=pid)) as f:
                    active_window_name = f.read()
        else:
            try:
                from gi.repository import Gtk, Wnck
                gi = "Installed"
            except ImportError:
                logging.info("gi.repository not installed")
                gi = None
            if gi is not None:
                Gtk.init([])  # necessary if not using a Gtk.main() loop
                screen = Wnck.Screen.get_default()
                screen.force_update()  # recommended per Wnck documentation
                active_window = screen.get_active_window()
                pid = active_window.get_pid()
                with open("/proc/{pid}/cmdline".format(pid=pid)) as f:
                    active_window_name = f.read()
    elif sys.platform in ['Windows', 'win32', 'cygwin']:
        # https://stackoverflow.com/a/608814/562769
        import win32gui
        window = win32gui.GetForegroundWindow()
        active_window_name = win32gui.GetWindowText(window)
    elif sys.platform in ['Mac', 'darwin', 'os2', 'os2emx']:
        # https://stackoverflow.com/a/373310/562769
        from AppKit import NSWorkspace
        from Quartz import (
            CGWindowListCopyWindowInfo,
            kCGWindowListOptionOnScreenOnly,
            kCGWindowListExcludeDesktopElements,
            kCGNullWindowID
        )
        app = NSWorkspace.sharedWorkspace().activeApplication()
        active_application = app['NSApplicationName']
        active_window_name = getInfo()["title"]
        # active_application = (NSWorkspace.sharedWorkspace()
        #                       .activeApplication()['NSApplicationName'])
    else:
        print("sys.platform={platform} is unknown. Please report."
              .format(platform=sys.platform))
        print(sys.version)
    return active_application, active_window_name

if __name__ == "__main__":
    active = True
    while active:
        try:
            active_app, active_window = get_active_window()
            print(f"Active window: {active_app} - {active_window}")
            sleep(5)
        except KeyboardInterrupt:
            print("Do you want to exit (y/n)?:")
            response = input()
            if response == "y":
                active = False