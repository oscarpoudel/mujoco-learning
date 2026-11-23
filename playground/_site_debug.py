"""Why isn't usercustomize.py taking effect in this interpreter?

Checks each link in the chain: file exists -> directory on sys.path -> site
module actually imports it -> PATH mutation happens.
"""

import os
import site
import sys

print("sys.executable :", sys.executable)
print("no_user_site   :", sys.flags.no_user_site)
print("no_site        :", sys.flags.no_site)
print("ENABLE_USER_SITE:", site.ENABLE_USER_SITE)

sp = site.getsitepackages()
print("\nsite-packages dirs:")
for p in sp:
    print(f"  {p}  exists={os.path.isdir(p)}")

uc = os.path.join(sp[0], "usercustomize.py")
print(f"\n{uc}\n  exists = {os.path.exists(uc)}")
if os.path.exists(uc):
    text = open(uc, encoding="utf-8-sig").read()
    print(f"  bytes  = {len(text)}")
    print(f"  contains 'shims' = {'shims' in text}")
    print("  first 200 chars:")
    for line in text[:200].splitlines():
        print("    " + line)

print("\nimported as a module? 'usercustomize' in sys.modules:", "usercustomize" in sys.modules)

# If site did import it, PATH must already mention the shim dir.
print("\nPATH mentions shims:", "shims" in os.environ.get("PATH", ""))

# Try importing it by hand to surface a swallowed exception.
try:
    import usercustomize  # noqa: F401

    print("manual import: OK")
    print("  PATH mentions shims after manual import:",
          "shims" in os.environ.get("PATH", ""))
except Exception as exc:  # noqa: BLE001
    print(f"manual import: {type(exc).__name__}: {exc}")
