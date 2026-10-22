# Spécification PyInstaller : un dossier autonome (et un .app sur macOS), sans Python requis.
# Construire avec `make package` depuis molonaviz2/. Un build par système d'exploitation.
import sys
import tomllib
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_submodules, copy_metadata

ROOT = Path(SPECPATH).parent
NAME = "Molonaviz2"
VERSION = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["version"]  # source unique

datas = collect_data_files("molonaviz2")
hiddenimports = ["h5netcdf", "h5py", "scipy.special.cython_special"]
for package in ("arviz", "arviz_base", "arviz_stats", "arviz_plots", "xarray_einstats"):
    datas += collect_data_files(package)
    hiddenimports += collect_submodules(package)
# molonaviz2 : sa version est lue par importlib.metadata, et entre dans engine_version() et les twin_id.
for package in ("molonaviz2", "xarray", "arviz", "arviz_base", "arviz_stats", "arviz_plots", "jax", "jaxlib", "h5netcdf", "numpy", "pandas", "scipy", "pyqtgraph"):
    datas += copy_metadata(package)
datas += collect_data_files("xarray")

a = Analysis(
    [str(ROOT / "src" / "molonaviz2" / "__main__.py")],
    pathex=[str(ROOT / "src")],
    datas=datas,
    hiddenimports=hiddenimports,
    excludes=["pyheatmy", "numba", "llvmlite", "IPython", "pytest", "nbformat", "nbconvert", "tkinter", "PyQt5", "PyQt6", "jupyter"],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name=NAME, console=False, argv_emulation=False)
coll = COLLECT(exe, a.binaries, a.datas, name=NAME)
if sys.platform == "darwin":
    app = BUNDLE(coll, name=f"{NAME}.app", bundle_identifier="fr.molonari.molonaviz2",
                 info_plist={"NSHighResolutionCapable": True, "CFBundleShortVersionString": VERSION.split("-")[0].split("+")[0]})
