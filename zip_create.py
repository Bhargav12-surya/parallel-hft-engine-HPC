import zipfile, pathlib, sys

src = pathlib.Path(r'D:\HPC')
out = pathlib.Path(r'D:\hpc_full.zip')
with zipfile.ZipFile(out, 'w', zipfile.ZIP_DEFLATED) as zf:
    for file in src.rglob('*'):
        if file.is_file():
            zf.write(file, file.relative_to(src))
print(f'ZIP created at {out}')
