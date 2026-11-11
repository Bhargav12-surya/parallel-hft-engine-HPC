# Third-party components

- cJSON 1.7.19, vendored as `vendor/cJSON.c` and `vendor/cJSON.h`; its MIT notice is `vendor/cJSON-LICENSE`.
- Headless-preview bitmap glyphs are derived from DejaVu Sans Mono. The Bitstream Vera / DejaVu notices are in `vendor/DejaVu-LICENSE`. Windows uses an installed system font instead.
- Bundled Windows GCC/OpenMP/pthread support DLLs retain their GCC Runtime Library Exception and MinGW notices in `vendor/GCC-RUNTIME-LICENSE`, `vendor/GPL-3` and `vendor/MinGW-LICENSE`.
- Optional OpenMP, MPI, OpenCL and NVIDIA headers/libraries are discovered from the user's installed SDK/runtime. They are not redistributed in the source package. Vendor implementation internals may use other languages; this project's authored source and build language are C.

The bundled Windows support DLLs should remain beside the executables. `validation/windows-imports.log` records the imported DLL names.
