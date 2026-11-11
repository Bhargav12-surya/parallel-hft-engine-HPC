# Quick start

The default simulation uses 16 symbols, seed 42 and 10,000 events per symbol. Prices, quantities and cash are integer units; a tick defaults to 0.01 currency units. GUI P&L values explicitly labeled ticks must be multiplied by tick size for currency.

1. Open **Market**, press Run, observe candles, depth, the latest strategy fill and symbol heatmap.
2. Open **Strategy lab**, choose any of the 13 strategies. Window, quantity and position controls cycle through safe values; edit the supplied JSON for other parameters.
3. Open **Engines**, select an available backend and adjust symbols, events, workers, batch or OpenMP schedule.
4. **Results** shows the completed portfolio curve and measured metrics. **Export report pack** writes to `results/`.
5. **Cloud & AI** trains an MLP or Q table on a background thread. Export afterwards to save `results/model.bin`. Load model reads this path. OpenCL MLP training is used when OpenCL is selected; the desktop Q button uses the CPU/OpenMP episode trainer. CLI S6 can run OpenCL Q training.
6. **Settings → Load results/config.json** imports a user-edited configuration. The CLI accepts any config path.

Keyboard: Space starts/pauses, Escape cancels, 1–8 changes screens, P captures PNG, F11 toggles timed presentation navigation. Presentation mode cycles screens; it is not exclusive fullscreen mode.

Run/pause/stop are for the active job. MPI cancellation terminates the owned worker process group/job. Training checks cancellation between epochs/episodes; benchmark suite cancellation is checked between scenarios. Closing the desktop waits for the current cancellation boundary. Avoid very large workloads for a live demonstration.

Only symbol 0 has the detailed chart in this compact desktop; the heatmap includes up to 64 symbols. Full configured symbol counts are simulated. The displayed tape is the latest strategy fill, not a complete exchange trade feed.

Never launch directly inside a ZIP viewer: exports need a writable working directory. The bundled Windows executables use the CPU backends. Install optional SDKs and rebuild to enable MPI/OpenCL/NVIDIA.
