# Weather data acquisition review

Reviewer verdict: approved for this January acquisition milestone and incremental expansion through the same schema, subject to the documented coverage and final data-audit results. No unresolved critical or major code findings.

Reviewed scripts/data/acquire_weather.py, scripts/data/inspect_weather.py and tests/test_weather_acquisition.py under the repository verification instructions.

Resolved findings:

- Decoded GRIB initialization, lead and valid time are checked before coordinate reshaping; control number must be zero before relabeling.
- Physical unit whitelists and the ARCO final-ERA5 cutoff prevent unnoticed unit changes or preliminary ERA5T ingestion.
- Control, mean50, CRPS and spread diagnostics now use an identical finite-member/reference mask and a declared ERA5-wave ocean mask. Empty eligible sets fail.
- Running-code hash uncertainty during pilot development is disclosed; future acquisitions capture a startup script hash, and final validation records separate audit hashes.

Atomic publication and resume: per-target advisory locks serialize writers. A crash between NetCDF and receipt publication leaves an unverifiable pair, which resume rebuilds. Resume requires exact request, hash and content checks. Source legacy files are read-only. HTTP ranges verify status, length, Content-Range and consistent source object version. The 50-member empirical CRPS formula and tie-aware 51-bin ranks were checked.

Added an end-to-end mocked decode regression: wrong raw GRIB date must fail through decode before requested coordinates can overwrite it. Targeted verification:

```sh
PYTHONPATH=. .venv/bin/python -m pytest tests/test_weather_acquisition.py tests/test_forecast_data_downloads.py -q
```

Result: 14 passed in 0.49 seconds. An existing netCDF4/NumPy import warning is narrowly filtered in these tests; no dependency changes were made.

Scientific limits: this is a sparse January ensemble pilot rather than annual completeness or calibration certification. Historical forecast availability remains unknown and must not be equated with initialization. ERA5 is a related reference, not observed truth. Scalar mean wind speed in diagnostics differs from magnitude of mean wind components. Independent observation checks and seasonal/extreme-event replication remain future work. Final actual-data validation and explicit coverage reporting remain required before presenting results.
