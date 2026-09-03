# Missing random annotation batches

The repository provides 800 rare/active-learning Label Studio records. Earlier
work refers to 1,400 total annotations, so approximately 600 corrected random
pilot/round records are not committed.

Copy each original Label Studio export here, then import it from the project
root, using a batch name beginning with `random_`:

```bat
scripts\ie\00_import_additional_annotations.bat data\aviation\annotations\pilot.json random_pilot
```

Do not reconstruct annotations from model predictions. The final split guard
requires a random batch so active-learning examples cannot leak into the
representative test/dev population.
