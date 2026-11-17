# Screening audit (Phase 2)

Detector: the project's frozen DullRazor-style black-hat/top-hat screen, coverage threshold 1%,
config hash e232efe4705133a1...

- Test set: **186** hair-free images out of 1502 (**87.6% flagged hair-present**).
- Whole cohort: **8993** hair-present out of 10015 (89.8%).
- Full breakdown by split and class: `screening_audit.csv`.

This is the rate produced by the current detector. It is not a ground-truth hair rate: on 500 real
annotated masks in the external validation the same detector had image-level specificity 0.0175,
so it over-flags real hair. The rate is reported here for the first time and used to describe which
images enter arms A*, B and Cv2.
