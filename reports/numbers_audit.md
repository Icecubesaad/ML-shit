# Numbers audit — v2 deliverables

Source: v2 CSVs; targets: `artifacts/reports/DM-AHR-ViT_Skin_Lesion_Classification_FINAL.docx` and `reports/real_hair_validation/audit_v2_integration.md`.

- Checks run: 41
- Passed: 41
- Failed: 0


## Passed checks

- docx: macro-F1 full/A = `0.751`
- report: report macro-F1 full/A = `0.751`
- docx: macro-F1 full/B = `0.732`
- report: report macro-F1 full/B = `0.732`
- docx: macro-F1 full/Astar = `0.740`
- report: report macro-F1 full/Astar = `0.740`
- docx: macro-F1 full/Cv2 = `0.721`
- report: report macro-F1 full/Cv2 = `0.721`
- docx: macro-F1 hair_present/A = `0.614`
- report: report macro-F1 hair_present/A = `0.614`
- docx: macro-F1 hair_present/B = `0.567`
- report: report macro-F1 hair_present/B = `0.567`
- docx: macro-F1 hair_present/Astar = `0.578`
- report: report macro-F1 hair_present/Astar = `0.578`
- docx: macro-F1 hair_present/Cv2 = `0.490`
- report: report macro-F1 hair_present/Cv2 = `0.490`
- docx: diff full/A-B = `0.019`
- docx: holm full/A-B = `0.550`
- docx: diff full/A-Cv2 = `0.030`
- docx: holm full/A-Cv2 = `0.406`
- docx: diff full/B-Cv2 = `0.011`
- docx: holm full/B-Cv2 = `0.550`
- docx: diff hair_present/A-Cv2 = `0.125`
- docx: holm hair_present/A-Cv2 = `0.219`
- docx: PSNR 128px/hairy = `29.25`
- docx: PSNR 128px/B = `37.81`
- docx: PSNR 128px/Cv2 = `35.18`
- docx: PSNR native/hairy = `25.73`
- docx: PSNR native/B = `34.82`
- docx: PSNR native/Cv2 = `32.96`
- docx: XAI gradcam_ln1 A = `0.160`
- docx: XAI gradcam_ln1 B = `0.121`
- docx: XAI gradcam_ln1 Astar = `0.163`
- docx: XAI gradcam_ln1 Cv2 = `0.199`
- docx: 8seed diff hair_present/A-Cv2 = `0.127`
- docx: 8seed holm hair_present/A-Cv2 = `0.111`
- docx: 8seed diff hair_present/B-Cv2 = `0.090`
- docx: 8seed holm hair_present/B-Cv2 = `0.111`
- docx: DERMAHAIR PSNR = `25.50`
- docx: research-only statement = `Research prototype only`
- docx: no significance claim = `no comparison reaches Holm p < 0.05`
