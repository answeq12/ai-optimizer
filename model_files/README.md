# 제출 모델 파일

보고서 8장 2·3번의 베이스라인·최적화 모델 파일이다. 지문은 `SHA256SUMS_models.txt`.

| 파일 | 내용 | 크기 |
|---|---|---|
| `clip_vitb16_image_fp32.onnx` (part0~3로 나눠 저장) | 베이스라인: OpenAI CLIP ViT-B/16 이미지 인코더, FP32 | 344.9 MB |
| `DA5-SPL-b567-k32.onnx` | **최종 최적화 모델**: 도메인 적응 가중치 + SPL(블록 5·6·7, 32채널) 정적 INT8 | 89.8 MB |
| `DA5_text_state.pt` | 최종 모델과 짝인 적응 텍스트 인코더(검색 평가에 필요) | 51.5 MB |
| `SPL-b567-k32.onnx` | 참고: 원본 가중치 + SPL | 89.8 MB |
| `clip_vit_b_16_image_encoder.splf_b7_k32.hailo10h.hef` | 참고: Hailo-10H 컴파일 결과(원본 가중치) | 79.6 MB |

포맷: ONNX opset 17, ONNX Runtime 1.23.2 (QDQ) / HEF: Hailo DFC 5.4.0, hailo10h

## 베이스라인 파일 합치기

GitHub 파일 크기 제한(100 MB) 때문에 베이스라인만 4조각으로 나눠 올렸다.

```bash
cd model_files
cat clip_vitb16_image_fp32.onnx.part0 clip_vitb16_image_fp32.onnx.part1 \
    clip_vitb16_image_fp32.onnx.part2 clip_vitb16_image_fp32.onnx.part3 > clip_vitb16_image_fp32.onnx
sha256sum -c SHA256SUMS_models.txt
```

## 이 파일로 바로 평가하기

`bash reproduce.sh` 를 한 번 실행해 데이터와 텍스트 임베딩을 준비한 뒤(보고서 9장), 같은 터미널에서:

```bash
python -m src.coco_retrieval eval --model model_files/clip_vitb16_image_fp32.onnx --code SUB-A0 --split confirm --i-know-final
HALIO_MODELS_SUBDIR=models_da5 python -m src.coco_retrieval eval \
  --model model_files/DA5-SPL-b567-k32.onnx --code SUB-DA5-SPL --split confirm --i-know-final
```

최종 모델은 이미지·텍스트 인코더를 함께 적응했으므로, 검색 평가는 `models_da5` 의 적응 텍스트 임베딩(reproduce.sh 가 만들거나 `DA5_text_state.pt`)과 짝으로 써야 보고서 수치가 나온다.
