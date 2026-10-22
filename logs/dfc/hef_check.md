# Hailo-10H 호환성 확인: CPU 로 넘어가는 층이 있나

- 파싱 호출: `translate_onnx_model(onnx, NET)` — start/end 노드를 지정하지 않음 (ONNX 전체를 변환)

## H-official (공식)

- HAR: `clip_vit_b_16_image_encoder.compiled.har` 있음 / HEF: `clip_vit_b_16_image_encoder.hailo10h.hef` 있음 (79.3 MB)
- 층 336개, 종류 17가지
- 입력 층: {'clip_vit_b_16_image_encoder/input_layer1': [[-1, 224, 224, 3]]}
- 출력 층: {'clip_vit_b_16_image_encoder/output_layer1': [[-1, 1, 1, 512]]}
- 호스트/후처리로 보이는 층 종류: 없음
- 정규화 층 존재(입력 정규화가 칩 안): 예

| 층 종류 | 개수 |
|---|---|
| conv | 74 |
| ew_sub | 38 |
| reduce_sum | 38 |
| ew_mult | 38 |
| activation | 26 |
| reduce_mean | 26 |
| feature_multiplier | 26 |
| ew_add | 25 |
| matmul | 24 |
| reduce_max | 12 |
| const_input | 2 |
| normalization | 2 |
| input_layer | 1 |
| space_to_depth | 1 |
| concat | 1 |
| slice | 1 |
| output_layer | 1 |
- hailortcli 없음 (HailoRT 미설치) → HEF 구조는 보드에서 본선 때 확인

## H-splf-b7-k32 (동결)

- HAR: `clip_vit_b_16_image_encoder.splf_b7_k32.compiled.har` 있음 / HEF: `clip_vit_b_16_image_encoder.splf_b7_k32.hailo10h.hef` 있음 (79.6 MB)
- 층 339개, 종류 17가지
- 입력 층: {'clip_vit_b_16_image_encoder/input_layer1': [[-1, 224, 224, 3]]}
- 출력 층: {'clip_vit_b_16_image_encoder/output_layer1': [[-1, 1, 1, 512]]}
- 호스트/후처리로 보이는 층 종류: 없음
- 정규화 층 존재(입력 정규화가 칩 안): 예

| 층 종류 | 개수 |
|---|---|
| conv | 76 |
| ew_sub | 38 |
| reduce_sum | 38 |
| ew_mult | 38 |
| activation | 26 |
| ew_add | 26 |
| reduce_mean | 26 |
| feature_multiplier | 26 |
| matmul | 24 |
| reduce_max | 12 |
| const_input | 2 |
| normalization | 2 |
| input_layer | 1 |
| space_to_depth | 1 |
| concat | 1 |
| slice | 1 |
| output_layer | 1 |
- hailortcli 없음 (HailoRT 미설치) → HEF 구조는 보드에서 본선 때 확인

## H-fc2b7 (참고)

- HAR: `clip_vit_b_16_image_encoder.coco64_fc2b7.compiled.har` 있음 / HEF: `clip_vit_b_16_image_encoder.coco64_fc2b7.hailo10h.hef` 있음 (82.0 MB)
- 층 335개, 종류 17가지
- 입력 층: {'clip_vit_b_16_image_encoder/input_layer1': [[-1, 224, 224, 3]]}
- 출력 층: {'clip_vit_b_16_image_encoder/output_layer1': [[-1, 1, 1, 512]]}
- 호스트/후처리로 보이는 층 종류: 없음
- 정규화 층 존재(입력 정규화가 칩 안): 예

| 층 종류 | 개수 |
|---|---|
| conv | 74 |
| ew_sub | 38 |
| reduce_sum | 38 |
| ew_mult | 38 |
| activation | 26 |
| reduce_mean | 26 |
| feature_multiplier | 26 |
| ew_add | 24 |
| matmul | 24 |
| reduce_max | 12 |
| const_input | 2 |
| normalization | 2 |
| input_layer | 1 |
| space_to_depth | 1 |
| concat | 1 |
| slice | 1 |
| output_layer | 1 |
- hailortcli 없음 (HailoRT 미설치) → HEF 구조는 보드에서 본선 때 확인

